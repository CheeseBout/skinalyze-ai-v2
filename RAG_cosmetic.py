"""
RAG Cosmetic Chatbot Core - Stateless for NestJS Backend Integration
Merged Features: 
- Stateless design (NestJS handles session/history)
- Advanced Skin Condition Detection
- Currency Conversion (USD -> VND)
- Smart Product Grouping & Filtering (Improved)
- VLM Skin Analysis (Base64/Bytes support)
- Improved chunk_size and format_docs logic
"""

import os
import re
from pathlib import Path
import torch
from PIL import Image
import google.generativeai as genai
import base64
import io
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# CONFIGURATION
PATH = Path(__file__).parent.resolve()
CHUNKS_FILE = PATH / "data" / "product_chunks.txt"
PERSIST_DIRECTORY = PATH / "db_chroma"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Tỷ giá USD → VND (cố định)
USD_TO_VND = 26349

# Global cache for embeddings
_CACHED_EMBEDDINGS = None

# DATA MAPPING - EXTENDED
SKIN_CONDITION_TO_SKIN_TYPE = {
    #ƯU TIÊN Từ khóa DÀI/CỤ THỂ trước, NGẮN/CHUNG sau
    "warts": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    "mụn cóc": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    "cóc": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    
    "mụn trứng cá": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
    
    "acne": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
    "mụn": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
    
    "actinic keratosis": ["Khô", "Thường"],
    "da dày sừng": ["Khô", "Thường"],
    "dày sừng": ["Khô", "Thường"],
    
    "drug eruption": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    "phát ban do thuốc": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    "phát ban thuốc": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    
    "eczema": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    "chàm": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    "viêm da": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    
    "psoriasis": ["Khô"],
    "vảy nến": ["Khô"],
    
    "rosacea": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
    "trứng cá đỏ": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
    "da đỏ": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
    
    "seborrheic keratoses": ["Thường", "Dầu", "Nhạy cảm"],
    "viêm da tiết bã": ["Thường", "Dầu", "Nhạy cảm"],
    
    "sun damage": ["Hỗn hợp", "Khô", "Thường", "Nhạy cảm"],
    "hư tổn do nắng": ["Hỗn hợp", "Khô", "Thường", "Nhạy cảm"],
    "tổn thương nắng": ["Hỗn hợp", "Khô", "Thường", "Nhạy cảm"],
    
    "tinea": ["Hỗn hợp", "Dầu"],
    "nấm da": ["Hỗn hợp", "Dầu"],
    "nấm": ["Hỗn hợp", "Dầu"],
}

# HELPER FUNCTIONS
def setup_api_key():
    """Setup Google API Key"""
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("\n❌ CRITICAL ERROR: GOOGLE_API_KEY not found in environment variables.")
        print("Please create a .env file and add GOOGLE_API_KEY=your_new_key")
        raise ValueError("GOOGLE_API_KEY is missing.")
    
    genai.configure(api_key=api_key)
    print("✅ API Key configured successfully from environment!\n")

def extract_product_name(chunk_text):
    """Trích xuất tên sản phẩm từ chunk text"""
    match = re.search(r'Product Name:\s*(.+?)(?:\n|$)', chunk_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'Tên sản phẩm:\s*(.+?)(?:\n|$)', chunk_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    lines = chunk_text.split('\n')
    for line in lines:
        if ':' in line:
            potential_name = line.split(':', 1)[1].strip()
            if len(potential_name) > 5:
                return potential_name
    return "Unknown Product"

def extract_field_from_chunk(chunk_text, field_name):
    """Trích xuất giá trị của field từ chunk text"""
    pattern = rf'{field_name}:\s*(.+?)(?:\n|$)'
    match = re.search(pattern, chunk_text, re.IGNORECASE)
    
    if match:
        value = match.group(1).strip()
        value = value.replace('---', '').strip()
        if value and value != 'N/A':
            return value
    return None

def convert_price_in_text(text):
    """Tìm và chuyển đổi giá USD sang VND trong text"""
    def replace_price(match):
        price_str = match.group(1)
        try:
            price_usd = float(price_str)
            price_vnd = int(price_usd * USD_TO_VND)
            return f"${price_usd:.0f} (≈ {price_vnd:,} VND)".replace(',', '.')
        except:
            return match.group(0)
    
    result = re.sub(r'\$([0-9]+(?:\.[0-9]+)?)', replace_price, text)
    return result

def detect_skin_condition_and_types(query):
    """
    Phát hiện bệnh da trong câu hỏi và trả về loại da phù hợp
    ⚠️ ƯU TIÊN: Kiểm tra từ khóa DÀI trước (mụn cóc) rồi mới đến NGẮN (mụn)
    """
    query_lower = query.lower()
    
    # Sắp xếp theo độ dài từ khóa (dài -> ngắn)
    sorted_conditions = sorted(
        SKIN_CONDITION_TO_SKIN_TYPE.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )
    
    for condition, skin_types in sorted_conditions:
        if condition in query_lower:
            return condition, skin_types
    
    return None, None

# VECTOR STORE
def load_or_create_vectorstore():
    """Load or create vector store with improved error handling"""
    global _CACHED_EMBEDDINGS
    
    print("=" * 80)
    print("📚 KHỞI TẠO VECTOR STORE")
    print("=" * 80)
    
    try:
        # Load Embedding Model
        if _CACHED_EMBEDDINGS is not None:
            print(f"\n⚡ Sử dụng cached embedding model")
            embeddings = _CACHED_EMBEDDINGS
        else:
            print(f"\n⏳ Đang tải embedding model: {MODEL_NAME}...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"   🖥️ Sử dụng thiết bị: {device}")
            
            embeddings = HuggingFaceEmbeddings(
                model_name=MODEL_NAME,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            _CACHED_EMBEDDINGS = embeddings
            print("✅ Đã tải embedding model!\n")

        # Check existing DB
        db = None
        should_create_new = True
        
        if os.path.exists(PERSIST_DIRECTORY) and len(os.listdir(PERSIST_DIRECTORY)) > 0:
            print(f"📂 Thư mục DB tồn tại: {PERSIST_DIRECTORY}")
            try:
                db = Chroma(
                    persist_directory=str(PERSIST_DIRECTORY),
                    embedding_function=embeddings
                )
                count = db._collection.count() if db._collection else 0
                
                if count > 0:
                    print(f"✅ Đã load thành công {count} documents")
                    should_create_new = False
                else:
                    print(f"⚠️  DB tồn tại nhưng RỖNG (0 docs). Sẽ tiến hành tạo lại...")
            except Exception as e:
                print(f"⚠️  Lỗi khi đọc DB cũ: {e}. Sẽ tạo mới...")
        
        # Create new if needed
        if should_create_new:
            print(f"🆕 Bắt đầu tạo mới Vector Store từ {CHUNKS_FILE.name}...\n")
            
            if not CHUNKS_FILE.exists():
                raise FileNotFoundError(f"❌ KHÔNG TÌM THẤY FILE DATA: {CHUNKS_FILE}")
            
            loader = TextLoader(str(CHUNKS_FILE), encoding='utf-8')
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["---"],
                chunk_size=800,
                chunk_overlap=100,
                length_function=len
            )
            docs = text_splitter.split_documents(documents)
            
            # Add metadata
            for doc in docs:
                product_name = extract_product_name(doc.page_content)
                doc.metadata['product_name'] = product_name

            print(f"   ✓ Split into {len(docs)} chunks")
            print("💾 Creating embeddings and saving...")
            
            # Batch processing
            batch_size = 50
            total_docs = len(docs)
            
            if total_docs == 0:
                print("   ⚠️ Không có chunk nào để thêm vào database.")
                return None, embeddings
            
            db = Chroma.from_documents(
                documents=docs[:batch_size],
                embedding=embeddings,
                persist_directory=str(PERSIST_DIRECTORY)
            )
            
            for i in range(batch_size, total_docs, batch_size):
                batch_end = min(i + batch_size, total_docs)
                print(f"   Processing batch {i}/{total_docs}...", end='\r')
                db.add_documents(docs[i:batch_end])
                
                if device == 'cuda':
                    torch.cuda.empty_cache()
            
            print(f"\n✅ Hoàn tất! Đã lưu {total_docs} vectors vào DB\n")

        return db, embeddings
        
    except Exception as e:
        print(f"\n❌ Error in Vector Store: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# RAG CHAIN SETUP
def setup_rag_chain(db):
    """Setup RAG chain with improved product grouping and metadata extraction"""
    print("\n" + "=" * 80)
    print("⛓️ KHỞI TẠO RAG CHAIN")
    print("=" * 80)
    
    # LLM
    print("\n🤖 [1/3] Connecting to Google Gemini...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.05,
        max_output_tokens=1500,
        convert_system_message_to_human=True,
        request_options={"timeout": 90},
        max_retries=3
    )
    print("   ✓ Connected to Gemini 2.0 Flash")
    
    # Retriever
    print("🔍 [2/3] Creating Retriever...")
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 30
        }
    )
    print("   ✓ Retriever ready (Similarity Search, k=30)")
    
    # Prompt Template
    print("📝 [3/3] Creating Prompt Template...")
    template = """Bạn là chuyên gia tư vấn mỹ phẩm chuyên nghiệp, thân thiện và hiểu tâm lý khách hàng.

PHÂN LOẠI CÂU HỎI VÀ CÁCH TRẢ LỜI:

🔹 **CHÀO HỎI/GIAO TIẾP CƠ BẢN**
Câu hỏi: "xin chào", "hi", "hello"
→ "Chào bạn! 👋 Mình là trợ lý tư vấn mỹ phẩm. Bạn muốn tìm sản phẩm gì hôm nay? 😊"

🔹 **HỎI VỀ BỆNH DA (ƯU TIÊN CAO)**
Câu hỏi: "tôi bị mụn", "chàm", "vảy nến", "nấm da"...
→ ĐI THẲNG VÀO: "Dạ, mình gợi ý sản phẩm cho [bệnh da] nhé:
   [LIỆT KÊ 2-3 SẢN PHẨM NGAY]"
   
⚠️ QUAN TRỌNG: 
• CHỈ trả lời về bệnh da được NÊU TRONG CÂU HỎI HIỆN TẠI
• KHÔNG được nhắc lại hoặc nhầm lẫn với câu hỏi trước
• KHÔNG cần giải thích dài dòng

🔹 **HỎI VỀ VẤN ĐỀ DA/LOẠI SẢN PHẨM**
Câu hỏi: "da khô", "da dầu", "kem dưỡng", "serum"...
→ Gợi ý 2-3 sản phẩm PHÙ HỢP từ database.

🔹 **HỎI GIÁ/MUA Ở ĐÂU**
→ "Xin lỗi, mình chỉ tư vấn về sản phẩm. Bạn có thể mua tại store chính hãng. Mình tư vấn thêm sản phẩm khác nhé? 😊"

---

**CHÚ Ý KHI TRẢ LỜI:**
- **NGẮN GỌN - ĐI THẲNG VÀO SẢN PHẨM**
- **⚠️ TẬP TRUNG VÀO CÂU HỎI HIỆN TẠI** - không nhắc lại câu hỏi cũ
- **GROUNDING:** CHỈ GỢI Ý sản phẩm CÓ TRONG DATABASE
- **SỐ LƯỢNG:** Mặc định 2-3 sản phẩm (tối đa 3)
- **⚠️ FORMAT MỖI SẢN PHẨM (BẮT BUỘC ĐIỀN ĐẦY ĐỦ):**
  **Số. Tên sản phẩm của THƯƠNG HIỆU** 
  Giá: XXX.XXX VND | Đánh giá: X/5 | Loại da: [...]
  
  ⚠️ QUAN TRỌNG: 
  • BẮT BUỘC phải có ĐẦY ĐỦ: Tên, Thương hiệu, Giá, Đánh giá, Loại da
  • KHÔNG hiển thị công dụng hoặc thành phần
  • NẾU thiếu thông tin → ghi "(Không có thông tin)"
  • TẤT CẢ sản phẩm phải có FORMAT GIỐNG NHAU
- **LOẠI DA PHẢI DỊCH SANG TIẾNG VIỆT:**
  Combination → Hỗn hợp | Dry → Khô | Normal → Thường | Oily → Dầu | Sensitive → Nhạy cảm
- **CHỈ HIỂN THỊ VND**, KHÔNG USD
- Dùng emoji phù hợp: 😊💄✨💕

THÔNG TIN SẢN PHẨM:
{context}

CÂU HỎI: {question}

TRẢ LỜI (2-3 sản phẩm):"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        """
        Format documents: NHÓM chunks theo product_name,
        trích xuất metadata, sắp xếp theo relevance,
        chọn top 3 sản phẩm với thông tin đầy đủ
        """
        if not docs or len(docs) == 0:
            return "KHÔNG TÌM THẤY SẢN PHẨM TRONG DATABASE"
        
        print(f"   🔍 Tìm được {len(docs)} chunks từ database")
        
        # Nhóm chunks theo product_name
        product_groups = {}
        
        for idx, doc in enumerate(docs):
            product_name = doc.metadata.get('product_name', 'Unknown Product')
            
            if product_name not in product_groups:
                # Trích xuất metadata
                metadata = {
                    'brand': extract_field_from_chunk(doc.page_content, 'Brand'),
                    'category': extract_field_from_chunk(doc.page_content, 'Category'),
                    'suitable_for': extract_field_from_chunk(doc.page_content, 'Suitable for'),
                    'rank': extract_field_from_chunk(doc.page_content, 'Rank'),
                    'price': extract_field_from_chunk(doc.page_content, 'Price')
                }
                
                product_groups[product_name] = {
                    'chunks': [],
                    'first_index': idx,
                    'metadata': metadata,
                    'has_summary': False
                }
            
            # Đánh dấu loại chunk
            if 'chunk type: product summary' in doc.page_content.lower():
                product_groups[product_name]['has_summary'] = True
            
            product_groups[product_name]['chunks'].append(doc)
        
        if not product_groups:
            return "KHÔNG TÌM THẤY SẢN PHẨM TRONG DATABASE"
        
        print(f"   📦 Tìm được {len(product_groups)} sản phẩm khác nhau")
        
        # Lọc sản phẩm có summary
        complete_products = [(name, data) for name, data in product_groups.items() if data['has_summary']]
        
        if not complete_products:
            complete_products = list(product_groups.items())
        
        # Sắp xếp theo relevance
        sorted_products = sorted(complete_products, key=lambda x: x[1]['first_index'])
        
        # Chọn top 3
        num_products = min(3, len(sorted_products))
        selected_products = sorted_products[:num_products]
        
        print(f"   ✅ Chọn {num_products} sản phẩm để tư vấn")
        
        # Format
        formatted = []
        for i, (product_name, data) in enumerate(selected_products, 1):
            chunks = data['chunks']
            metadata = data['metadata']
            
            # Loại bỏ duplicate
            seen_contents = set()
            unique_chunks = []
            for chunk in chunks:
                content_hash = hash(chunk.page_content.strip())
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_chunks.append(chunk)
            
            # Sắp xếp: Summary trước
            def chunk_priority(chunk):
                content = chunk.page_content.lower()
                if 'chunk type: product summary' in content:
                    return 0
                elif 'chunk type: ingredients' in content:
                    return 1
                else:
                    return 2
            
            sorted_chunks = sorted(unique_chunks, key=chunk_priority)
            
            # Gộp thông tin
            product_info = f"{'='*80}\n"
            product_info += f"SẢN PHẨM #{i}: {product_name}\n"
            product_info += f"{'='*80}\n"
            
            # Metadata tổng hợp
            if metadata['brand']:
                product_info += f"🏢 Thương hiệu: {metadata['brand']}\n"
            if metadata['category']:
                product_info += f"📁 Loại: {metadata['category']}\n"
            if metadata['suitable_for']:
                product_info += f"👤 Phù hợp: {metadata['suitable_for']}\n"
            if metadata['rank']:
                product_info += f"⭐ Đánh giá: {metadata['rank']}\n"
            if metadata['price']:
                price_vnd = convert_price_in_text(f"Price: {metadata['price']}")
                product_info += f"💰 {price_vnd}\n"
            
            product_info += f"{'-'*80}\n\n"
            
            # Thêm nội dung chi tiết
            for chunk in sorted_chunks:
                content = chunk.page_content.strip()
                content = convert_price_in_text(content)
                product_info += content + "\n\n"
            
            formatted.append(product_info)
        
        return "\n\n".join(formatted)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("\n✅ RAG Chain ready!\n")
    
    return rag_chain

# VISION ANALYSIS - STATELESS
def analyze_skin_image(image_input, note: str = None):
    """
    Analyze skin image - STATELESS version
    Args:
        image_input: PIL Image, base64 string, bytes, or file path
    Returns:
        str: Analysis result
    """
    try:
        print("\n📸 Analyzing skin image...")
        
        # Handle input types
        img = None
        if isinstance(image_input, str):
            if image_input.startswith('data:image'):
                image_input = image_input.split(',')[1]
            try:
                image_bytes = base64.b64decode(image_input)
                img = Image.open(io.BytesIO(image_bytes))
            except:
                img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        
        if img is None:
            raise ValueError("Invalid image input")

        vision_model = genai.GenerativeModel('gemini-2.0-flash')
        
        vision_prompt = """Bạn là chuyên gia da liễu. Phân tích ảnh da và TÓM TẮT NGẮN GỌN:

1. LOẠI DA: (khô/dầu/hỗn hợp/nhạy cảm/thường)

2. VẤN ĐỀ CHÍNH & MỨC ĐỘ NGHIÊM TRỌNG:
- Nếu có mụn: loại mụn, mức độ (NHẸ/TRUNG BÌNH/NẶNG/RẤT NẶNG)
- Nếu có thâm/sẹo: mức độ, màu sắc
- Nếu có lão hóa: mức độ

3. MỨC ĐỘ CHUNG: Chọn 1 trong 4 (QUAN TRỌNG):
   - NHẸ: Vấn đề nhỏ, tự chăm sóc.
   - TRUNG BÌNH: Cần sản phẩm chuyên dụng.
   - NẶNG: Viêm nhiều, cần điều trị tích cực.
   - RẤT NẶNG: Viêm trầm trọng, sẹo nhiều, CẦN GẶP BÁC SĨ.

4. GỢI Ý: (1 câu ngắn)

Trả lời NGẮN GỌN, bằng tiếng Việt."""

        if note:
            vision_prompt += f"\n\nGhi chú thêm từ người dùng: {note}"
        
        response = vision_model.generate_content([vision_prompt, img])
        analysis = response.text
        print("✅ Analysis complete!")
        return analysis
        
    except Exception as e:
        print(f"❌ Error analyzing image: {str(e)}")
        return None

# HELPER FUNCTIONS FOR NESTJS INTEGRATION
def analyze_with_context(question: str, conversation_history: list = None) -> str:
    """
    Analyze question with conversation context + Skin Condition Logic
    ⚠️ NẾU PHÁT HIỆN BỆNH DA → BỎ QUA CONTEXT để tránh nhầm lẫn
    """
    # Detect skin condition
    detected_condition, suitable_skin_types = detect_skin_condition_and_types(question)
    
    if detected_condition:
        # CÓ BỆNH DA → Không dùng context, chỉ mapping sang tiếng Anh
        skin_types_mapping = {
            "Khô": "Dry",
            "Thường": "Normal",
            "Dầu": "Oily",
            "Nhạy cảm": "Sensitive",
            "Hỗn hợp": "Combination"
        }
        
        english_skin_types = [skin_types_mapping.get(st, st) for st in suitable_skin_types]
        skin_query = " ".join(english_skin_types)
        enhanced_query = f"{detected_condition} {skin_query} skin treatment moisturizer serum toner cream"
        
        return enhanced_query
    
    # KHÔNG CÓ BỆNH DA → Dùng context bình thường
    context_str = ""
    if conversation_history:
        recent_context = conversation_history[-3:]
        context_str = "LỊCH SỬ HỘI THOẠI GẦN ĐÂY:\n" + "\n".join([
            f"User: {ctx[0]}\nBot: {ctx[1][:200]}..."
            for ctx in recent_context
        ])

    return f"""{context_str}

CÂU HỎI HIỆN TẠI: {question}

Hãy trả lời dựa trên câu hỏi hiện tại. Chỉ tham khảo lịch sử nếu user đang hỏi tiếp về cùng topic."""

def build_image_analysis_query(skin_analysis: str, additional_text: str = None) -> str:
    """Build RAG query based on Image Analysis Result"""
    is_severe = any(keyword in skin_analysis.upper() for keyword in ['RẤT NẶNG', 'RẤT NGHIÊM TRỌNG', 'CẦN GẶP BÁC SĨ'])
    
    warning = "(RẤT NGHIÊM TRỌNG - CẦN GẶP BÁC SĨ)" if is_severe else "(từ phân tích ảnh)"
    advice_req = "Gợi ý 1-2 sản phẩm HỖ TRỢ NHẸ NHÀNG. NHẤN MẠNH: Cần gặp bác sĩ." if is_severe else "Tư vấn 2-3 sản phẩm CỤ THỂ phù hợp với MỨC ĐỘ."
    
    user_req = f"\nYêu cầu thêm của user: {additional_text}" if additional_text else ""
    
    return f"""Tình trạng da {warning}:
{skin_analysis}
{user_req}

{advice_req}"""

def check_severity(analysis: str) -> bool:
    """Check if skin condition is severe"""
    if not analysis: return False
    return any(keyword in analysis.upper() for keyword in ['RẤT NẶNG', 'RẤT NGHIÊM TRỌNG'])

# PRODUCT SUGGESTION HELPERS
def get_product_suggestions_by_skin_types(db, skin_types: list, num_products: int = 5) -> list:
    """
    Truy vấn sản phẩm phù hợp với loại da
    Returns: list of product names
    """
    if not db or not skin_types:
        print("⚠️ No database or skin types provided")
        return []
    
    try:
        print(f"🔍 Searching products for skin types: {skin_types}")
        
        # Map tiếng Việt sang tiếng Anh
        vietnamese_to_english = {
            "Khô": "Dry",
            "Thường": "Normal",
            "Dầu": "Oily",
            "Hỗn hợp": "Combination",
            "Nhạy cảm": "Sensitive"
        }
        
        # Tạo search terms (cả VN và EN)
        search_terms = []
        for skin_type in skin_types:
            search_terms.append(skin_type)
            if skin_type in vietnamese_to_english:
                search_terms.append(vietnamese_to_english[skin_type])
        
        print(f"🔍 Search terms (VN + EN): {search_terms}")
        
        query = f"sản phẩm chăm sóc da {' '.join(search_terms)}"
        
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": num_products * 5,
                "fetch_k": num_products * 10,
                "lambda_mult": 0.5
            }
        )
        
        docs = retriever.invoke(query)
        print(f"📚 Retrieved {len(docs)} documents from vector store")
        
        product_names = []
        seen_products = set()
        
        for doc in docs:
            product_name = doc.metadata.get('product_name')
            
            if not product_name:
                content_lines = doc.page_content.split('\n')
                for line in content_lines:
                    if 'Product Name:' in line:
                        product_name = line.split(':', 1)[1].strip()
                        break
            
            if product_name and product_name not in seen_products:
                content_lower = doc.page_content.lower()
                metadata_str = str(doc.metadata).lower()
                
                match = any(
                    term.lower() in content_lower or
                    term.lower() in metadata_str
                    for term in search_terms
                )
                
                if match:
                    product_names.append(product_name)
                    seen_products.add(product_name)
                    print(f"✓ Found: {product_name}")
                    
                    if len(product_names) >= num_products:
                        break
        
        if len(product_names) < num_products:
            print(f"⚠️ Only found {len(product_names)} matching products, adding general products...")
            for doc in docs:
                product_name = doc.metadata.get('product_name')
                if not product_name:
                    content_lines = doc.page_content.split('\n')
                    for line in content_lines:
                        if 'Product Name:' in line:
                            product_name = line.split(':', 1)[1].strip()
                            break
                
                if product_name and product_name not in seen_products:
                    product_names.append(product_name)
                    seen_products.add(product_name)
                    print(f"✓ Added general: {product_name}")
                    if len(product_names) >= num_products:
                        break
        
        print(f"✅ Returning {len(product_names)} product suggestions")
        return product_names[:num_products]
        
    except Exception as e:
        print(f"❌ Error getting product suggestions: {e}")
        import traceback
        traceback.print_exc()
        return []

def map_disease_to_skin_types(disease_class: str) -> list:
    """Map disease class sang skin types phù hợp"""
    print(f"🔍 Mapping disease: {disease_class}")
    
    disease_lower = disease_class.lower().replace('_', ' ')
    
    disease_mapping = {
        'acne': ['Hỗn hợp', 'Dầu', 'Nhạy cảm'],
        'actinic keratosis': ['Khô', 'Thường'],
        'drug eruption': ['Hỗn hợp', 'Khô', 'Thường', 'Dầu', 'Nhạy cảm'],
        'eczema': ['Hỗn hợp', 'Khô', 'Thường', 'Dầu', 'Nhạy cảm'],
        'psoriasis': ['Khô'],
        'rosacea': ['Hỗn hợp', 'Dầu', 'Nhạy cảm'],
        'seborrh keratoses': ['Thường', 'Dầu', 'Nhạy cảm'],
        'sun sunlight damage': ['Hỗn hợp', 'Khô', 'Thường', 'Nhạy cảm'],
        'tinea': ['Hỗn hợp', 'Dầu'],
        'warts': ['Hỗn hợp', 'Khô', 'Thường', 'Dầu', 'Nhạy cảm'],
        'normal': ['Thường']
    }
    
    for key, skin_types in disease_mapping.items():
        if key in disease_lower or disease_lower in key:
            print(f"✓ Mapped to skin types: {skin_types}")
            return skin_types
    
    for condition_key, skin_types in SKIN_CONDITION_TO_SKIN_TYPE.items():
        if condition_key in disease_lower or disease_lower in condition_key:
            print(f"✓ Mapped via SKIN_CONDITION_TO_SKIN_TYPE: {skin_types}")
            return skin_types
    
    print(f"⚠️ No specific mapping found, using default")
    return ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"]