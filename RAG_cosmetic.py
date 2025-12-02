"""
RAG Cosmetic Chatbot Core - Stateless for NestJS Backend Integration
Merged Features: 
- Stateless design (NestJS handles session/history)
- Advanced Skin Condition Detection
- Currency Conversion (USD -> VND)
- Smart Product Grouping & Filtering
- VLM Skin Analysis (Base64/Bytes support)
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

# =============================================================================
# CONFIGURATION
# =============================================================================
# Sử dụng đường dẫn tương đối để tương thích khi deploy cùng NestJS
PATH = Path(__file__).parent.resolve()
CHUNKS_FILE = PATH / "data" / "product_chunks.txt"
PERSIST_DIRECTORY = PATH / "db_chroma"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Tỷ giá USD → VND (cố định)
USD_TO_VND = 26349

# Global cache for embeddings
_CACHED_EMBEDDINGS = None

# =============================================================================
# DATA MAPPING (TỪ FILE MỚI)
# =============================================================================
SKIN_CONDITION_TO_SKIN_TYPE = {
    "acne": ["Hỗn hợp", "Dầu", "Nhạy cảm"],  # Mụn
    "mụn": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
    "mụn trứng cá": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
    
    "actinic keratosis": ["Khô", "Thường"],  # Dày sừng
    "dày sừng": ["Khô", "Thường"],
    "da dày sừng": ["Khô", "Thường"],
    
    "phát ban thuốc": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    "phát ban do thuốc": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    
    "eczema": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],  # Chàm
    "chàm": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    "viêm da": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    
    "psoriasis": ["Khô"],  # Vảy nến
    "vảy nến": ["Khô"],
    
    "rosacea": ["Hỗn hợp", "Dầu", "Nhạy cảm"],  # Trứng cá đỏ
    "trứng cá đỏ": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
    "da đỏ": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
    
    "seborrheic keratoses": ["Thường", "Dầu", "Nhạy cảm"],  # Viêm da tiết bã
    "viêm da tiết bã": ["Thường", "Dầu", "Nhạy cảm"],
    
    "sun damage": ["Hỗn hợp", "Khô", "Thường", "Nhạy cảm"],  # Tổn thương nắng
    "tổn thương nắng": ["Hỗn hợp", "Khô", "Thường", "Nhạy cảm"],
    "hư tổn do nắng": ["Hỗn hợp", "Khô", "Thường", "Nhạy cảm"],
    
    "tinea": ["Hỗn hợp", "Dầu"],  # Nấm da
    "nấm da": ["Hỗn hợp", "Dầu"],
    "nấm": ["Hỗn hợp", "Dầu"],
    
    "warts": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],  # Mụn cóc
    "mụn cóc": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    "cóc": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def setup_api_key():
    """Setup Google API Key"""
    # Attempt to get the key from the environment
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("\n❌ CRITICAL ERROR: GOOGLE_API_KEY not found in environment variables.")
        print("Please create a .env file and add GOOGLE_API_KEY=your_new_key")
        # DO NOT fallback to a hardcoded key. It is a security risk.
        raise ValueError("GOOGLE_API_KEY is missing.")
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    print("✅ API Key configured successfully from environment!\n")

def extract_product_name(chunk_text):
    """Trích xuất tên sản phẩm từ chunk text"""
    # Tìm "Product Name: ..."
    match = re.search(r'Product Name:\s*(.+?)(?:\n|$)', chunk_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Tìm "Tên sản phẩm: ..."
    match = re.search(r'Tên sản phẩm:\s*(.+?)(?:\n|$)', chunk_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: lấy dòng đầu tiên
    lines = chunk_text.split('\n')
    for line in lines:
        if ':' in line:
            potential_name = line.split(':', 1)[1].strip()
            if len(potential_name) > 5:
                return potential_name
    return "Unknown Product"

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
    """Phát hiện bệnh da trong câu hỏi và trả về loại da phù hợp"""
    query_lower = query.lower()
    for condition, skin_types in SKIN_CONDITION_TO_SKIN_TYPE.items():
        if condition in query_lower:
            return condition, skin_types
    return None, None

# =============================================================================
# VECTOR STORE
# =============================================================================
def load_or_create_vectorstore():
    """Load or create vector store (Sửa lỗi 0 documents)"""
    global _CACHED_EMBEDDINGS
    
    print("=" * 80)
    print("📚 KHỞI TẠO VECTOR STORE")
    print("=" * 80)
    
    try:
        # 1. Load Embedding Model
        if _CACHED_EMBEDDINGS is not None:
            print(f"\n⚡ Sử dụng cached embedding model")
            embeddings = _CACHED_EMBEDDINGS
        else:
            print(f"\n⏳ Đang tải embedding model: {MODEL_NAME}...")
            # Ép dùng CPU cho ổn định trên Hugging Face
            embeddings = HuggingFaceEmbeddings(
                model_name=MODEL_NAME,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            _CACHED_EMBEDDINGS = embeddings
            print("✅ Đã tải embedding model!\n")

        # 2. Logic thông minh: Kiểm tra DB có dữ liệu thực sự không
        db = None
        should_create_new = True
        
        if os.path.exists(PERSIST_DIRECTORY) and len(os.listdir(PERSIST_DIRECTORY)) > 0:
            print(f"📂 Thư mục DB tồn tại: {PERSIST_DIRECTORY}")
            try:
                # Thử load lên
                db = Chroma(
                    persist_directory=str(PERSIST_DIRECTORY),
                    embedding_function=embeddings
                )
                # Kiểm tra số lượng
                count = db._collection.count() if db._collection else 0
                
                if count > 0:
                    print(f"✅ Đã load thành công {count} documents")
                    should_create_new = False  # Có dữ liệu rồi, không cần tạo mới
                else:
                    print(f"⚠️  DB tồn tại nhưng RỖNG (0 docs). Sẽ tiến hành tạo lại...")
            except Exception as e:
                print(f"⚠️  Lỗi khi đọc DB cũ: {e}. Sẽ tạo mới...")
        
        # 3. Tạo mới nếu cần (Khi chưa có DB hoặc DB bị rỗng)
        if should_create_new:
            print(f"🆕 Bắt đầu tạo mới Vector Store từ {CHUNKS_FILE.name}...\n")
            
            if not CHUNKS_FILE.exists():
                # Fallback: Nếu không thấy trong data/, thử tìm ở thư mục gốc
                fallback_path = BASE_DIR / "product_chunks.txt"
                if fallback_path.exists():
                     print(f"⚠️ Không thấy file trong data/, tìm thấy ở gốc: {fallback_path}")
                     loader = TextLoader(str(fallback_path), encoding='utf-8')
                else:
                     raise FileNotFoundError(f"❌ KHÔNG TÌM THẤY FILE DATA: {CHUNKS_FILE}")
            else:
                loader = TextLoader(str(CHUNKS_FILE), encoding='utf-8')
                
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["---"],
                chunk_size=400,
                chunk_overlap=50,
                length_function=len
            )
            docs = text_splitter.split_documents(documents)
            
            # Add metadata
            for doc in docs:
                product_name = extract_product_name(doc.page_content)
                doc.metadata['product_name'] = product_name

            print(f"   ✓ Split into {len(docs)} chunks")
            print("💾 Creating embeddings and saving...")
            
            # Batch processing để tránh lỗi memory trên HF Space
            batch_size = 50
            total_docs = len(docs)
            
            # Khởi tạo DB với batch đầu tiên
            db = Chroma.from_documents(
                documents=docs[:batch_size],
                embedding=embeddings,
                persist_directory=str(PERSIST_DIRECTORY)
            )
            
            # Add các batch còn lại
            for i in range(batch_size, total_docs, batch_size):
                batch_end = min(i + batch_size, total_docs)
                print(f"   Processing batch {i}/{total_docs}...", end='\r')
                db.add_documents(docs[i:batch_end])
            
            print(f"\n✅ Hoàn tất! Đã lưu {total_docs} vectors vào DB\n")

        return db, embeddings
        
    except Exception as e:
        print(f"\n❌ Error in Vector Store: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# =============================================================================
# RAG CHAIN SETUP (Integrated Logic)
# =============================================================================
def setup_rag_chain(db):
    """Setup RAG chain with Advanced Prompt and Grouping"""
    print("\n" + "=" * 80)
    print("⛓️ KHỞI TẠO RAG CHAIN")
    print("=" * 80)
    
    # LLM
    print("\n🤖 [1/3] Connecting to Google Gemini...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        max_output_tokens=1200,
        convert_system_message_to_human=True,
        request_options={"timeout": 60},
        max_retries=2
    )
    print("   ✓ Connected to Gemini 2.0 Flash")
    
    # Retriever
    print("🔍 [2/3] Creating Retriever...")
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,           # Fetch 8 chunks
            "fetch_k": 16,
            "lambda_mult": 0.7
        }
    )
    print("   ✓ Retriever ready (MMR, Smart Grouping)")
    
    # Prompt Template (Updated from New File)
    print("📝 [3/3] Creating Prompt Template...")
    template = """Bạn là chuyên gia tư vấn mỹ phẩm chuyên nghiệp, thân thiện và hiểu tâm lý khách hàng.

PHÂN LOẠI CÂU HỎI VÀ CÁCH TRẢ LỜI:

🔹 **CHÀO HỎI/GIAO TIẾP CƠ BẢN**
Câu hỏi: "xin chào", "hi", "hello"
→ "Chào bạn! 👋 Mình là trợ lý tư vấn mỹ phẩm. Bạn muốn tìm sản phẩm gì hôm nay? 😊"

🔹 **HỎI VỀ BỆNH DA (ƯU TIÊN CAO)**
Câu hỏi: "tôi bị mụn", "chàm", "vảy nến", "nấm da"...
→ **BƯỚC 1:** Xác định BỆNH DA và LOẠI DA PHÙ HỢP (đã có trong context).
→ **BƯỚC 2:** GỢI Ý 2 SẢN PHẨM phù hợp nhất từ database.

🔹 **HỎI VỀ VẤN ĐỀ DA/LOẠI SẢN PHẨM**
Câu hỏi: "da khô", "da dầu", "kem dưỡng", "serum"...
→ Gợi ý 2 sản phẩm PHÙ HỢP từ database, nêu rõ CÔNG DỤNG và LOẠI DA phù hợp.

🔹 **HỎI GIÁ/MUA Ở ĐÂU**
→ "Xin lỗi, mình chỉ tư vấn về sản phẩm. Bạn có thể mua tại store chính hãng. Mình tư vấn thêm sản phẩm khác nhé? 😊"

---

**CHÚ Ý KHI TRẢ LỜI:**
- Luôn THÂN THIỆN, dùng "mình/bạn".
- **GROUNDING:** CHỈ GỢI Ý sản phẩm CÓ TRONG DATABASE bên dưới.
- **SỐ LƯỢNG:** Mặc định 2 sản phẩm (trừ khi user hỏi cụ thể số lượng).
- **FORMAT:**
  **Số. Tên sản phẩm** Giá: XXX.XXX VND | Loại da: [...]
  Công dụng: [tóm tắt ngắn]
- **KHÔNG HIỂN THỊ USD**, chỉ hiển thị VND.
- Dùng emoji phù hợp: 😊💄✨💕💊

THÔNG TIN SẢN PHẨM TỪ DATABASE:
{context}

LỊCH SỬ/CONTEXT CÂU HỎI:
{question}

TRẢ LỜI:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Advanced Formatting Function (from New File)
    def format_docs(docs):
        """Format documents: NHÓM chunks theo product_name, chỉ lấy 2 sản phẩm đầu tiên"""
        if not docs or len(docs) == 0:
            return "KHÔNG TÌM THẤY SẢN PHẨM TRONG DATABASE"
        
        # Nhóm các chunks theo product_name
        product_groups = {}
        for doc in docs:
            product_name = doc.metadata.get('product_name', 'Unknown Product')
            if product_name not in product_groups:
                product_groups[product_name] = []
            product_groups[product_name].append(doc)
        
        if not product_groups:
            return "KHÔNG TÌM THẤY SẢN PHẨM TRONG DATABASE"
        
        # Chỉ lấy 2 sản phẩm đầu tiên (hoặc nhiều hơn nếu cần logic mở rộng sau này)
        selected_products = list(product_groups.keys())[:2]
        
        formatted = []
        for i, product_name in enumerate(selected_products, 1):
            chunks = product_groups[product_name]
            product_info = f"=== SẢN PHẨM {i}: {product_name} ===\n"
            for chunk in chunks:
                content = chunk.page_content.strip()
                # Áp dụng chuyển đổi tiền tệ
                content = convert_price_in_text(content)
                product_info += content + "\n"
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

# =============================================================================
# VISION ANALYSIS - STATELESS (Accepts PIL Image or base64)
# =============================================================================
def analyze_skin_image(image_input, note: str = None):
    """
    Analyze skin image - STATELESS version
    Args:
        image_input: PIL Image, base64 string, or file path
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
                # Try base64
                image_bytes = base64.b64decode(image_input)
                img = Image.open(io.BytesIO(image_bytes))
            except:
                # Try file path
                img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        
        if img is None:
            raise ValueError("Invalid image input")

        # Use updated model from New File logic (Gemini 2.5 Flash if available, else 2.0)
        # Using 2.0 Flash Exp/Stable as a safe bet from the provided code context
        vision_model = genai.GenerativeModel('gemini-2.0-flash') 
        
        # Updated Prompt from New File (Severity Focused)
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
            vision_prompt += f"\n\n Ghi chú thêm từ người dùng: {note}"
        
        response = vision_model.generate_content([vision_prompt, img])
        analysis = response.text
        print("✅ Analysis complete!")
        return analysis
        
    except Exception as e:
        print(f"❌ Error analyzing image: {str(e)}")
        return None

# =============================================================================
# HELPER FUNCTIONS FOR NESTJS INTEGRATION
# =============================================================================
def analyze_with_context(question: str, conversation_history: list = None) -> str:
    """
    Analyze question with conversation context + Skin Condition Logic
    Args:
        question: User's question
        conversation_history: List of (user_msg, bot_response) tuples
    Returns:
        str: Enhanced query string for the RAG chain
    """
    # 1. Logic phát hiện bệnh da (từ file mới)
    detected_condition, suitable_skin_types = detect_skin_condition_and_types(question)
    
    enhanced_part = ""
    if detected_condition:
        skin_types_str = ", ".join(suitable_skin_types)
        enhanced_part = f"""
THÔNG TIN BỔ SUNG TỪ HỆ THỐNG:
- Phát hiện bệnh da: {detected_condition}
- Loại da phù hợp: {skin_types_str}
- Vui lòng tìm sản phẩm cho các loại da: {skin_types_str}"""

    # 2. Logic Context
    context_str = ""
    if conversation_history:
        recent_context = conversation_history[-3:]
        context_str = "LỊCH SỬ HỘI THOẠI GẦN ĐÂY:\n" + "\n".join([
            f"User: {ctx[0]}\nBot: {ctx[1][:200]}..." 
            for ctx in recent_context
        ])

    # 3. Combine
    final_query = f"""{context_str}

CÂU HỎI HIỆN TẠI: {question}
{enhanced_part}

Hãy trả lời dựa trên LỊCH SỬ và câu hỏi hiện tại."""

    return final_query

def build_image_analysis_query(skin_analysis: str, additional_text: str = None) -> str:
    """
    Build RAG query based on Image Analysis Result
    """
    # Check severity (logic from New File)
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

# =============================================================================
# PRODUCT SUGGESTION HELPERS
# =============================================================================
def get_product_suggestions_by_skin_types(db, skin_types: list, num_products: int = 3) -> list:
    """
    Truy vấn sản phẩm phù hợp với loại da
    Args:
        db: Vector store database
        skin_types: Danh sách loại da (e.g., ["Hỗn hợp", "Dầu"])
        num_products: Số lượng sản phẩm tối đa
    Returns:
        list: Danh sách tên sản phẩm
    """
    if not db or not skin_types:
        print("⚠️ No database or skin types provided")
        return []
    
    try:
        print(f"🔍 Searching products for skin types: {skin_types}")
        
        # Tạo query từ loại da
        skin_types_query = " ".join(skin_types)
        query = f"sản phẩm chăm sóc da {skin_types_query}"
        
        # Truy vấn vector store
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": num_products * 5,  # Lấy nhiều hơn để lọc
                "fetch_k": num_products * 10,
                "lambda_mult": 0.5
            }
        )
        
        docs = retriever.invoke(query)
        print(f"📚 Retrieved {len(docs)} documents from vector store")
        
        # Nhóm theo product_name và lấy unique products
        product_names = []
        seen_products = set()
        
        for doc in docs:
            # Lấy tên sản phẩm từ metadata hoặc extract từ content
            product_name = doc.metadata.get('product_name')
            
            if not product_name:
                # Fallback: extract từ page_content
                content_lines = doc.page_content.split('\n')
                for line in content_lines:
                    if 'Tên sản phẩm:' in line or 'Product Name:' in line:
                        product_name = line.split(':', 1)[1].strip()
                        break
            
            if product_name and product_name not in seen_products:
                # Kiểm tra nếu product phù hợp với skin types
                content_lower = doc.page_content.lower()
                metadata_str = str(doc.metadata).lower()
                
                match = any(
                    skin_type.lower() in content_lower or 
                    skin_type.lower() in metadata_str 
                    for skin_type in skin_types
                )
                
                if match:
                    product_names.append(product_name)
                    seen_products.add(product_name)
                    print(f"✓ Found: {product_name}")
                    
                    if len(product_names) >= num_products:
                        break
        
        # Nếu không tìm được đủ, lấy bất kỳ sản phẩm nào
        if len(product_names) < num_products:
            print(f"⚠️ Only found {len(product_names)} matching products, adding general products...")
            for doc in docs:
                product_name = doc.metadata.get('product_name')
                if product_name and product_name not in seen_products:
                    product_names.append(product_name)
                    seen_products.add(product_name)
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
    """
    Map disease class sang skin types phù hợp
    Args:
        disease_class: Tên bệnh da (e.g., "Acne", "Eczema")
    Returns:
        list: Danh sách loại da phù hợp
    """
    print(f"🔍 Mapping disease: {disease_class}")
    
    # Normalize disease name
    disease_lower = disease_class.lower().replace('_', ' ')
    
    # Direct mapping từ SKIN_CLASSES
    disease_mapping = {
        'acne': ['Hỗn hợp', 'Dầu', 'Nhạy cảm'],
        'actinic keratosis': ['Khô', 'Thường'],
        'eczema': ['Hỗn hợp', 'Khô', 'Thường', 'Dầu', 'Nhạy cảm'],
        'psoriasis': ['Khô'],
        'rosacea': ['Hỗn hợp', 'Dầu', 'Nhạy cảm'],
        'seborrh keratoses': ['Thường', 'Dầu', 'Nhạy cảm'],
        'sun sunlight damage': ['Hỗn hợp', 'Khô', 'Thường', 'Nhạy cảm'],
        'tinea': ['Hỗn hợp', 'Dầu'],
        'warts': ['Hỗn hợp', 'Khô', 'Thường', 'Dầu', 'Nhạy cảm'],
        'normal': ['Thường']
    }
    
    # Tìm trong mapping
    for key, skin_types in disease_mapping.items():
        if key in disease_lower or disease_lower in key:
            print(f"✓ Mapped to skin types: {skin_types}")
            return skin_types
    
    # Fallback: tìm trong SKIN_CONDITION_TO_SKIN_TYPE
    for condition_key, skin_types in SKIN_CONDITION_TO_SKIN_TYPE.items():
        if condition_key in disease_lower or disease_lower in condition_key:
            print(f"✓ Mapped via SKIN_CONDITION_TO_SKIN_TYPE: {skin_types}")
            return skin_types
    
    # Default fallback
    print(f"⚠️ No specific mapping found, using default")
    return ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"]

def map_skin_condition_to_types(condition: str) -> list:
    """
    Map skin condition (Dry/Normal/Oily) sang loại da tiếng Việt
    Args:
        condition: "Dry", "Normal", hoặc "Oily"
    Returns:
        list: Danh sách loại da
    """
    print(f"🔍 Mapping condition: {condition}")
    
    mapping = {
        "dry": ["Khô"],
        "normal": ["Thường"],
        "oily": ["Dầu"],
    }
    
    result = mapping.get(condition.lower(), ["Thường"])
    print(f"✓ Mapped to: {result}")
    return result