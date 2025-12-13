"""
RAG Cosmetic Chatbot Core - Stateless for NestJS Backend Integration
Merged Features: 
- Stateless design (NestJS handles session/history)
- Advanced Skin Condition Detection (Length-prioritized matching)
- Currency Conversion (USD -> VND)
- Smart Product Grouping & Filtering (Improved)
- VLM Skin Analysis (Base64/Bytes support)
- Improved chunk_size and format_docs logic
- Enhanced error handling from standalone
- Comprehensive prompt template
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

# =============================================================================
# DATA MAPPING - EXTENDED (Ưu tiên từ khóa DÀI trước, NGẮN sau)
# =============================================================================
SKIN_CONDITION_TO_SKIN_TYPE = {
    # ⚠️ ƯU TIÊN: Từ khóa DÀI/CỤ THỂ trước, NGẮN/CHUNG sau để tránh nhầm lẫn
    
    # Mụn cóc (warts) - ƯU TIÊN TRƯỚC "mụn"
    "warts": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    "mụn cóc": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    "cóc": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
    
    # Mụn trứng cá - SAU "mụn cóc", TRƯỚC "mụn"
    "mụn trứng cá": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
    
    # Mụn (acne) - CUỐI CÙNG
    "acne": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
    "mụn": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
    
    # Các bệnh khác
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

# Danh sách bệnh da được hỗ trợ tư vấn (để kiểm tra phạm vi)
SUPPORTED_SKIN_CONDITIONS = [
    "mụn", "acne", "mụn trứng cá",
    "chàm", "eczema", "viêm da",
    "vảy nến", "psoriasis",
    "trứng cá đỏ", "rosacea", "da đỏ",
    "dày sừng", "actinic keratosis", "da dày sừng",
    "nấm da", "tinea", "nấm",
    "viêm da tiết bã", "seborrheic keratoses",
    "tổn thương nắng", "sun damage", "hư tổn do nắng",
    "mụn cóc", "warts", "cóc",
    "phát ban thuốc", "drug eruption", "phát ban do thuốc"
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
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
    # Tìm "Product Name: ..."
    match = re.search(r'Product Name:\s*(.+?)(?:\n|$)', chunk_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Tìm "Tên sản phẩm: ..."
    match = re.search(r'Tên sản phẩm:\s*(.+?)(?:\n|$)', chunk_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: lấy dòng đầu tiên có dấu :
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
    Returns: (detected_condition, skin_types_list) hoặc (None, None)
    
    ⚠️ ƯU TIÊN: Kiểm tra từ khóa DÀI trước (mụn cóc) rồi mới đến NGẮN (mụn)
    để tránh nhầm lẫn khi "mụn cóc" chứa từ "mụn"
    """
    query_lower = query.lower()
    
    # Sắp xếp theo độ dài từ khóa (dài -> ngắn) để ưu tiên match cụ thể trước
    sorted_conditions = sorted(
        SKIN_CONDITION_TO_SKIN_TYPE.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )
    
    for condition, skin_types in sorted_conditions:
        if condition in query_lower:
            return condition, skin_types
    
    return None, None

def is_supported_condition(condition):
    """Kiểm tra bệnh da có trong danh sách hỗ trợ không"""
    if not condition:
        return False
    condition_lower = condition.lower()
    return any(supported in condition_lower or condition_lower in supported 
               for supported in SUPPORTED_SKIN_CONDITIONS)

# =============================================================================
# VECTOR STORE - Enhanced Error Handling
# =============================================================================
def load_or_create_vectorstore():
    """Load or create vector store with comprehensive error handling"""
    global _CACHED_EMBEDDINGS
    
    print("=" * 80)
    print("📚 KHỞI TẠO VECTOR STORE")
    print("=" * 80)
    
    db = None
    embeddings = None
    
    try:
        # ----- Tải Embedding Model (với cache) -----
        if _CACHED_EMBEDDINGS is not None:
            print(f"\n⚡ Sử dụng cached embedding model")
            embeddings = _CACHED_EMBEDDINGS
        else:
            print(f"\n⏳ Đang tải embedding model: {MODEL_NAME}...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"   🖥️ Sử dụng thiết bị: {device}")
            
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name=MODEL_NAME,
                    model_kwargs={'device': device},
                    encode_kwargs={'normalize_embeddings': True}
                )
                _CACHED_EMBEDDINGS = embeddings
                print("✅ Đã tải embedding model!\n")
            except Exception as e_embed:
                print(f"\n❌ LỖI NGHIÊM TRỌNG khi tải embedding model: {e_embed}")
                print("   Kiểm tra lại tên model, kết nối mạng và cài đặt thư viện.")
                return None, None

        # ----- Load hoặc Tạo Database -----
        should_create_new = True
        
        if os.path.exists(PERSIST_DIRECTORY) and len(os.listdir(PERSIST_DIRECTORY)) > 0:
            print(f"📂 Phát hiện Vector Store có sẵn tại: {PERSIST_DIRECTORY}")
            print("⏳ Đang load database...\n")
            
            try:
                db = Chroma(
                    persist_directory=str(PERSIST_DIRECTORY),
                    embedding_function=embeddings
                )
                count = db._collection.count() if db._collection else 0
                
                if count > 0:
                    print(f"✅ Đã load Vector Store thành công!")
                    print(f"   📊 Số documents trong database: {count}\n")
                    should_create_new = False
                else:
                    print("   ⚠️ Database có sẵn nhưng RỖNG (0 docs). Sẽ tạo lại...")
                    
            except Exception as e_db_load:
                print(f"\n❌ LỖI khi load Vector Store có sẵn: {e_db_load}")
                print(f"   Thử xóa thư mục '{PERSIST_DIRECTORY}' và chạy lại để tạo mới.")
                return None, embeddings
        
        # ----- Tạo Database mới nếu cần -----
        if should_create_new:
            print(f"🆕 Bắt đầu tạo mới Vector Store từ {CHUNKS_FILE.name}...\n")
            
            # Kiểm tra file tồn tại
            if not CHUNKS_FILE.exists():
                print(f"❌ KHÔNG TÌM THẤY FILE DATA: {CHUNKS_FILE}")
                return None, embeddings
            
            try:
                # Load file chunks
                print("📖 [1/4] Đang load file chunks...")
                loader = TextLoader(str(CHUNKS_FILE), encoding='utf-8')
                documents = loader.load()
                print(f"   ✓ Đã load {len(documents)} document base")
                
                # Split documents
                print("✂️  [2/4] Đang split thành từng chunk và thêm metadata...")
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["---"],
                    chunk_size=800,
                    chunk_overlap=100,
                    length_function=len
                )
                docs = text_splitter.split_documents(documents)
                
                if not docs:
                    print("   ⚠️ Không split được chunk nào. Kiểm tra file và separator.")
                    return None, embeddings
                
                # Thêm metadata product_name cho mỗi chunk
                for doc in docs:
                    product_name = extract_product_name(doc.page_content)
                    doc.metadata['product_name'] = product_name
                
                print(f"   ✓ Đã split thành {len(docs)} chunks với metadata product_name")
                
                # Tạo embeddings và lưu
                print("💾 [3/4] Đang tạo embeddings và lưu vào database...")
                print("   (Quá trình này có thể mất vài phút...)\n")
                
                start_time = time.time()
                batch_size = 50
                total_docs = len(docs)
                
                if total_docs == 0:
                    print("   ⚠️ Không có chunk nào để thêm vào database.")
                    return None, embeddings
                
                # Batch đầu tiên - tạo database
                print(f"   ⏳ Đang xử lý batch 1/{(total_docs-1)//batch_size + 1}...")
                db = Chroma.from_documents(
                    documents=docs[:batch_size],
                    embedding=embeddings,
                    persist_directory=str(PERSIST_DIRECTORY)
                )
                
                # Các batch tiếp theo
                for i in range(batch_size, total_docs, batch_size):
                    batch_end = min(i + batch_size, total_docs)
                    batch_num = (i // batch_size) + 1
                    total_batches = (total_docs - 1) // batch_size + 1
                    
                    print(f"   → Batch {batch_num}/{total_batches}: docs {i}-{batch_end}...", end='\r')
                    
                    try:
                        db.add_documents(docs[i:batch_end])
                    except Exception as batch_error:
                        print(f"\n   ❌ Lỗi batch {batch_num}: {batch_error}")
                    
                    # Giải phóng bộ nhớ GPU
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                
                end_time = time.time()
                count_after = db._collection.count() if db._collection else 0
                
                print(f"\n   ✓ Hoàn thành sau {end_time - start_time:.2f} giây")
                print(f"   📊 Đã tạo và lưu {count_after} vectors")
                
                if count_after != total_docs:
                    print(f"   ⚠️ Cảnh báo: Số vector ({count_after}) không khớp số chunk ({total_docs})")
                
                print("\n✅ Đã tạo Vector Store thành công!")
                
            except FileNotFoundError as e_file:
                print(f"\n❌ LỖI: {e_file}")
                return None, embeddings
            except Exception as e_create:
                print(f"\n❌ LỖI khi tạo Vector Store: {e_create}")
                import traceback
                traceback.print_exc()
                
                # Xóa thư mục có thể bị tạo dở
                if os.path.exists(PERSIST_DIRECTORY):
                    try:
                        import shutil
                        shutil.rmtree(PERSIST_DIRECTORY)
                        print(f"   Đã xóa thư mục '{PERSIST_DIRECTORY}' có thể bị lỗi.")
                    except:
                        pass
                return None, embeddings

        return db, embeddings
        
    except Exception as e_global:
        print(f"\n❌ ĐÃ XẢY RA LỖI KHÔNG XÁC ĐỊNH: {e_global}")
        import traceback
        traceback.print_exc()
        return None, None

# =============================================================================
# RAG CHAIN SETUP - Enhanced Prompt & Format
# =============================================================================
def setup_rag_chain(db):
    """Setup RAG chain with comprehensive prompt and smart product grouping"""
    print("\n" + "=" * 80)
    print("⛓️ KHỞI TẠO RAG CHAIN")
    print("=" * 80)
    
    if db is None:
        print("\n❌ LỖI: Vector store chưa được khởi tạo thành công!")
        return None
    
    # 1. LLM Configuration (giữ nguyên thông số production)
    print("\n🤖 [1/3] Đang kết nối với Google Gemini...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.05,
        max_output_tokens=1500,
        convert_system_message_to_human=True,
        request_options={"timeout": 90},
        max_retries=3
    )
    print("   ✓ Đã kết nối Gemini 2.0 Flash (temperature=0.05, max_tokens=1500)")
    
    # 2. Retriever
    print("🔍 [2/3] Đang tạo Retriever...")
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 30}
    )
    print("   ✓ Retriever: tìm 30 chunks relevant nhất (similarity search)")
    
    # 3. Enhanced Prompt Template (từ standalone)
    print("📝 [3/3] Đang tạo Prompt Template...")
    
    template = """You are a strict assistant. You must answer questions based ONLY on the provided context below. DO NOT use your internal knowledge to update or guess prices. If the price is not mentioned in the context, say 'Price not available'.
Bạn là chuyên gia tư vấn mỹ phẩm chuyên nghiệp, thân thiện và hiểu tâm lý khách hàng.
PHÂN LOẠI CÂU HỎI VÀ CÁCH TRẢ LỜI:
🔹 **CHÀO HỎI/GIAO TIẾP CƠ BẢN**
Câu hỏi: "xin chào", "hi", "hello", "chào bạn", "hey"
→ "Chào bạn! 👋 Mình là trợ lý tư vấn mỹ phẩm. Bạn muốn tìm sản phẩm gì hôm nay? 😊"
🔹 **HỎI VỀ CHỨC NĂNG/GIỚI THIỆU**
Câu hỏi: "bạn là ai", "bạn làm gì", "có thể giúp gì", "bạn biết gì"
→ "Mình là chuyên gia tư vấn mỹ phẩm! 💄 Mình có thể giúp bạn:
• Tìm sản phẩm theo loại da (khô, dầu, nhạy cảm, hỗn hợp, thường)
• Tư vấn sản phẩm theo BỆNH DA (mụn, chàm, vảy nến, trứng cá đỏ, nấm da...)
• Tư vấn kem dưỡng, serum, toner, mặt nạ, sữa rửa mặt, kem chống nắng
• Giải thích thành phần và công dụng sản phẩm
Bạn đang gặp vấn đề gì về da hoặc cần tìm sản phẩm nào? 😊"
🔹 **HỎI CHUNG CHUNG KHÔNG CỤ THỂ**
Câu hỏi: "có sản phẩm gì", "cho xem sản phẩm", "gợi ý sản phẩm"
→ "Mình có rất nhiều sản phẩm! 😊 Để tư vấn chính xác, bạn cho mình biết:
• Loại da của bạn? (khô/dầu/hỗn hợp/nhạy cảm/thường)
• Bệnh da (nếu có)? (mụn/chàm/vảy nến/trứng cá đỏ/nấm da...)
• Loại sản phẩm cần? (kem dưỡng/serum/toner/mặt nạ/sữa rửa mặt...)
Cho mình biết để mình tư vấn đúng nhu cầu nhé! 💕"
🔹 **HỎI VỀ BỆNH DA (ƯU TIÊN CAO)**
Câu hỏi: "tôi bị mụn", "tôi bị chàm", "da bị vảy nến", "bị trứng cá đỏ", "nấm da", "mụn cóc"...
⚠️ **KIỂM TRA PHẠM VI TƯ VẤN:**
→ CHỈ tư vấn cho CÁC BỆNH DA SAU (có trong database):
   • Mụn (Acne) → Hỗn hợp/Dầu/Nhạy cảm
   • Chàm (Eczema) → Hỗn hợp/Khô/Thường/Dầu/Nhạy cảm
   • Vảy nến (Psoriasis) → Khô
   • Trứng cá đỏ (Rosacea) → Hỗn hợp/Dầu/Nhạy cảm
   • Dày sừng (Actinic Keratosis) → Khô/Thường
   • Nấm da (Tinea) → Hỗn hợp/Dầu
   • Viêm da tiết bã (Seborrheic Keratoses) → Thường/Dầu/Nhạy cảm
   • Tổn thương nắng (Sun Damage) → Hỗn hợp/Khô/Thường/Nhạy cảm
   • Mụn cóc (Warts) → Hỗn hợp/Khô/Thường/Dầu/Nhạy cảm
   • Phát ban thuốc (Drug Eruption) → Hỗn hợp/Khô/Thường/Dầu/Nhạy cảm
→ **NẾU BỆNH DA KHÔNG TRONG DANH SÁCH TRÊN** (vd: ghẻ, lang ben, zona, herpes...):
   "⚠️ Xin lỗi, bệnh [tên bệnh] NẰM NGOÀI PHẠM VI tư vấn mỹ phẩm của mình.
   
   🏥 KHUYẾN CÁO:
   • Đây là bệnh da CẦN ĐIỀU TRỊ Y KHOA
   • Vui lòng ĐẶT LỊCH GẶP BÁC SĨ DA LIỄU để được khám và kê đơn thuốc phù hợp
   
   💡 Mình có thể tư vấn mỹ phẩm cho các vấn đề da thông thường như: mụn, chàm, vảy nến... Bạn có vấn đề da nào trong số này không?"
→ **NẾU BỆNH DA CÓ TRONG DANH SÁCH - TRẢ LỜI NGẮN GỌN:**
   "Dạ, mình gợi ý sản phẩm cho [tên bệnh] nhé:
   [LIỆT KÊ 2-3 SẢN PHẨM NGAY]"
   
   ⚠️ QUAN TRỌNG: 
   • CHỈ trả lời về bệnh da được NÊU TRONG CÂU HỎI HIỆN TẠI
   • KHÔNG được nhắc lại hoặc nhầm lẫn với câu hỏi trước
   • KHÔNG cần giải thích dài dòng
🔹 **HỎI VỀ VẤN ĐỀ DA (KHÔNG PHẢI BỆNH)**
Câu hỏi: "da tôi khô", "da dầu nhiều", "da nhạy cảm"
→ ĐI THẲNG VÀO: "Dạ, mình gợi ý sản phẩm cho da [loại da] nhé:
   [LIỆT KÊ 2-3 SẢN PHẨM NGAY]"
🔹 **HỎI THEO LOẠI SẢN PHẨM**
Câu hỏi: "có kem dưỡng nào...", "serum gì tốt", "toner cho da..."
→ ĐI THẲNG VÀO SẢN PHẨM, mặc định 2-3 sản phẩm
🔹 **HỎI VỀ THƯƠNG HIỆU**
Câu hỏi: "bạn có [tên thương hiệu] không"
→ Kiểm tra database, nếu có thì liệt kê, nếu không: "Mình chưa có thông tin về [brand] trong database. Bạn muốn tư vấn sản phẩm theo loại da không? 🔍"
🔹 **HỎI GIÁ/MUA Ở ĐÂU**
→ "Xin lỗi, mình chỉ tư vấn về sản phẩm thôi nhé. Bạn có thể mua tại các store chính hãng. Mình tư vấn thêm sản phẩm khác không? 😊"
🔹 **CẢM ƠN/TẠM BIỆT**
→ "Không có gì! 😊 Chúc bạn có làn da đẹp! Hẹn gặp lại! 💕"
🔹 **CÂU HỎI NGOÀI LỀ**
→ "Xin lỗi, mình chỉ chuyên về mỹ phẩm và skincare thôi 💄 Bạn có muốn hỏi về chăm sóc da không?"
---
**CHÚ Ý KHI TRẢ LỜI:**
- Luôn THÂN THIỆN, dùng "mình/bạn"
- **NGẮN GỌN - ĐI THẲNG VÀO SẢN PHẨM**
- **⚠️ TẬP TRUNG VÀO CÂU HỎI HIỆN TẠI** - không nhắc câu hỏi cũ
- **GROUNDING:** CHỈ GỢI Ý sản phẩm CÓ TRONG DATABASE
  • Nếu context chứa "KHÔNG TÌM THẤY SẢN PHẨM" → trả lời:
    "Xin lỗi, mình không tìm thấy sản phẩm phù hợp. 😔 Bạn thử mô tả chi tiết hơn?"
  • TUYỆT ĐỐI KHÔNG TỰ BỊA sản phẩm
- **SỐ LƯỢNG:** Mặc định 2-3 sản phẩm (tối đa 3)
- **⚠️ FORMAT MỖI SẢN PHẨM (BẮT BUỘC):**
  **Số. Tên sản phẩm của THƯƠNG HIỆU** 
  Giá: XXX.XXX VND | Đánh giá: X/5 | Loại da: [...]
  
  ⚠️ BẮT BUỘC: Tên, Thương hiệu, Giá, Đánh giá, Loại da
  • NẾU thiếu thông tin → ghi "(Không có thông tin)"
  • KHÔNG hiển thị công dụng hoặc thành phần
- **GIÁ:** CHỈ HIỂN THỊ VND, KHÔNG USD
- **LOẠI DA PHẢI DỊCH SANG TIẾNG VIỆT:**
  Combination → Hỗn hợp | Dry → Khô | Normal → Thường | Oily → Dầu | Sensitive → Nhạy cảm
- Dùng emoji phù hợp: 😊💄✨💕👋
THÔNG TIN SẢN PHẨM:
{context}
CÂU HỎI: {question}
TRẢ LỜI (2-3 sản phẩm):"""
    
    prompt = ChatPromptTemplate.from_template(template)
    print("   ✓ Đã tạo Prompt Template (comprehensive + grounding)")
    
    # 4. Format docs function với metadata extraction
    def format_docs(docs):
        """
        Format documents: NHÓM chunks theo product_name,
        trích xuất metadata, sắp xếp theo relevance,
        chọn top 3 sản phẩm với thông tin đầy đủ
        """
        # GROUNDING CHECK
        if not docs or len(docs) == 0:
            return "KHÔNG TÌM THẤY SẢN PHẨM TRONG DATABASE"
        
        print(f"   🔍 Tìm được {len(docs)} chunks từ database")
        
        # Bước 1: Nhóm chunks theo product_name
        product_groups = {}
        
        for idx, doc in enumerate(docs):
            product_name = doc.metadata.get('product_name', 'Unknown Product')
            
            if product_name not in product_groups:
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
                    'has_summary': False,
                    'has_ingredients': False
                }
            
            # Đánh dấu loại chunk
            content_lower = doc.page_content.lower()
            if 'chunk type: product summary' in content_lower:
                product_groups[product_name]['has_summary'] = True
            if 'chunk type: ingredients' in content_lower:
                product_groups[product_name]['has_ingredients'] = True
            
            product_groups[product_name]['chunks'].append(doc)
        
        if not product_groups:
            return "KHÔNG TÌM THẤY SẢN PHẨM TRONG DATABASE"
        
        print(f"   📦 Tìm được {len(product_groups)} sản phẩm khác nhau")
        
        # Bước 2: Lọc sản phẩm có đủ thông tin (ưu tiên có summary)
        complete_products = [(name, data) for name, data in product_groups.items() if data['has_summary']]
        
        if not complete_products:
            complete_products = list(product_groups.items())
        
        # Bước 3: Sắp xếp theo relevance (first_index)
        sorted_products = sorted(complete_products, key=lambda x: x[1]['first_index'])
        
        # Bước 4: Chọn top 3 sản phẩm
        num_products = min(3, len(sorted_products))
        selected_products = sorted_products[:num_products]
        
        print(f"   ✅ Chọn {num_products} sản phẩm để tư vấn")
        
        # Bước 5: Format output
        formatted = []
        for i, (product_name, data) in enumerate(selected_products, 1):
            chunks = data['chunks']
            metadata = data['metadata']
            
            # Loại bỏ duplicate chunks
            seen_contents = set()
            unique_chunks = []
            for chunk in chunks:
                content_hash = hash(chunk.page_content.strip())
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_chunks.append(chunk)
            
            # Sắp xếp: Summary trước, Ingredients sau
            def chunk_priority(chunk):
                content = chunk.page_content.lower()
                if 'chunk type: product summary' in content:
                    return 0
                elif 'chunk type: ingredients' in content:
                    return 1
                else:
                    return 2
            
            sorted_chunks = sorted(unique_chunks, key=chunk_priority)
            
            # Gộp thông tin sản phẩm
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
    
    # Build RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("\n✅ RAG Chain đã sẵn sàng!")
    print("\n📊 Luồng hoạt động (CẢI TIẾN):")
    print("   1️⃣  User Question → Retriever")
    print("   2️⃣  Retriever → 30 chunks (similarity search)")
    print("   3️⃣  Trích xuất metadata từ chunks")
    print("   4️⃣  NHÓM theo product_name + Filter sản phẩm có đủ thông tin")
    print("   5️⃣  Sắp xếp theo relevance → Chọn top 3 sản phẩm")
    print("   6️⃣  Loại bỏ duplicate + Sắp xếp: Summary → Ingredients")
    print("   7️⃣  Format structured với metadata rõ ràng")
    print("   8️⃣  Context + Question → LLM → 2-3 sản phẩm CHÍNH XÁC ⚡")
    
    return rag_chain

# =============================================================================
# VISION ANALYSIS - STATELESS (Base64/Bytes/PIL/Path support)
# =============================================================================
def analyze_skin_image(image_input, note: str = None):
    """
    Analyze skin image - STATELESS version with severity detection
    Args:
        image_input: PIL Image, base64 string, bytes, or file path
        note: Additional note from user
    Returns:
        str: Analysis result
    """
    try:
        print("\n📸 Analyzing skin image...")
        
        # Handle multiple input types
        img = None
        if isinstance(image_input, str):
            # Check for data URI
            if image_input.startswith('data:image'):
                image_input = image_input.split(',')[1]
            try:
                # Try base64 decode
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

        vision_model = genai.GenerativeModel('gemini-2.5-flash')
        
        vision_prompt = """Bạn là chuyên gia da liễu. Phân tích ảnh da và TÓM TẮT NGẮN GỌN:
1. LOẠI DA: (khô/dầu/hỗn hợp/nhạy cảm/thường)
2. VẤN ĐỀ CHÍNH & MỨC ĐỘ NGHIÊM TRỌNG:
- Nếu có mụn: loại mụn (viêm/đầu đen/đầu trắng/bọc), mức độ (NHẸ/TRUNG BÌNH/NẶNG/RẤT NẶNG)
- Nếu có thâm/sẹo: mức độ (NHẸ/TRUNG BÌNH/NẶNG/RẤT NẶNG), màu sắc, phân bố
- Nếu có lão hóa: mức độ (NHẸ/TRUNG BÌNH/NẶNG)
- Nếu có vấn đề khác: nêu rõ
3. MỨC ĐỘ CHUNG: Chọn 1 trong 4 (QUAN TRỌNG):
   - NHẸ: Vấn đề nhỏ, ít nốt, có thể tự chăm sóc
   - TRUNG BÌNH: Vấn đề rõ ràng, nhiều nốt, cần sản phẩm chuyên dụng
   - NẶNG: Vấn đề lan rộng, viêm nhiều, cần điều trị tích cực
   - RẤT NẶNG: Viêm trầm trọng, sẹo nhiều, CẦN GẶP BÁC SĨ
4. GỢI Ý: (1 câu ngắn)
QUAN TRỌNG: Phải ghi rõ MỨC ĐỘ.
Trả lời NGẮN GỌN, bằng tiếng Việt."""

        if note:
            vision_prompt += f"\n\nGhi chú thêm từ người dùng: {note}"
        
        response = vision_model.generate_content([vision_prompt, img])
        analysis = response.text
        print("✅ Analysis complete!")
        return analysis
        
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file ảnh")
        return None
    except Exception as e:
        print(f"❌ Error analyzing image: {str(e)}")
        return None

# =============================================================================
# HELPER FUNCTIONS FOR NESTJS INTEGRATION
# =============================================================================
def analyze_with_context(question: str, conversation_history: list = None) -> str:
    """
    Analyze question with conversation context + Skin Condition Logic
    ⚠️ NẾU PHÁT HIỆN BỆNH DA → BỎ QUA CONTEXT để tránh nhầm lẫn
    """
    # Detect skin condition
    detected_condition, suitable_skin_types = detect_skin_condition_and_types(question)
    
    if detected_condition:
        # CÓ BỆNH DA → Không dùng context, mapping sang tiếng Anh
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
    """Build RAG query based on Image Analysis Result with severity awareness"""
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
    if not analysis:
        return False
    return any(keyword in analysis.upper() for keyword in ['RẤT NẶNG', 'RẤT NGHIÊM TRỌNG'])

# =============================================================================
# PRODUCT SUGGESTION HELPERS
# =============================================================================
def get_product_suggestions_by_skin_types(db, skin_types: list, num_products: int = 5) -> list:
    """
    Truy vấn sản phẩm phù hợp với loại da (bilingual search)
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
        
        # Fallback: add general products if not enough
        if len(product_names) < num_products:
            print(f"⚠️ Only found {len(product_names)} matching products, adding general...")
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
    
    # Fallback to SKIN_CONDITION_TO_SKIN_TYPE
    for condition_key, skin_types in SKIN_CONDITION_TO_SKIN_TYPE.items():
        if condition_key in disease_lower or disease_lower in condition_key:
            print(f"✓ Mapped via SKIN_CONDITION_TO_SKIN_TYPE: {skin_types}")
            return skin_types
    
    print(f"⚠️ No specific mapping found, using default")
    return ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"]