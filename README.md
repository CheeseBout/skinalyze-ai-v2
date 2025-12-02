# 🧴 RAG Cosmetic Chatbot

Chatbot tư vấn mỹ phẩm thông minh kết hợp RAG, VLM và Conversation Memory.

## ✨ Tính năng

- 🤖 **RAG (Retrieval-Augmented Generation)**: Tìm kiếm và tư vấn sản phẩm từ database
- 📸 **VLM (Vision Language Model)**: Phân tích ảnh da, xác định mức độ nghiêm trọng
- 🧠 **Conversation Memory**: Nhớ ngữ cảnh trong suốt phiên chat
- ⚠️ **Severity Detection**: Cảnh báo gặp bác sĩ nếu tình trạng da rất nghiêm trọng
- 💾 **Chat History**: Lưu lịch sử hội thoại

## 🛠️ Công nghệ

- **LangChain**: Framework RAG
- **Google Gemini 2.5 Flash**: Vision & Text AI
- **ChromaDB**: Vector database
- **Sentence Transformers**: Embedding model
- **PIL**: Image processing

## 📋 Yêu cầu

- Python 3.11+
- Google Gemini API Key ([Lấy tại đây](https://makersuite.google.com/app/apikey))

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/your-username/rag-cosmetic-chatbot.git
cd rag-cosmetic-chatbot
```

### 2. Tạo virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình API Key

Copy file `.env.example` thành `.env` và điền API key:

```bash
cp .env.example .env
```

Mở file `.env` và thay `your-api-key-here` bằng Gemini API key của bạn:

```
GOOGLE_API_KEY=AIzaSy...
```

### 5. Chuẩn bị data

Đặt file `product_chunks.txt` vào thư mục `data/`:

```
data/
  └── product_chunks.txt
```

## 💬 Sử dụng

### Chạy chatbot trong terminal:

```bash
python RAG_cosmetic.py
```

### Các lệnh:

- **Text chat**: Gõ câu hỏi trực tiếp
  ```
  Tôi cần kem dưỡng cho da khô
  ```

- **Upload ảnh**: Gửi đường dẫn ảnh
  ```
  C:\Users\Photos\my_skin.jpg
  ```

- **Hỏi tiếp**: Bot nhớ context
  ```
  Còn sản phẩm nào khác không?
  Cái đầu tiên có tốt không?
  ```

- **Thoát**: Gõ `exit`, `quit`, hoặc `thoát`

## 📁 Cấu trúc project

```
rag-cosmetic-chatbot/
├── RAG_cosmetic.py          # Main chatbot
├── config.py                # Cấu hình
├── requirements.txt         # Dependencies
├── .env.example            # Template API key
├── .gitignore              # Git ignore rules
├── README.md               # Documentation
├── data/
│   └── product_chunks.txt  # Dữ liệu sản phẩm
├── db_chroma/              # Vector database (auto-generated)
└── chat_history/           # Lịch sử chat (auto-generated)
```

## 🎯 Ví dụ sử dụng

### 1. Tư vấn sản phẩm

```
🧑 Bạn: Tôi cần kem dưỡng cho da khô nhạy cảm
🤖 Bot: Mình gợi ý 2 sản phẩm phù hợp:
        1. REN CLEAN SKINCARE Evercalm™ Gentle Cleansing Milk...
        2. ...
```

### 2. Phân tích ảnh da

```
🧑 Bạn: C:\Photos\acne_skin.jpg
📸 Bot: Đang phân tích...
✅ Mức độ: TRUNG BÌNH
💄 Gợi ý: [Sản phẩm trị mụn phù hợp]
```

### 3. Hỏi tiếp (nhờ Memory)

```
🧑 Bạn: Tôi cần serum vitamin C
🤖 Bot: [Gợi ý A, B, C]

🧑 Bạn: So sánh 2 cái đầu giúp tôi
🤖 Bot: [So sánh A vs B dựa trên context]
```

## ⚠️ Cảnh báo

Nếu tình trạng da **RẤT NẶNG**, bot sẽ:
- ⚠️ Hiển thị cảnh báo rõ ràng
- 🏥 Khuyên gặp bác sĩ da liễu
- 💄 Chỉ gợi ý sản phẩm hỗ trợ nhẹ nhàng (không thay thế y khoa)

## 🔧 Cấu hình nâng cao

Chỉnh sửa `config.py` để tùy chỉnh:

- `RETRIEVER_K`: Số sản phẩm gợi ý (mặc định: 2)
- `LLM_TEMPERATURE`: Độ sáng tạo (0-1, mặc định: 0.1)
- `MAX_CONTEXT_MESSAGES`: Số tin nhắn nhớ (mặc định: 3)

## 📝 License

MIT License

## 👤 Tác giả

[Your Name] - [Your Email]

## 🙏 Credit

- Google Gemini API
- LangChain
- ChromaDB
- Hugging Face Sentence Transformers