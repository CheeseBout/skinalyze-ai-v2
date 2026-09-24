---
title: Skinalyze API
emoji: 🧴
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# 🧴 Skinalyze API

API RESTful cho hệ thống phân tích da liễu và tư vấn mỹ phẩm, dùng cho backend NestJS. Kết hợp:

- **Phân loại bệnh da** từ ảnh bằng EfficientNet-B0 tự host (11 lớp: mụn, chàm, vảy nến, trứng cá đỏ...)
- **Khoanh vùng tổn thương** bằng SAM2 (MedSAM2 fine-tune cho da liễu)
- **Chatbot RAG tư vấn mỹ phẩm** dùng OpenAI (`gpt-4o-mini`) + ChromaDB, dữ liệu sản phẩm từ `data/`

Toàn bộ hệ thống là **stateless** — không lưu session/chat history, NestJS backend chịu trách nhiệm quản lý hội thoại.

## 📋 Mục lục

- [Cấu trúc project](#-cấu-trúc-project)
- [Cài đặt](#-cài-đặt)
- [Chạy server](#-chạy-server)
- [API Endpoints](#-api-endpoints)
- [Ví dụ sử dụng](#-ví-dụ-sử-dụng)
- [Test](#-test)
- [Cập nhật dữ liệu sản phẩm](#-cập-nhật-dữ-liệu-sản-phẩm)
- [Triển khai](#-triển-khai)

## 📁 Cấu trúc project

```
skinalyze/
├── app/
│   ├── main.py        # FastAPI app, CORS, lifespan, router registration
│   ├── config.py       # Đường dẫn + cấu hình đọc từ .env
│   ├── state.py          # AppState (models đã load, RAG chain, LLM status)
│   ├── schemas.py          # Pydantic request/response models
│   ├── ml.py                 # Load model classification/segmentation/face-detection
│   ├── rag.py                  # Vector store, RAG chain, logic gợi ý sản phẩm
│   └── routes/                   # health.py, chat.py, analysis.py, disease.py
├── data/                # CSV sản phẩm + chunks đã format cho vector store
├── models/              # Model weights (Git LFS)
├── scripts/update_data.py  # Regenerate product_chunks.txt từ CSV
├── tests/test_api.py       # Script gọi thử các endpoint
├── .env.example
├── Dockerfile
└── requirements.txt
```

## 🔧 Cài đặt

```bash
python -m venv venv
# Windows: venv\Scripts\activate | macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Mở .env và điền OPENAI_API_KEY
```

Xem [.env.example](.env.example) để biết toàn bộ biến môi trường (bắt buộc + tuỳ chọn).

## 🚀 Chạy server

```bash
# Development (auto-reload)
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Server chạy tại `http://localhost:8000`. Swagger UI: `http://localhost:8000/docs`, ReDoc: `http://localhost:8000/redoc`.

## 📚 API Endpoints

### Health Check

**GET** `/` hoặc **GET** `/health`

```json
{
  "status": "healthy",
  "message": "AI Dermatology & Cosmetic API",
  "vectorstore_status": "ready",
  "classification_model_status": "loaded",
  "segmentation_model_status": "loaded",
  "llm_status": "ready",
  "llm_model": "gpt-4o-mini",
  "llm_error": null,
  "timestamp": "2026-09-24T10:30:00"
}
```

### Chat tư vấn (text + ảnh tuỳ chọn)

**POST** `/chat` — `multipart/form-data`

| Field | Bắt buộc | Ghi chú |
|---|---|---|
| `question` | ✅ | Câu hỏi của người dùng |
| `conversation_history` | ❌ | Chuỗi JSON: `[{"role":"user","content":"..."},{"role":"ai","content":"..."}]` |
| `image` | ❌ | File ảnh — nếu có, hệ thống chạy classifier để lấy thêm ngữ cảnh tình trạng da |

**Response:**
```json
{
  "answer": "Mình gợi ý cho bạn: 1. La Roche-Posay Toleriane...",
  "response_time": 2.5,
  "timestamp": "2026-09-24T10:30:00"
}
```

### Phân tích ảnh da → gợi ý sản phẩm

**POST** `/analyze-image` — `multipart/form-data`: `image` (file, bắt buộc), `additional_text` (tuỳ chọn)

**POST** `/analyze-image-base64` — JSON: `{"image_base64": "...", "additional_text": "..."}`

Cả hai dùng classifier EfficientNet-B0 để xác định tình trạng da rồi truyền qua RAG chain để tư vấn sản phẩm.

```json
{
  "skin_analysis": "Tình trạng da nhận diện được: Mụn (Acne) (độ tin cậy 87%)",
  "product_recommendation": "Dạ, mình gợi ý sản phẩm cho mụn nhé: ...",
  "severity_warning": null,
  "response_time": 3.1,
  "timestamp": "2026-09-24T10:30:00"
}
```

### Phân loại bệnh da (không qua RAG)

**POST** `/api/classification-disease` — `multipart/form-data`: `file` (bắt buộc), `notes` (tuỳ chọn, truyền `"facial"` để bắt buộc kiểm tra có khuôn mặt trong ảnh trước khi phân loại)

```json
{
  "predicted_class": "Acne",
  "confidence": 0.87,
  "all_predictions": {"Acne": 0.87, "Normal": 0.05, "...": "..."},
  "product_suggestions": ["CeraVe Foaming Facial Cleanser", "..."]
}
```

### Khoanh vùng tổn thương (segmentation)

**POST** `/api/segmentation-disease` — `multipart/form-data`: `file` (bắt buộc)

Trả về `mask` (PNG base64), `lesion_on_black` (JPEG base64 — vùng tổn thương trên nền đen), `original_size`, `confidence`.

### Kiểm tra có khuôn mặt trong ảnh

**POST** `/api/face-detection` — `multipart/form-data`: `file` (bắt buộc)

```json
{ "has_face": true }
```

## 💻 Ví dụ sử dụng

### cURL

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -F "question=Tôi cần kem dưỡng cho da khô nhạy cảm"

curl -X POST http://localhost:8000/analyze-image \
  -F "image=@my_skin.jpg" \
  -F "additional_text=Tư vấn sản phẩm cho tôi"
```

### Python (requests)

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    data={"question": "Tôi cần serum cho da dầu mụn"}
)
print(response.json())

with open("my_skin.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze-image",
        files={"image": f},
        data={"additional_text": "Tư vấn giúp tôi"}
    )
print(response.json())
```

## 🧪 Test

```bash
# Server phải đang chạy tại http://localhost:8000
python tests/test_api.py
```

## 🔄 Cập nhật dữ liệu sản phẩm

Khi thay đổi `data/fixed_cosmetic.csv`, chạy lại script này để tạo lại `data/product_chunks.txt` và xoá vector store cũ (sẽ tự build lại khi server khởi động):

```bash
python scripts/update_data.py
```

## 🚢 Triển khai

Repo được cấu hình để deploy trực tiếp lên **Hugging Face Spaces** (Docker SDK, xem header YAML ở đầu file này) — chỉ cần đặt `OPENAI_API_KEY` trong **Settings → Secrets** của Space.

Xem [DEPLOY.md](DEPLOY.md) để biết chi tiết deploy bằng Docker.
