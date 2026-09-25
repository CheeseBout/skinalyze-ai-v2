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

API RESTful cho hệ thống phân tích da liễu và tư vấn mỹ phẩm bằng AI, được thiết kế làm backend AI đứng sau một backend NestJS (NestJS quản lý user/session, gọi API này để lấy kết quả phân tích và tư vấn).

Ba khối chức năng chính:

- 🔬 **Phân loại bệnh da** từ ảnh bằng EfficientNet-B0 tự host — 11 lớp (mụn, chàm, vảy nến, trứng cá đỏ, nấm da...)
- ✂️ **Khoanh vùng tổn thương** (segmentation) bằng SAM2 — MedSAM2 fine-tune cho da liễu
- 💬 **Chatbot RAG tư vấn mỹ phẩm** — OpenAI `gpt-4o-mini` + ChromaDB, truy vấn trên dữ liệu sản phẩm trong `data/`

Toàn bộ API **stateless**: không lưu session hay chat history phía server — client (NestJS) tự quản lý và gửi kèm `conversation_history` nếu cần.

## 📋 Mục lục

- [Tech stack](#-tech-stack)
- [Cấu trúc project](#-cấu-trúc-project)
- [Cài đặt](#-cài-đặt)
- [Chạy server](#-chạy-server)
- [API Endpoints](#-api-endpoints)
- [Ví dụ sử dụng](#-ví-dụ-sử-dụng)
- [Test](#-test)
- [Cập nhật dữ liệu sản phẩm](#-cập-nhật-dữ-liệu-sản-phẩm)
- [Triển khai](#-triển-khai)
- [Troubleshooting](#-troubleshooting)

## 🛠 Tech stack

| Thành phần | Công nghệ |
|---|---|
| Runtime | Python 3.12.10 |
| Web framework | FastAPI + Uvicorn |
| LLM (chatbot) | OpenAI `gpt-4o-mini` qua LangChain |
| Vector store | ChromaDB, embedding `sentence-transformers/all-MiniLM-L6-v2` |
| Phân loại bệnh da | PyTorch — EfficientNet-B0 fine-tune, tự host (`models/efficientnet_b0_complete.pt`) |
| Segmentation tổn thương | SAM2 — MedSAM2 fine-tune, tự host (`models/medsam2_dermatology_best_aug2.pth`) |
| Face detection | OpenCV Haar cascade (bundled trong `opencv-python-headless`, không cần model rời) |
| Deploy | Docker, chạy trên Hugging Face Spaces |

## 📁 Cấu trúc project

```
skinalyze/
├── app/
│   ├── main.py         # FastAPI app, CORS, lifespan (load model + build RAG chain), đăng ký router
│   ├── config.py        # Nguồn cấu hình duy nhất: đường dẫn + biến môi trường (đọc .env)
│   ├── state.py          # AppState — models đã load, RAG chain, trạng thái LLM (singleton)
│   ├── schemas.py          # Pydantic request/response models
│   ├── ml.py                 # Load model classification/segmentation/face-detection + transforms
│   ├── rag.py                  # Vector store, RAG chain, phát hiện bệnh da, gợi ý sản phẩm
│   └── routes/
│       ├── health.py             # GET /, GET /health
│       ├── chat.py                # POST /chat
│       ├── analysis.py             # POST /analyze-image, /analyze-image-base64
│       └── disease.py               # POST /api/classification-disease, /api/segmentation-disease, /api/face-detection
├── data/
│   ├── fixed_cosmetic.csv    # Dữ liệu sản phẩm gốc
│   └── product_chunks.txt     # Đã format sẵn cho vector store (sinh ra từ CSV)
├── models/                # Model weights — theo dõi bằng Git LFS (*.pt, *.pth)
├── scripts/update_data.py   # Sinh lại product_chunks.txt từ CSV
├── tests/test_api.py          # Script gọi thử các endpoint
├── .env.example
├── Dockerfile
└── requirements.txt
```

## 🔧 Cài đặt

**Yêu cầu Python 3.12.10** (xem [.python-version](.python-version)).

```bash
python -m venv venv
# Windows: venv\Scripts\activate | macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Mở .env và điền OPENAI_API_KEY
```

> **Windows:** bước `pip install` sẽ build `chroma-hnswlib` từ source (chưa có
> wheel dựng sẵn cho Python 3.12 trên Windows) — cần cài **[Microsoft C++ Build
> Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)** (workload
> "Desktop development with C++") trước, nếu chưa có. Linux/macOS không cần bước
> này (đã có wheel dựng sẵn), Docker build tự cài `build-essential`.

`.env.example` liệt kê toàn bộ biến môi trường:

| Biến | Bắt buộc | Mặc định |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | — |
| `OPENAI_MODEL` | ❌ | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | ❌ | `sentence-transformers/all-MiniLM-L6-v2` |
| `HF_TOKEN` | ❌ | — (tránh cảnh báo rate limit khi tải `EMBEDDING_MODEL` lần đầu — xem [hướng dẫn lấy token](https://huggingface.co/settings/tokens)) |
| `LLM_TEMPERATURE` | ❌ | `0.05` |
| `LLM_MAX_TOKENS` | ❌ | `1500` |
| `LLM_MAX_RETRIES` | ❌ | `3` |
| `LLM_TIMEOUT` | ❌ | `90` |
| `USD_TO_VND` | ❌ | `26349` |
| `CORS_ORIGINS` | ❌ | `*` (comma-separated nếu giới hạn domain) |

## 🚀 Chạy server

```bash
# Development (auto-reload)
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Server chạy tại `http://localhost:8000`. Swagger UI: `http://localhost:8000/docs`, ReDoc: `http://localhost:8000/redoc`.

> Lần chạy đầu tiên sẽ tự tải embedding model và build vector store từ `data/product_chunks.txt` vào `db_chroma/` (mất vài phút) — các lần sau load lại từ cache.

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

`status` là `"healthy"` chỉ khi LLM sẵn sàng **và** cả 2 model classification/segmentation đã load; ngược lại là `"degraded"` — vẫn phục vụ được các phần còn hoạt động.

### Chat tư vấn (text + ảnh tuỳ chọn)

**POST** `/chat` — `multipart/form-data`

| Field | Bắt buộc | Ghi chú |
|---|---|---|
| `question` | ✅ | Câu hỏi của người dùng |
| `conversation_history` | ❌ | Chuỗi JSON: `[{"role":"user","content":"..."},{"role":"ai","content":"..."}]` — chỉ 3 lượt gần nhất được dùng làm ngữ cảnh |
| `image` | ❌ | File ảnh — nếu có, hệ thống chạy classifier để lấy thêm ngữ cảnh tình trạng da trước khi trả lời |

Nếu câu hỏi chứa tên bệnh da đã hỗ trợ (xem bảng bên dưới), hệ thống tự ưu tiên tìm sản phẩm phù hợp cho loại da tương ứng, bỏ qua lịch sử hội thoại để tránh nhầm lẫn.

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

Cả hai dùng classifier EfficientNet-B0 tự host để xác định tình trạng da (không gọi vision model bên thứ 3 — các model đó tự từ chối phân tích ảnh khuôn mặt người thật), rồi truyền kết quả qua RAG chain để tư vấn sản phẩm bằng văn bản.

```json
{
  "skin_analysis": "Tình trạng da nhận diện được: Mụn (Acne) (độ tin cậy 87%)",
  "product_recommendation": "Dạ, mình gợi ý sản phẩm cho mụn nhé: ...",
  "severity_warning": null,
  "response_time": 3.1,
  "timestamp": "2026-09-24T10:30:00"
}
```

`severity_warning` chỉ xuất hiện khi độ tin cậy nhận diện < 50%, khuyến nghị người dùng tham khảo thêm bác sĩ da liễu.

### Phân loại bệnh da (không qua RAG)

**POST** `/api/classification-disease` — `multipart/form-data`: `file` (bắt buộc), `notes` (tuỳ chọn — truyền `"facial"` để bắt buộc kiểm tra có khuôn mặt trong ảnh trước khi phân loại, trả lỗi 400 nếu không thấy mặt)

```json
{
  "predicted_class": "Acne",
  "confidence": 0.87,
  "all_predictions": {"Acne": 0.87, "Normal": 0.05, "...": "..."},
  "product_suggestions": ["CeraVe Foaming Facial Cleanser", "..."]
}
```

11 lớp mà classifier nhận diện (`predicted_class`):

| Class | Tên tiếng Việt |
|---|---|
| `Acne` | Mụn |
| `Actinic_Keratosis` | Dày sừng ánh sáng |
| `Drug_Eruption` | Phát ban do thuốc |
| `Eczema` | Chàm |
| `Normal` | Da bình thường |
| `Psoriasis` | Vảy nến |
| `Rosacea` | Trứng cá đỏ |
| `Seborrh_Keratoses` | Viêm da tiết bã |
| `Sun_Sunlight_Damage` | Tổn thương do nắng |
| `Tinea` | Nấm da |
| `Warts` | Mụn cóc |

### Khoanh vùng tổn thương (segmentation)

**POST** `/api/segmentation-disease` — `multipart/form-data`: `file` (bắt buộc)

```json
{
  "mask": "iVBORw0KGgo...",
  "lesion_on_black": "/9j/4AAQSkZJRg...",
  "format": "base64_png",
  "original_size": [1024, 768],
  "confidence": 0.93
}
```

`mask`: ảnh PNG base64 (grayscale mask). `lesion_on_black`: ảnh JPEG base64 — vùng tổn thương giữ nguyên, nền còn lại tô đen.

### Kiểm tra có khuôn mặt trong ảnh

**POST** `/api/face-detection` — `multipart/form-data`: `file` (bắt buộc)

```json
{ "has_face": true }
```

Nếu face detector chưa load được, endpoint trả `has_face: true` (fail-open) để không chặn luồng của client.

## 💻 Ví dụ sử dụng

### cURL

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -F "question=Tôi cần kem dưỡng cho da khô nhạy cảm"

curl -X POST http://localhost:8000/analyze-image \
  -F "image=@my_skin.jpg" \
  -F "additional_text=Tư vấn sản phẩm cho tôi"

curl -X POST http://localhost:8000/api/classification-disease \
  -F "file=@my_skin.jpg" \
  -F "notes=facial"
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

### JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append("question", "Tôi cần kem chống nắng cho da nhạy cảm");

fetch("http://localhost:8000/chat", { method: "POST", body: formData })
  .then((res) => res.json())
  .then((data) => console.log(data));
```

## 🧪 Test

```bash
# Server phải đang chạy tại http://localhost:8000
python tests/test_api.py
```

## 🔄 Cập nhật dữ liệu sản phẩm

Khi thay đổi `data/fixed_cosmetic.csv`, chạy lại script này để tạo lại `data/product_chunks.txt` và xoá vector store cũ (sẽ tự build lại từ chunks mới khi server khởi động):

```bash
python scripts/update_data.py
```

## 🚢 Triển khai

Repo được cấu hình để deploy trực tiếp lên **Hugging Face Spaces** (Docker SDK — xem header YAML ở đầu file này) — chỉ cần đặt `OPENAI_API_KEY` trong **Settings → Repository secrets** của Space.

Deploy bằng Docker thủ công:

```bash
docker build -t skinalyze .
docker run -p 7860:7860 --env-file .env skinalyze
```

Xem [DEPLOY.md](DEPLOY.md) để biết chi tiết đầy đủ (checklist production, bảo mật API key, xử lý sự cố khi deploy).

## 🆘 Troubleshooting

| Vấn đề | Kiểm tra |
|---|---|
| `/health` trả `llm_status: "error"` | `OPENAI_API_KEY` trong `.env`, xem `llm_error` trong response để biết lý do cụ thể |
| `503 RAG chain not initialized` | `data/product_chunks.txt` có tồn tại không, xem log lúc server khởi động |
| `503 Classification/Segmentation model not loaded` | File weight trong `models/` có tồn tại không (cần Git LFS pull đầy đủ) |

Xem thêm bảng troubleshooting chi tiết hơn trong [DEPLOY.md](DEPLOY.md).
