# 🚀 Hướng dẫn triển khai

## Chuẩn bị

```bash
cp .env.example .env
# Điền OPENAI_API_KEY vào .env — KHÔNG commit file này lên Git
```

`.env` đã nằm trong `.gitignore`. Xem [.env.example](.env.example) để biết toàn bộ biến môi trường.

## Deploy bằng Docker (khuyến nghị)

```bash
docker build -t skinalyze .
docker run -p 7860:7860 --env-file .env skinalyze
```

Server chạy tại `http://localhost:7860`.

## Deploy lên Hugging Face Spaces

Repo đã được cấu hình sẵn cho Spaces (xem header YAML ở đầu [README.md](README.md): `sdk: docker`, `app_port: 7860`).

1. Tạo Space mới, chọn SDK **Docker**.
2. Push repo này lên Space (qua Git remote hoặc kéo thả file trên web UI).
3. Vào **Settings → Repository secrets**, thêm `OPENAI_API_KEY`. Không đặt API key trong file `.env` khi deploy lên Spaces — dùng Secrets.
4. Space sẽ tự build Dockerfile và chạy `uvicorn app.main:app --host 0.0.0.0 --port 7860`.

## Chạy local không qua Docker

Yêu cầu Python 3.12.10 (xem [.python-version](.python-version)).

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

> **Windows:** `pip install` sẽ build `chroma-hnswlib` từ source (chưa có wheel
> dựng sẵn cho Python 3.12 trên Windows) — cần cài **[Microsoft C++ Build
> Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)** (workload
> "Desktop development with C++") trước khi `pip install`.

## 🔒 Bảo mật API key

**KHÔNG BAO GIỜ** commit file `.env` hoặc hard-code API key trong code:

```python
# ❌ SAI
OPENAI_API_KEY = "sk-..."  # NGUY HIỂM!
```

Nếu vô tình commit API key lên Git:

1. **Thu hồi ngay** key đó trên [platform.openai.com](https://platform.openai.com/api-keys) và tạo key mới.
2. Xoá key khỏi lịch sử Git bằng [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) hoặc `git filter-repo`, sau đó `git push --force`.

## Production checklist

- [ ] **CORS:** đặt `CORS_ORIGINS` trong `.env` thành domain cụ thể của NestJS backend/frontend thay vì `*`
- [ ] **HTTPS:** đặt server sau reverse proxy (Nginx, Caddy) hoặc dùng HTTPS có sẵn của platform (Spaces, Cloud Run...)
- [ ] **Rate limiting / Authentication:** endpoint hiện không có auth — nếu expose public, thêm API key/JWT hoặc rate limit ở tầng gateway/NestJS
- [ ] **Model weights:** đảm bảo Git LFS đã pull đầy đủ `models/*.pt`, `models/*.pth` trước khi build image

## 🆘 Troubleshooting

| Vấn đề | Nguyên nhân thường gặp |
|---|---|
| `GET /health` trả `llm_status: "error"` | Thiếu hoặc sai `OPENAI_API_KEY`, hoặc model bị đổi tên/khai tử — xem `llm_error` trong response |
| `503 RAG chain not initialized` | Vector store chưa khởi tạo được — kiểm tra `data/product_chunks.txt` tồn tại và log lúc startup |
| `503 Classification/Segmentation model not loaded` | File weight trong `models/` không tồn tại hoặc chưa được Git LFS pull |
| Lỗi khi build SAM2 trong Docker | Kiểm tra bước tải `sam2.1_hiera_t.yaml` trong `Dockerfile` chạy thành công (cần mạng lúc build) |
| `error: Microsoft Visual C++ 14.0 or greater is required` khi `pip install` trên Windows | Cần cài Microsoft C++ Build Tools — xem mục "Chạy local không qua Docker" ở trên |
