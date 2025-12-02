# Sử dụng Python 3.10
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và Git
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tạo các thư mục cần thiết để tránh lỗi permission
RUN mkdir -p data models db_chroma chat_history

# Cấp quyền cho user 1000 (Hugging Face yêu cầu chạy non-root)
RUN chown -R 1000:1000 /app

# Chuyển sang user non-root
USER 1000

# Copy toàn bộ code vào container (làm bước cuối để tận dụng cache Docker)
COPY --chown=1000:1000 . .

# Mở port 7860 (Port bắt buộc của Hugging Face Spaces)
EXPOSE 7860

# Lệnh chạy server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]