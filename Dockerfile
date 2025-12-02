# Sử dụng Python 3.10
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống (SỬA Ở ĐÂY: đổi libgl1-mesa-glx thành libgl1)
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tạo các thư mục cần thiết
RUN mkdir -p data models db_chroma chat_history

# Cấp quyền cho user 1000
RUN chown -R 1000:1000 /app

# Chuyển sang user non-root
USER 1000

# Copy toàn bộ code vào container
COPY --chown=1000:1000 . .

# Mở port 7860
EXPOSE 7860

# Lệnh chạy server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]