# Sử dụng Python 3.10
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt thư viện hệ thống
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tạo các thư mục cần thiết
RUN mkdir -p data models db_chroma chat_history

# --- FIX SAM2 CONFIG: Tải file config về đặt sẵn ---
# Tạo thư mục configs chuẩn của SAM2 để tránh lỗi tìm kiếm
RUN mkdir -p /usr/local/lib/python3.10/site-packages/sam2/configs/sam2.1/
RUN curl -L -o /usr/local/lib/python3.10/site-packages/sam2/configs/sam2.1/sam2.1_hiera_t.yaml https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml

# FIX LỖI PERMISSION
ENV MPLCONFIGDIR=/tmp

# Cấp quyền và User
RUN chown -R 1000:1000 /app
USER 1000

# Copy code
COPY --chown=1000:1000 . .

# Expose và Run
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]