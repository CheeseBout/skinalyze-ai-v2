"""
Cấu hình tập trung cho toàn bộ ứng dụng.
Mọi đường dẫn thư mục/file và tham số model đều đọc từ đây — không hardcode
rải rác ở các module khác. Biến môi trường được nạp 1 lần duy nhất ở đây
(qua load_dotenv()); các module khác chỉ import hằng số từ module này.

Xem .env.example ở thư mục gốc repo để biết danh sách đầy đủ biến môi trường.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# ĐƯỜNG DẪN
# =============================================================================
# app/config.py -> app/ -> repo root
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_DIR = BASE_DIR / "db_chroma"

CHUNKS_FILE = DATA_DIR / "product_chunks.txt"
CSV_FILE = DATA_DIR / "fixed_cosmetic.csv"

CLASSIFICATION_MODEL_PATH = MODELS_DIR / "efficientnet_b0_complete.pt"
SEGMENTATION_MODEL_PATH = MODELS_DIR / "medsam2_dermatology_best_aug2.pth"

# =============================================================================
# OPENAI (LLM + embeddings)
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.05"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1500"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "90"))

# =============================================================================
# ỨNG DỤNG
# =============================================================================
# Tỷ giá USD -> VND (dùng để hiển thị giá sản phẩm)
USD_TO_VND = int(os.getenv("USD_TO_VND", "26349"))

# CORS: mặc định "*" (mọi origin). Đặt CORS_ORIGINS="https://a.com,https://b.com"
# trong .env để giới hạn origin ở production.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
