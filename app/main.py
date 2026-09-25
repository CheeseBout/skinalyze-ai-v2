"""
AI Dermatology & Cosmetic Consultant API
Stateless API for NestJS Backend Integration
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.ml import load_classification_model, load_face_detection_model, load_segmentation_model
from app.rag import load_or_create_vectorstore, setup_api_key, setup_rag_chain, test_llm_connection
from app.routes import routers
from app.state import state


# =============================================================================
# LIFESPAN (STARTUP/SHUTDOWN)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("\n" + "=" * 80)
    print("🚀 STARTING AI DERMATOLOGY & COSMETIC API SERVER")
    print("=" * 80)

    state.classification_model = load_classification_model()
    state.segmentation_model = load_segmentation_model()
    state.face_detector = load_face_detection_model()

    # --- 1. API Key setup ---
    try:
        setup_api_key()
    except Exception as e:
        state.llm_error = str(e)
        print(f"\n❌ API key setup failed: {e}\n")
    else:
        # --- 2. Vector store + RAG chain ---
        try:
            db, embeddings = load_or_create_vectorstore()
            if db is None:
                raise RuntimeError("Vector store not initialized")
            state.vectorstore = db
            state.rag_chain = setup_rag_chain(db)
            print("\n✅ RAG chain constructed")
        except Exception as e:
            state.llm_error = str(e)
            print(f"\n❌ Error initializing RAG chain: {e}\n")

        # --- 3. LLM self-test (chỉ chạy nếu rag_chain đã dựng xong) ---
        # Constructing ChatOpenAI() KHÔNG gọi API thật, nên phải tự test
        # 1 request nhỏ để phát hiện model bị đổi/khai tử hoặc sai API key
        # ngay lúc khởi động, thay vì chờ user gặp lỗi 500.
        if state.rag_chain is not None:
            try:
                print(f"\n🧪 Testing LLM connection ({state.llm_model})...")
                test_llm_connection()
                state.llm_ready = True
                state.llm_error = None
                print("✅ LLM self-test passed — RAG Chatbot ready")
            except Exception as e:
                state.llm_ready = False
                state.llm_error = str(e)
                print(f"❌ LLM self-test failed: {e}")

    print("\n✅ Server ready!")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("=" * 80 + "\n")

    yield

    print("Shutting down models...")

# =============================================================================
# FASTAPI APP DEFINITION
# =============================================================================
app = FastAPI(
    title="AI Dermatology & Cosmetic Consultant API",
    description="Stateless API for skin disease classification, segmentation, and cosmetic consultation",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in routers:
    app.include_router(router)

# Run with `python main.py` (repo root) or `uvicorn app.main:app` — not by
# executing this file directly, which breaks the `app` package import.
