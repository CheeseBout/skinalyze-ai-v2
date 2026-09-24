"""Global in-process state: loaded models + RAG chain, shared across routers."""

from typing import Optional

from app.config import OPENAI_MODEL


class AppState:
    rag_chain = None
    classification_model = None
    segmentation_model = None
    face_detector = None
    vectorstore = None

    # --- LLM (RAG chat) status ---
    llm_ready: bool = False
    llm_error: Optional[str] = None
    llm_model: str = OPENAI_MODEL


state = AppState()
