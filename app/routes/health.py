from datetime import datetime

from fastapi import APIRouter

from app.schemas import HealthResponse
from app.state import state

router = APIRouter()


@router.get("/", response_model=HealthResponse)
@router.get("/health", response_model=HealthResponse)
async def health_check():
    core_ready = (
        state.llm_ready
        and state.classification_model is not None
        and state.segmentation_model is not None
    )
    return HealthResponse(
        status="healthy" if core_ready else "degraded",
        message="AI Dermatology & Cosmetic API",
        vectorstore_status="ready" if state.rag_chain else "not_initialized",
        classification_model_status="loaded" if state.classification_model else "not_loaded",
        segmentation_model_status="loaded" if state.segmentation_model else "not_loaded",
        llm_status="ready" if state.llm_ready else "error",
        llm_model=state.llm_model,
        llm_error=state.llm_error,
        timestamp=datetime.now().isoformat()
    )
