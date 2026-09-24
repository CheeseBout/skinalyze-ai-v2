"""Pydantic request/response models for the API."""

from typing import Dict, List, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    answer: str
    response_time: float
    timestamp: str


class ImageAnalysisRequest(BaseModel):
    image_base64: str
    additional_text: Optional[str] = None


class ImageAnalysisResponse(BaseModel):
    skin_analysis: str
    product_recommendation: str
    severity_warning: Optional[str] = None
    response_time: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    message: str
    vectorstore_status: str
    classification_model_status: str
    segmentation_model_status: str
    llm_status: str
    llm_model: str
    llm_error: Optional[str] = None
    timestamp: str
