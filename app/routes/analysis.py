"""Image analysis endpoints: classifier-based skin condition detection + RAG recommendation."""

import base64
import io
import time
from datetime import datetime
from typing import Optional

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from app.ml import IMAGE_TRANSFORMS, SKIN_CLASSES, device
from app.rag import build_classification_analysis_query
from app.schemas import ImageAnalysisRequest, ImageAnalysisResponse
from app.state import state

router = APIRouter()

LOW_CONFIDENCE_WARNING = (
    "Độ tin cậy nhận diện thấp — kết quả chỉ mang tính tham khảo, "
    "nên tham khảo thêm ý kiến bác sĩ da liễu để xác nhận."
)


@router.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image_endpoint(
    image: UploadFile = File(...),
    additional_text: Optional[str] = Form(None)
):
    """
    v4.2.0: Dùng classifier EfficientNet-B0 tự host thay vì VLM bên thứ 3
    để xác định tình trạng da (OpenAI + Gemini đều từ chối phân tích ảnh
    khuôn mặt người thật ở tầng model, không thể fix bằng đổi prompt).
    LLM chỉ còn dùng cho bước tư vấn sản phẩm bằng văn bản.
    """
    if state.rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    if state.classification_model is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded")

    try:
        start_time = time.time()
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        input_tensor = IMAGE_TRANSFORMS['classification'](pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = state.classification_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        pred_index = predicted.item()
        if pred_index >= len(SKIN_CLASSES):
            raise HTTPException(status_code=400, detail="Cannot analyze image")

        predicted_class = SKIN_CLASSES[pred_index]
        confidence_score = float(confidence.item())
        is_low_confidence = confidence_score < 0.5

        skin_analysis, rag_query = build_classification_analysis_query(
            predicted_class, confidence_score, additional_text
        )
        product_recommendation = state.rag_chain.invoke(rag_query)

        return ImageAnalysisResponse(
            skin_analysis=skin_analysis,
            product_recommendation=product_recommendation,
            severity_warning=LOW_CONFIDENCE_WARNING if is_low_confidence else None,
            response_time=round(time.time() - start_time, 2),
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/analyze-image-base64", response_model=ImageAnalysisResponse)
async def analyze_image_base64_endpoint(request: ImageAnalysisRequest):
    """
    v4.2.0: Dùng classifier EfficientNet-B0 tự host — xem ghi chú ở
    /analyze-image phía trên.
    """
    if state.rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    if state.classification_model is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded")

    try:
        start_time = time.time()
        image_data = request.image_base64
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        input_tensor = IMAGE_TRANSFORMS['classification'](pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = state.classification_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        pred_index = predicted.item()
        if pred_index >= len(SKIN_CLASSES):
            raise HTTPException(status_code=400, detail="Cannot analyze image")

        predicted_class = SKIN_CLASSES[pred_index]
        confidence_score = float(confidence.item())
        is_low_confidence = confidence_score < 0.5

        skin_analysis, rag_query = build_classification_analysis_query(
            predicted_class, confidence_score, request.additional_text
        )
        product_recommendation = state.rag_chain.invoke(rag_query)

        return ImageAnalysisResponse(
            skin_analysis=skin_analysis,
            product_recommendation=product_recommendation,
            severity_warning=LOW_CONFIDENCE_WARNING if is_low_confidence else None,
            response_time=round(time.time() - start_time, 2),
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
