"""RAG chatbot endpoint: text + optional image (classifier-based) + product recommendations."""

import io
import json
import time
from datetime import datetime
from typing import Optional

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from app.ml import IMAGE_TRANSFORMS, SKIN_CLASSES, device
from app.rag import build_classification_analysis_query, detect_skin_condition_and_types
from app.schemas import ChatResponse
from app.state import state

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    question: str = Form(...),
    conversation_history: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Stateless chat endpoint supporting Text + Optional Image (classifier-based) + Intelligent Product Recommendations.
    """
    if state.rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    try:
        start_time = time.time()

        # 1. Parse conversation history
        history_list = []
        if conversation_history:
            try:
                history_list = json.loads(conversation_history)
            except json.JSONDecodeError:
                print("Failed to parse conversation_history JSON")
                history_list = []

        # 2. Image analysis via EfficientNet classifier (If image is provided)
        image_context_str = ""
        if image:
            if state.classification_model is None:
                raise HTTPException(status_code=503, detail="Classification model not loaded")

            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Uploaded file is not an image")

            image_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            input_tensor = IMAGE_TRANSFORMS['classification'](pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = state.classification_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            pred_index = predicted.item()
            if pred_index < len(SKIN_CLASSES):
                predicted_class = SKIN_CLASSES[pred_index]
                confidence_score = float(confidence.item())
                skin_analysis, _ = build_classification_analysis_query(
                    predicted_class, confidence_score
                )

                image_context_str = f"""
\n[THÔNG TIN TỪ ẢNH NGƯỜI DÙNG GỬI KÈM]:
Hệ thống đã phân tích ảnh da của người dùng với kết quả sau:
{skin_analysis}
-----------------------------------
"""

        # 3. Intelligent Product Recommendation Logic
        detected_condition, suitable_skin_types = detect_skin_condition_and_types(question)

        condition_context_str = ""
        if detected_condition:
            skin_types_str = ", ".join(suitable_skin_types) if suitable_skin_types else "mọi loại da"
            condition_context_str = f"""
[HỆ THỐNG PHÁT HIỆN VẤN ĐỀ DA]:
- Vấn đề phát hiện: {detected_condition}
- Loại da phù hợp để tư vấn sản phẩm: {skin_types_str}
- ƯU TIÊN tìm kiếm và gợi ý các sản phẩm trong database dành cho: {skin_types_str}
-----------------------------------
"""

        # 4. Build Context from History
        context_str = ""
        if history_list:
            context_pairs = []
            for i in range(0, len(history_list) - 1, 2):
                if i + 1 < len(history_list):
                    user_msg = history_list[i]
                    ai_msg = history_list[i + 1]
                    if user_msg.get('role') == 'user' and ai_msg.get('role') == 'ai':
                        context_pairs.append((
                            user_msg.get('content', ''),
                            ai_msg.get('content', '')
                        ))

            if context_pairs:
                recent = context_pairs[-3:]
                context_str = "LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ:\n" + "\n".join([
                    f"User: {ctx[0]}\nAI: {ctx[1][:200]}..."
                    for ctx in recent
                ]) + "\n"

        # 5. Construct Final Prompt for RAG
        full_query = f"""{context_str}
{image_context_str}
{condition_context_str}

CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÙNG: {question}

Yêu cầu: Hãy trả lời câu hỏi của người dùng dựa trên thông tin sản phẩm có trong database.
Nếu có thông tin từ ảnh hoặc vấn đề da được phát hiện, hãy sử dụng nó để lọc và tư vấn sản phẩm chính xác hơn."""

        # 6. Invoke RAG Chain
        response = state.rag_chain.invoke(full_query)

        return ChatResponse(
            answer=response,
            response_time=round(time.time() - start_time, 2),
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
