"""Classification, segmentation and face-detection endpoints used by the NestJS backend."""

import base64
import io
from typing import Dict, Optional

import numpy as np
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from app.ml import IMAGE_TRANSFORMS, SKIN_CLASSES, detect_faces, device
from app.rag import get_product_suggestions_by_skin_types, map_disease_to_skin_types
from app.state import state

router = APIRouter()


@router.post("/api/classification-disease")
async def classify_skin_disease(
    file: UploadFile = File(...),
    notes: Optional[str] = Form(None)
) -> Dict:
    """
    Classify skin disease using EfficientNet.
    Checks for face visibility ONLY if notes == 'facial'.
    Returns product suggestions based on detected disease.
    """
    if state.classification_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Conditional Face Detection
        if notes == 'facial':
            if state.face_detector:
                image_np = np.array(image)
                if not detect_faces(state.face_detector, image_np):
                     raise HTTPException(
                         status_code=400,
                         detail="No face detected. Please upload a clear image of a face for facial analysis."
                     )
            else:
                print("Face detector skipped (not loaded)")

        input_tensor = IMAGE_TRANSFORMS['classification'](image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = state.classification_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            all_probs = probabilities[0].cpu().numpy()

        # Safe Prediction Logic
        pred_index = predicted.item()
        if pred_index >= len(SKIN_CLASSES):
            return {
                "predicted_class": "Unknown",
                "confidence": float(confidence.item()),
                "note": "Model prediction index out of bounds for current class list",
                "product_suggestions": []
            }

        predicted_class = SKIN_CLASSES[pred_index]

        # Get product suggestions
        product_suggestions = []
        if state.vectorstore:
            # Map disease to skin types
            suitable_skin_types = map_disease_to_skin_types(predicted_class)

            # Get product suggestions
            product_suggestions = get_product_suggestions_by_skin_types(
                state.vectorstore,
                suitable_skin_types,
                num_products=5
            )

        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence.item()),
            "all_predictions": {SKIN_CLASSES[i]: float(all_probs[i]) for i in range(min(len(SKIN_CLASSES), len(all_probs)))},
            "product_suggestions": product_suggestions
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/segmentation-disease")
async def segment_skin_lesion(file: UploadFile = File(...)) -> Dict:
    if state.segmentation_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size
        image_np = np.array(image)

        state.segmentation_model.set_image(image_np)

        with torch.no_grad():
            masks, scores, _ = state.segmentation_model.predict(
                point_coords=None, point_labels=None, box=None, multimask_output=False
            )

        mask = masks[0] if len(masks) > 0 else np.zeros(image_np.shape[:2], dtype=np.uint8)

        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        black_bg = Image.new("RGB", original_size, (0, 0, 0))
        mask_pil = Image.fromarray(mask).convert("L")

        if mask_pil.size != original_size:
            mask_pil = mask_pil.resize(original_size, Image.NEAREST)

        full_image_on_black = Image.composite(image, black_bg, mask_pil)
        buffer_black_bg = io.BytesIO()
        full_image_on_black.save(buffer_black_bg, format="JPEG", quality=90)
        black_bg_base64 = base64.b64encode(buffer_black_bg.getvalue()).decode("utf-8")

        mask_image = Image.fromarray(mask)
        if mask_image.size != original_size:
            mask_image = mask_image.resize(original_size, Image.NEAREST)

        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "mask": mask_base64,
            "lesion_on_black": black_bg_base64,
            "format": "base64_png",
            "original_size": original_size,
            "confidence": float(scores[0]) if len(scores) > 0 else 0.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/face-detection")
async def face_detection(file: UploadFile = File(...)) -> Dict[str, bool]:
    """
    Checks if the image contains a face.
    Returns: {"has_face": boolean}
    """
    if state.face_detector is None:
        print("Face detection model not loaded")
        return {"has_face": True}
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        image_np = np.array(image)

        return {"has_face": detect_faces(state.face_detector, image_np)}
    except Exception as e:
        print(f"Error during face detection: {e}")
        return {"has_face": False}
