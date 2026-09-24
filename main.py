"""
AI Dermatology & Cosmetic Consultant API
Stateless API for NestJS Backend Integration
"""

# =============================================================================
# IMPORTS
# =============================================================================
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import numpy as np
from typing import Dict, Optional, List
import os
import time
import json
from datetime import datetime
import mediapipe as mp
from dotenv import load_dotenv

load_dotenv()

# Ensure RAG_cosmetic.py is in the same directory
from RAG_cosmetic import (
    setup_api_key,
    load_or_create_vectorstore,
    setup_rag_chain,
    build_classification_analysis_query,
    detect_skin_condition_and_types,
    get_product_suggestions_by_skin_types,
    map_disease_to_skin_types,
    test_llm_connection,
    OPENAI_LLM_MODEL,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
# ✅ UPDATED: 11 classes (khớp với training notebook)
SKIN_CLASSES = [
    'Acne',
    'Actinic_Keratosis',
    'Drug_Eruption',  # ← ✅ THÊM LẠI
    'Eczema',
    'Normal',
    'Psoriasis',
    'Rosacea',
    'Seborrh_Keratoses',
    'Sun_Sunlight_Damage',
    'Tinea',
    'Warts'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    'classification': os.path.join(BASE_DIR, "models/efficientnet_b0_complete.pt"),
    'segmentation': os.path.join(BASE_DIR, "models/medsam2_dermatology_best_aug2.pth")
}

IMAGE_TRANSFORMS = {
    'classification': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# =============================================================================
# GLOBAL STATE
# =============================================================================
class AppState:
    rag_chain = None
    classification_model = None
    segmentation_model = None
    face_detector = None
    vectorstore = None

    # --- LLM (RAG chat) status ---
    llm_ready: bool = False
    llm_error: Optional[str] = None
    llm_model: str = OPENAI_LLM_MODEL

state = AppState()

# =============================================================================
# MODEL LOADING
# =============================================================================
def load_classification_model():
    """
    ✅ FIXED: Load EfficientNet-B0 với architecture khớp training notebook
    Architecture: 1280 → [Dropout 0.4] → 512 → [BN + ReLU + Dropout 0.3] → 256 → [BN + ReLU] → 11
    """
    model_path = MODEL_PATHS['classification']
    if not os.path.exists(model_path):
        print(f"⚠️  Classification model not found at {model_path}")
        return None

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # 1. If checkpoint is a full model object (unlikely)
        if not isinstance(checkpoint, dict):
            print("ℹ️  Checkpoint is a full model object.")
            model = checkpoint
            model.to(device)
            model.eval()
            return model

        # 2. Extract state_dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # ✅ Get architecture config from checkpoint
        config = checkpoint.get('config', {})
        dropout1 = config.get('dropout1', 0.4)  # Default from notebook
        dropout2 = config.get('dropout2', 0.3)  # Default from notebook
        num_classes = checkpoint.get('num_classes', len(SKIN_CLASSES))

        print(f"ℹ️  Loading model with config:")
        print(f"   - Num classes: {num_classes}")
        print(f"   - Dropout 1: {dropout1}")
        print(f"   - Dropout 2: {dropout2}")

        # Initialize EfficientNet-B0 base
        model = models.efficientnet_b0(weights=None)

        # ✅ Reconstruct EXACT classifier from training notebook
        num_features = 1280  # EfficientNet-B0 default

        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout1),              # Layer 0: Dropout 0.4
            nn.Linear(num_features, 512),         # Layer 1: 1280 → 512
            nn.BatchNorm1d(512),                  # Layer 2: BatchNorm
            nn.ReLU(),                            # Layer 3: ReLU
            nn.Dropout(p=dropout2),              # Layer 4: Dropout 0.3
            nn.Linear(512, 256),                  # Layer 5: 512 → 256
            nn.BatchNorm1d(256),                  # Layer 6: BatchNorm
            nn.ReLU(),                            # Layer 7: ReLU
            nn.Linear(256, num_classes)          # Layer 8: 256 → 11
        )

        # Load trained weights
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        print(f"✅ Classification model loaded successfully")
        print(f"   Architecture: 1280 → 512 → 256 → {num_classes}")

        return model

    except Exception as e:
        print(f"❌ Error loading classification model: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_segmentation_model():
    """Load SAM2 segmentation model (Fixed Relative Path for Hydra)"""
    model_path = MODEL_PATHS['segmentation']
    if not os.path.exists(model_path):
        print(f"⚠️  Segmentation model not found")
        return None

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        config_file = "configs/sam2.1/sam2.1_hiera_t.yaml"
        print(f"ℹ️ Loading SAM2 with config: {config_file}")

        sam2_model = build_sam2(
            config_file=config_file,
            ckpt_path=None,
            device=device,
            mode='eval',
            apply_postprocessing=False
        )

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            sam2_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        predictor = SAM2ImagePredictor(sam2_model)
        print("✅ SAM2 segmentation model loaded")
        return predictor

    except Exception as e:
        print(f"❌ Error loading segmentation model: {e}")
        return None

def load_face_detection_model():
    """Load Mediapipe Face Detection model"""
    try:
        mp_face_detection = mp.solutions.face_detection
        detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        print("✅ Face detection model loaded")
        return detector
    except Exception as e:
        print(f"❌ Error loading face detection model: {e}")
        return None

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================
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

# =============================================================================
# HEALTH CHECK
# =============================================================================
@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
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

# =============================================================================
# RAG CHATBOT ENDPOINTS
# =============================================================================
@app.post("/chat", response_model=ChatResponse)
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
                print("⚠️ Failed to parse conversation_history JSON")
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

@app.post("/analyze-image", response_model=ImageAnalysisResponse)
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
            severity_warning=(
                "⚠️ Độ tin cậy nhận diện thấp — kết quả chỉ mang tính tham khảo, "
                "nên tham khảo thêm ý kiến bác sĩ da liễu để xác nhận."
            ) if is_low_confidence else None,
            response_time=round(time.time() - start_time, 2),
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/analyze-image-base64", response_model=ImageAnalysisResponse)
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
            severity_warning=(
                "⚠️ Độ tin cậy nhận diện thấp — kết quả chỉ mang tính tham khảo, "
                "nên tham khảo thêm ý kiến bác sĩ da liễu để xác nhận."
            ) if is_low_confidence else None,
            response_time=round(time.time() - start_time, 2),
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# =============================================================================
# CLASSIFICATION & SEGMENTATION ENDPOINTS
# =============================================================================
@app.post("/api/classification-disease")
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
                results = state.face_detector.process(image_np)

                if not results.detections:
                     raise HTTPException(
                         status_code=400,
                         detail="No face detected. Please upload a clear image of a face for facial analysis."
                     )
            else:
                print("⚠️ Face detector skipped (not loaded)")

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

@app.post("/api/segmentation-disease")
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

# =============================================================================
# FACE DETECTION ENDPOINT
# =============================================================================
@app.post("/api/face-detection")
async def face_detection(file: UploadFile = File(...)) -> Dict[str, bool]:
    """
    Checks if the image contains a face.
    Returns: {"has_face": boolean}
    """
    if state.face_detector is None:
        print("⚠️  Face detection model not loaded")
        return {"has_face": True}
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        image_np = np.array(image)

        results = state.face_detector.process(image_np)

        if results.detections:
            return {"has_face": True}
        return {"has_face": False}
    except Exception as e:
        print(f"❌ Error during face detection: {e}")
        return {"has_face": False}

# =============================================================================
# RUN SERVER
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)