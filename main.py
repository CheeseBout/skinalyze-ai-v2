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
    analyze_skin_image,
    check_severity,
    build_image_analysis_query,
    detect_skin_condition_and_types,
    get_product_suggestions_by_skin_types,
    map_disease_to_skin_types,
    map_skin_condition_to_types
)

# =============================================================================
# CONFIGURATION
# =============================================================================
# UPDATED: Exactly 10 classes (Removed Drug_Eruption)
SKIN_CLASSES = [
    'Acne', 
    'Actinic_Keratosis', 
    'Eczema', 
    'Normal', 
    'Psoriasis', 
    'Rosacea', 
    'Seborrh_Keratoses', 
    'Sun_Sunlight_Damage', 
    'Tinea', 
    'Warts'
]

SKIN_CONDITION_CLASSES = ['Dry', 'Normal', 'Oily']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    'classification': os.path.join(BASE_DIR, "models/efficientnet_b0_complete.pt"),
    'segmentation': os.path.join(BASE_DIR, "models/medsam2_dermatology_best_aug2.pth"),
    'skin_condition': os.path.join(BASE_DIR, "models/efficient-net-skin-conditions-classifier.pth")
}

IMAGE_TRANSFORMS = {
    'classification': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'condition': transforms.Compose([
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
    skin_condition_model = None
    face_detector = None
    vectorstore = None  # Thêm để lưu vector store

state = AppState()

# =============================================================================
# MODEL LOADING
# =============================================================================
def load_classification_model():
    """
    Robust loader for EfficientNet-B0 classification model.
    Matches the custom architecture:
    Dropout -> Linear(512) -> BN -> ReLU -> Dropout -> Linear(256) -> BN -> ReLU -> Linear(10)
    """
    model_path = MODEL_PATHS['classification']
    if not os.path.exists(model_path):
        print(f"⚠️  Classification model not found at {model_path}")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 1. If checkpoint is a full model object
        if not isinstance(checkpoint, dict):
            print("ℹ️  Checkpoint is a full model object.")
            model = checkpoint
            model.to(device)
            model.eval()
            return model

        # 2. If checkpoint is a state dict
        state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
        
        # Initialize Base Model
        model = models.efficientnet_b0(weights=None)
        
        # Reconstruct Custom Classifier Head
        # Based on your logic: 1280 -> 512 -> 256 -> 10
        # Note: Dropout p values (0.3, 0.5 etc) don't affect weight loading, only training.
        
        try:
            # Attempt to dynamically size based on saved weights if possible
            # Usually: classifier.1.weight (512, 1280)
            #          classifier.5.weight (256, 512)
            #          classifier.8.weight (10, 256)
            
            linear1_in = 1280 # EfficientNet-B0 default
            linear1_out = 512
            linear2_out = 256
            num_classes = len(SKIN_CLASSES) # Should be 10

            if 'classifier.1.weight' in state_dict:
                linear1_out = state_dict['classifier.1.weight'].shape[0]
            if 'classifier.5.weight' in state_dict:
                linear2_out = state_dict['classifier.5.weight'].shape[0]
            if 'classifier.8.weight' in state_dict:
                num_classes = state_dict['classifier.8.weight'].shape[0]

            print(f"ℹ️  Reconstructing Classifier: {linear1_in} -> {linear1_out} -> {linear2_out} -> {num_classes}")

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),                # 0
                nn.Linear(linear1_in, linear1_out), # 1
                nn.BatchNorm1d(linear1_out),      # 2
                nn.ReLU(),                        # 3
                nn.Dropout(p=0.5),                # 4
                nn.Linear(linear1_out, linear2_out),# 5
                nn.BatchNorm1d(linear2_out),      # 6
                nn.ReLU(),                        # 7
                nn.Linear(linear2_out, num_classes) # 8
            )
            
            # Load weights
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            print(f"✅ Classification model loaded successfully")
            return model

        except Exception as e:
            print(f"⚠️ Error reconstructing head: {e}")
            return None
        
    except Exception as e:
        print(f"❌ Error loading classification model: {e}")
        return None

def load_skin_condition_model():
    """Load EfficientNet skin condition model"""
    model_path = MODEL_PATHS['skin_condition']
    if not os.path.exists(model_path):
        print(f"⚠️  Skin condition model not found")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
        
        for variant_name, variant_fn in [
            ('EfficientNet-B0', models.efficientnet_b0),
            ('EfficientNet-B1', models.efficientnet_b1),
            ('EfficientNet-B2', models.efficientnet_b2),
        ]:
            try:
                model = variant_fn(weights=None)
                num_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_features, len(SKIN_CONDITION_CLASSES))
                model.load_state_dict(state_dict, strict=False)
                model.to(device)
                model.eval()
                print(f"✅ Skin condition model loaded ({variant_name})")
                return model
            except:
                continue
        return None
    except Exception as e:
        print(f"❌ Error loading skin condition model: {e}")
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
        
        # --- FIX: Dùng đường dẫn tương đối (Relative Path) ---
        # Hydra sẽ tự tìm file này trong folder cài đặt của thư viện sam2
        # (File này đã được Dockerfile tải về đúng chỗ)
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
    state.skin_condition_model = load_skin_condition_model()
    state.face_detector = load_face_detection_model()

    try:
        setup_api_key()
        db, embeddings = load_or_create_vectorstore()
        
        if db is None:
            print("\n⚠️  Vector Store not initialized")
        else:
            state.vectorstore = db  # Lưu vector store
            state.rag_chain = setup_rag_chain(db)
            print("\n✅ RAG Chatbot ready")
        
        print("\n✅ Server ready!")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error initializing RAG: {e}\n")
    
    yield
    
    print("Shutting down models...")

# =============================================================================
# FASTAPI APP DEFINITION
# =============================================================================
app = FastAPI(
    title="AI Dermatology & Cosmetic Consultant API",
    description="Stateless API for skin disease classification, segmentation, and cosmetic consultation",
    version="3.4.0",
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
    skin_condition_model_status: str
    timestamp: str

class VLMAnalysisResponse(BaseModel):
    skin_analysis: str
    response_time: float
    timestamp: str

# =============================================================================
# HEALTH CHECK
# =============================================================================
@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if state.rag_chain else "degraded",
        message="AI Dermatology & Cosmetic API",
        vectorstore_status="ready" if state.rag_chain else "not_initialized",
        classification_model_status="loaded" if state.classification_model else "not_loaded",
        segmentation_model_status="loaded" if state.segmentation_model else "not_loaded",
        skin_condition_model_status="loaded" if state.skin_condition_model else "not_loaded",
        timestamp=datetime.now().isoformat()
    )

# =============================================================================
# RAG CHATBOT ENDPOINTS (UPDATED FOR MULTIMODAL + SMART PRODUCT SEARCH)
# =============================================================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    question: str = Form(...),
    conversation_history: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Stateless chat endpoint supporting Text + Optional Image (VLM) + Intelligent Product Recommendations.
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

        # 2. VLM Analysis (If image is provided)
        vlm_context_str = ""
        if image:
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Uploaded file is not an image")
            
            image_bytes = await image.read()
            skin_analysis = analyze_skin_image(image_bytes, note=question)
            
            if skin_analysis:
                vlm_context_str = f"""
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
{vlm_context_str}
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
        
    except Exception as e:
        print(f"Chat Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image_endpoint(
    image: UploadFile = File(...),
    additional_text: Optional[str] = Form(None)
):
    if state.rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    
    try:
        start_time = time.time()
        image_bytes = await image.read()
        
        skin_analysis = analyze_skin_image(image_bytes)
        if not skin_analysis:
            raise HTTPException(status_code=400, detail="Cannot analyze image")
        
        is_severe = check_severity(skin_analysis)
        rag_query = build_image_analysis_query(skin_analysis, additional_text)
        
        product_recommendation = state.rag_chain.invoke(rag_query)
        
        return ImageAnalysisResponse(
            skin_analysis=skin_analysis,
            product_recommendation=product_recommendation,
            severity_warning="⚠️ SEVERE: Please consult a dermatologist immediately!" if is_severe else None,
            response_time=round(time.time() - start_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/analyze-image-base64", response_model=ImageAnalysisResponse)
async def analyze_image_base64_endpoint(request: ImageAnalysisRequest):
    if state.rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    
    try:
        start_time = time.time()
        skin_analysis = analyze_skin_image(request.image_base64)
        if not skin_analysis:
            raise HTTPException(status_code=400, detail="Cannot analyze image")
        
        is_severe = check_severity(skin_analysis)
        rag_query = build_image_analysis_query(skin_analysis, request.additional_text)
        product_recommendation = state.rag_chain.invoke(rag_query)
        
        return ImageAnalysisResponse(
            skin_analysis=skin_analysis,
            product_recommendation=product_recommendation,
            severity_warning="⚠️ SEVERE: Please consult a dermatologist!" if is_severe else None,
            response_time=round(time.time() - start_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
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

@app.post("/api/classification-condition")
async def classify_skin_condition(file: UploadFile = File(...)) -> Dict:
    """
    Classify skin condition (Dry/Normal/Oily) - Always checks for face.
    Returns product suggestions based on detected skin condition.
    """
    if state.skin_condition_model is None:
        raise HTTPException(status_code=503, detail="Skin condition model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        if state.face_detector:
            image_np = np.array(image)
            face_results = state.face_detector.process(image_np)
            
            if not face_results.detections:
                 raise HTTPException(
                     status_code=400, 
                     detail="No face detected. Please upload a clear image of a face for skin condition analysis."
                 )
        else:
            print("⚠️ Face detector skipped (not loaded)")

        input_tensor = IMAGE_TRANSFORMS['condition'](image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = state.skin_condition_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            all_probs = probabilities[0].cpu().numpy()
        
        predicted_condition = SKIN_CONDITION_CLASSES[predicted.item()]
        
        # Get product suggestions
        product_suggestions = []
        if state.vectorstore:
            # Map condition to skin types
            suitable_skin_types = map_skin_condition_to_types(predicted_condition)
            
            # Get product suggestions
            product_suggestions = get_product_suggestions_by_skin_types(
                state.vectorstore,
                suitable_skin_types,
                num_products=5
            )
        
        return {
            "predicted_condition": predicted_condition,
            "confidence": float(confidence.item()),
            "all_predictions": {SKIN_CONDITION_CLASSES[i]: float(all_probs[i]) for i in range(len(SKIN_CONDITION_CLASSES))},
            "face_detected": True,
            "product_suggestions": product_suggestions
        }
        
    except HTTPException as he:
        raise he 
    except Exception as e:
        print(f"Error processing: {str(e)}")
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

@app.post("/api/analyze-skin-image-vlm", response_model=VLMAnalysisResponse)
async def analyze_skin_image_vlm_endpoint(file: UploadFile = File(...), note: Optional[str] = Form(None)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try: 
        start_time = time.time()
        image_bytes = await file.read()
        skin_analysis = analyze_skin_image(image_bytes, note)
        
        if not skin_analysis:
            raise HTTPException(status_code=500, detail="VLM failed to analyze the image. Please try again.")
        
        return VLMAnalysisResponse(
            skin_analysis=skin_analysis,
            response_time=round(time.time() - start_time, 2),
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Error in VLM endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# =============================================================================
# RUN SERVER
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)