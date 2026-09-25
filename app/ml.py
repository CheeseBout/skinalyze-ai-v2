"""
Computer-vision models: skin disease classification (EfficientNet-B0),
lesion segmentation (SAM2), và face detection (OpenCV Haar cascade).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2

from app.config import CLASSIFICATION_MODEL_PATH, SEGMENTATION_MODEL_PATH

# =============================================================================
# CONFIGURATION
# =============================================================================
# UPDATED: 11 classes (khớp với training notebook)
SKIN_CLASSES = [
    'Acne',
    'Actinic_Keratosis',
    'Drug_Eruption',  # ← THÊM LẠI
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

IMAGE_TRANSFORMS = {
    'classification': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# =============================================================================
# MODEL LOADING
# =============================================================================
def load_classification_model():
    """
    FIXED: Load EfficientNet-B0 với architecture khớp training notebook
    Architecture: 1280 → [Dropout 0.4] → 512 → [BN + ReLU + Dropout 0.3] → 256 → [BN + ReLU] → 11
    """
    model_path = str(CLASSIFICATION_MODEL_PATH)
    if not os.path.exists(model_path):
        print(f"Classification model not found at {model_path}")
        return None

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # 1. If checkpoint is a full model object (unlikely)
        if not isinstance(checkpoint, dict):
            print("Checkpoint is a full model object.")
            model = checkpoint
            model.to(device)
            model.eval()
            return model

        # 2. Extract state_dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Get architecture config from checkpoint
        config = checkpoint.get('config', {})
        dropout1 = config.get('dropout1', 0.4)  # Default from notebook
        dropout2 = config.get('dropout2', 0.3)  # Default from notebook
        num_classes = checkpoint.get('num_classes', len(SKIN_CLASSES))

        print(f"Loading model with config:")
        print(f"   - Num classes: {num_classes}")
        print(f"   - Dropout 1: {dropout1}")
        print(f"   - Dropout 2: {dropout2}")

        # Initialize EfficientNet-B0 base
        model = models.efficientnet_b0(weights=None)

        # Reconstruct EXACT classifier from training notebook
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

        print(f"Classification model loaded successfully")
        print(f"   Architecture: 1280 → 512 → 256 → {num_classes}")

        return model

    except Exception as e:
        print(f"Error loading classification model: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_segmentation_model():
    """Load SAM2 segmentation model (Fixed Relative Path for Hydra)"""
    model_path = str(SEGMENTATION_MODEL_PATH)
    if not os.path.exists(model_path):
        print(f"Segmentation model not found")
        return None

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        config_file = "configs/sam2.1/sam2.1_hiera_t.yaml"
        print(f"Loading SAM2 with config: {config_file}")

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
        print("SAM2 segmentation model loaded")
        return predictor

    except Exception as e:
        print(f"Error loading segmentation model: {e}")
        return None

def load_face_detection_model():
    """Load OpenCV Haar Cascade face detector (bundled with opencv-python-headless)."""
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            raise RuntimeError(f"Failed to load cascade file at {cascade_path}")
        print("Face detection model loaded")
        return detector
    except Exception as e:
        print(f"Error loading face detection model: {e}")
        return None

def detect_faces(detector: cv2.CascadeClassifier, image_np: np.ndarray) -> bool:
    """Return True if at least one face is detected in the given RGB image array."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return len(faces) > 0
