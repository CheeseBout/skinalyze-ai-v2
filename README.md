---
title: Skinalyze API
emoji: 🧴
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Skinalyze API

RESTful API for an AI-powered dermatology analysis and cosmetics recommendation system, designed as the AI backend behind a NestJS service (NestJS owns users/sessions and calls this API to get analysis and recommendation results).

Three core capabilities:

- **Skin disease classification** from images using a self-hosted EfficientNet-B0 — 11 classes (acne, eczema, psoriasis, rosacea, tinea...)
- **Lesion segmentation** using SAM2 — a MedSAM2 fine-tune for dermatology
- **RAG cosmetics chatbot** — OpenAI `gpt-4o-mini` + ChromaDB, querying the product data in `data/`

The entire API is **stateless**: no session or chat history is stored server-side — the client (NestJS) manages state and passes `conversation_history` along if needed.

## Table of contents

- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running the server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Usage examples](#usage-examples)
- [Testing](#testing)
- [Updating product data](#updating-product-data)
- [Deployment](#deployment)
- [API key security](#api-key-security)
- [Production checklist](#production-checklist)
- [Troubleshooting](#troubleshooting)

## Tech stack

| Component | Technology |
|---|---|
| Runtime | Python 3.12.10 |
| Web framework | FastAPI + Uvicorn |
| LLM (chatbot) | OpenAI `gpt-4o-mini` via LangChain |
| Vector store | ChromaDB, embedding `sentence-transformers/all-MiniLM-L6-v2` |
| Skin disease classification | PyTorch — EfficientNet-B0 fine-tune, self-hosted (`models/efficientnet_b0_complete.pt`) |
| Lesion segmentation | SAM2 — MedSAM2 fine-tune, self-hosted (`models/medsam2_dermatology_best_aug2.pth`) |
| Face detection | OpenCV Haar cascade (bundled in `opencv-python-headless`, no separate model needed) |
| Deploy | Docker, running on Hugging Face Spaces |

## Project structure

```
skinalyze/
├── app/
│   ├── main.py         # FastAPI app, CORS, lifespan (load models + build RAG chain), router registration
│   ├── config.py        # Single source of configuration: paths + env vars (reads .env)
│   ├── state.py          # AppState — loaded models, RAG chain, LLM status (singleton)
│   ├── schemas.py          # Pydantic request/response models
│   ├── ml.py                 # Loads classification/segmentation/face-detection models + transforms
│   ├── rag.py                  # Vector store, RAG chain, skin condition detection, product suggestions
│   └── routes/
│       ├── health.py             # GET /, GET /health
│       ├── chat.py                # POST /chat
│       ├── analysis.py             # POST /analyze-image
│       └── disease.py               # POST /api/classification-disease, /api/segmentation-disease, /api/face-detection
├── data/
│   ├── fixed_cosmetic.csv    # Raw product data
│   └── product_chunks.txt     # Pre-formatted for the vector store (generated from the CSV)
├── models/                # Model weights — tracked with Git LFS (*.pt, *.pth)
├── scripts/update_data.py   # Regenerates product_chunks.txt from the CSV
├── tests/test_api.py          # Script for smoke-testing the endpoints
├── .env.example
├── Dockerfile
└── requirements.txt
```

## Setup

**Requires Python 3.12.10** (see [.python-version](.python-version)).

```bash
python -m venv venv
# Windows: venv\Scripts\activate | macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Open .env and fill in OPENAI_API_KEY
```

> **Windows:** `pip install` will build `chroma-hnswlib` from source (no
> prebuilt wheel for Python 3.12 on Windows yet) — install **[Microsoft C++
> Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)**
> (the "Desktop development with C++" workload) first if you don't have it.
> Linux/macOS don't need this (prebuilt wheels exist), and the Docker build
> installs `build-essential` automatically.

`.env.example` lists every environment variable:

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | Yes | — |
| `OPENAI_MODEL` | No | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` |
| `HF_TOKEN` | No | — (avoids the "unauthenticated requests" rate-limit warning when downloading `EMBEDDING_MODEL` for the first time — see [how to get a token](https://huggingface.co/settings/tokens)) |
| `LLM_TEMPERATURE` | No | `0.05` |
| `LLM_MAX_TOKENS` | No | `1500` |
| `LLM_MAX_RETRIES` | No | `3` |
| `LLM_TIMEOUT` | No | `90` |
| `USD_TO_VND` | No | `26349` |
| `CORS_ORIGINS` | No | `*` (comma-separated if restricting to specific domains) |

## Running the server

```bash
# Development (auto-reload)
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

The server runs at `http://localhost:8000`. Swagger UI: `http://localhost:8000/docs`, ReDoc: `http://localhost:8000/redoc`.

> On first run, the embedding model is downloaded and the vector store is built from `data/product_chunks.txt` into `db_chroma/` (takes a few minutes) — subsequent runs load from cache.

## API Endpoints

### Health Check

**GET** `/` or **GET** `/health`

```json
{
  "status": "healthy",
  "message": "AI Dermatology & Cosmetic API",
  "vectorstore_status": "ready",
  "classification_model_status": "loaded",
  "segmentation_model_status": "loaded",
  "llm_status": "ready",
  "llm_model": "gpt-4o-mini",
  "llm_error": null,
  "timestamp": "2026-09-24T10:30:00"
}
```

`status` is `"healthy"` only when the LLM is ready **and** both the classification and segmentation models are loaded; otherwise it's `"degraded"` — the parts that are still working continue to serve requests.

### Chat (text + optional image)

**POST** `/chat` — `multipart/form-data`

| Field | Required | Notes |
|---|---|---|
| `question` | Yes | The user's question |
| `conversation_history` | No | JSON string: `[{"role":"user","content":"..."},{"role":"ai","content":"..."}]` — only the last 3 turns are used as context |
| `image` | No | Image file — if provided, the classifier runs first to add skin-condition context before answering |

If the question mentions a supported skin condition (see the table below), the system automatically prioritizes finding products suited to the matching skin type, ignoring conversation history to avoid confusion.

**Response:**
```json
{
  "answer": "Mình gợi ý cho bạn: 1. La Roche-Posay Toleriane...",
  "response_time": 2.5,
  "timestamp": "2026-09-24T10:30:00"
}
```

### Skin image analysis → product recommendation

**POST** `/analyze-image` — `multipart/form-data`: `image` (file, required), `additional_text` (optional)

Uses the self-hosted EfficientNet-B0 classifier to determine the skin condition (no third-party vision model is called — those models refuse to analyze real human face images), then feeds the result through the RAG chain to produce a written product recommendation.

```json
{
  "skin_analysis": "Tình trạng da nhận diện được: Mụn (Acne) (độ tin cậy 87%)",
  "product_recommendation": "Dạ, mình gợi ý sản phẩm cho mụn nhé: ...",
  "severity_warning": null,
  "response_time": 3.1,
  "timestamp": "2026-09-24T10:30:00"
}
```

`severity_warning` only appears when the recognition confidence is below 50%, recommending the user consult a dermatologist.

### Skin disease classification (no RAG)

**POST** `/api/classification-disease` — `multipart/form-data`: `file` (required), `notes` (optional — pass `"facial"` to require a face-detection check before classifying, returns a 400 if no face is found)

```json
{
  "predicted_class": "Acne",
  "confidence": 0.87,
  "all_predictions": {"Acne": 0.87, "Normal": 0.05, "...": "..."},
  "product_suggestions": ["CeraVe Foaming Facial Cleanser", "..."]
}
```

The 11 classes the classifier recognizes (`predicted_class`):

| Class | Description |
|---|---|
| `Acne` | Acne |
| `Actinic_Keratosis` | Actinic keratosis |
| `Drug_Eruption` | Drug eruption |
| `Eczema` | Eczema |
| `Normal` | Normal skin |
| `Psoriasis` | Psoriasis |
| `Rosacea` | Rosacea |
| `Seborrh_Keratoses` | Seborrheic keratosis |
| `Sun_Sunlight_Damage` | Sun/sunlight damage |
| `Tinea` | Tinea (ringworm) |
| `Warts` | Warts |

### Lesion segmentation

**POST** `/api/segmentation-disease` — `multipart/form-data`: `file` (required)

```json
{
  "mask": "iVBORw0KGgo...",
  "lesion_on_black": "/9j/4AAQSkZJRg...",
  "format": "base64_png",
  "original_size": [1024, 768],
  "confidence": 0.93
}
```

`mask`: base64 PNG (grayscale mask). `lesion_on_black`: base64 JPEG — the lesion area is kept intact, the rest of the image is blacked out.

### Face detection check

**POST** `/api/face-detection` — `multipart/form-data`: `file` (required)

```json
{ "has_face": true }
```

If the face detector failed to load, the endpoint returns `has_face: true` (fail-open) so it doesn't block the client's flow.

## Usage examples

### cURL

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -F "question=Tôi cần kem dưỡng cho da khô nhạy cảm"

curl -X POST http://localhost:8000/analyze-image \
  -F "image=@my_skin.jpg" \
  -F "additional_text=Tư vấn sản phẩm cho tôi"

curl -X POST http://localhost:8000/api/classification-disease \
  -F "file=@my_skin.jpg" \
  -F "notes=facial"
```

### Python (requests)

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    data={"question": "Tôi cần serum cho da dầu mụn"}
)
print(response.json())

with open("my_skin.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze-image",
        files={"image": f},
        data={"additional_text": "Tư vấn giúp tôi"}
    )
print(response.json())
```

### JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append("question", "Tôi cần kem chống nắng cho da nhạy cảm");

fetch("http://localhost:8000/chat", { method: "POST", body: formData })
  .then((res) => res.json())
  .then((data) => console.log(data));
```

## Testing

```bash
# Server must be running at http://localhost:8000
python tests/test_api.py
```

## Updating product data

After changing `data/fixed_cosmetic.csv`, rerun this script to regenerate `data/product_chunks.txt` and clear the old vector store (it will be rebuilt from the new chunks the next time the server starts):

```bash
python scripts/update_data.py
```

## Deployment

### Preparation

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY in .env — DO NOT commit this file to Git
```

`.env` is already listed in `.gitignore`.

### Deploying with Docker (recommended)

```bash
docker build -t skinalyze .
docker run -p 7860:7860 --env-file .env skinalyze
```

The server runs at `http://localhost:7860`.

### Deploying to Hugging Face Spaces

The repo is pre-configured for Spaces (`sdk: docker`, `app_port: 7860` — see the YAML front matter block at the top of this file).

1. Create a new Space, selecting the **Docker** SDK.
2. Push this repo to the Space (via Git remote or drag-and-drop in the web UI).
3. Under **Settings → Repository secrets**, add `OPENAI_API_KEY`. Don't put the API key in a `.env` file when deploying to Spaces — use Secrets instead.
4. The Space will automatically build the Dockerfile and run `uvicorn app.main:app --host 0.0.0.0 --port 7860`.

## API key security

**NEVER** commit the `.env` file or hard-code an API key in source:

```python
# WRONG
OPENAI_API_KEY = "sk-..."  # DANGEROUS!
```

If an API key is accidentally committed to Git:

1. **Revoke it immediately** on [platform.openai.com](https://platform.openai.com/api-keys) and issue a new one.
2. Purge the key from Git history with [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) or `git filter-repo`, then `git push --force`.

## Production checklist

- [ ] **CORS:** set `CORS_ORIGINS` in `.env` to the specific domain(s) of the NestJS backend/frontend instead of `*`
- [ ] **HTTPS:** put the server behind a reverse proxy (Nginx, Caddy) or use the platform's built-in HTTPS (Spaces, Cloud Run...)
- [ ] **Rate limiting / Authentication:** the endpoints currently have no auth — if exposed publicly, add API key/JWT auth or rate limiting at the gateway/NestJS layer
- [ ] **Model weights:** make sure Git LFS has fully pulled `models/*.pt` and `models/*.pth` before building the image

## Troubleshooting

| Issue | Likely cause |
|---|---|
| `GET /health` returns `llm_status: "error"` | Missing or invalid `OPENAI_API_KEY`, or the model was renamed/deprecated — check `llm_error` in the response |
| `503 RAG chain not initialized` | The vector store failed to initialize — check that `data/product_chunks.txt` exists and review the startup logs |
| `503 Classification/Segmentation model not loaded` | The weight file in `models/` is missing or wasn't pulled via Git LFS |
| SAM2 build fails in Docker | Check that the `sam2.1_hiera_t.yaml` download step in `Dockerfile` succeeded (needs network access at build time) |
| `error: Microsoft Visual C++ 14.0 or greater is required` on `pip install` on Windows | Install Microsoft C++ Build Tools — see the [Setup](#setup) section above |
