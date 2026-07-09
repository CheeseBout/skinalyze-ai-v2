# Skinalyze AI 

Skinalyze AI is a FastAPI-based backend service for AI-assisted skin analysis and personalized skincare product recommendation. The system combines image classification, lesion segmentation, vision-language analysis, and retrieval-augmented generation to support cosmetic consultation workflows.

This project is designed as a stateless API that can be integrated with a separate frontend or backend service.

## Features

* Skin disease classification from uploaded images
* Skin lesion segmentation with mask output
* Face detection for facial skin analysis validation
* Vision-language skin image analysis using Gemini
* Retrieval-augmented chatbot for skincare and cosmetic consultation
* Personalized product recommendation based on skin condition, skin type, age, gender, and allergies
* Docker-ready deployment

## Supported Skin Classes

The classification model supports the following classes:

* Acne
* Actinic Keratosis
* Drug Eruption
* Eczema
* Normal
* Psoriasis
* Rosacea
* Seborrheic Keratoses
* Sun or Sunlight Damage
* Tinea
* Warts

## Tech Stack

* Python
* FastAPI
* PyTorch
* TorchVision
* EfficientNet-B0
* SAM2
* MediaPipe
* Google Gemini
* LangChain
* ChromaDB
* HuggingFace Sentence Transformers
* Docker

## Project Structure

```text
.
├── main.py                 # FastAPI application and API endpoints
├── RAG_cosmetic.py          # RAG pipeline, product retrieval, and image analysis logic
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker deployment configuration
├── data/                    # Product knowledge base files
├── models/                  # Model checkpoint files
├── db_chroma_v2/            # Persisted Chroma vector database
└── chat_history/            # Optional local chat history storage
```

## Requirements

Before running the project, prepare the following:

* Python 3.10 or later
* Google API key for Gemini
* Trained classification model checkpoint
* Trained segmentation model checkpoint
* Product knowledge base data for RAG

Expected model paths:

```text
models/efficientnet_b0_complete.pt
models/medsam2_dermatology_best_aug2.pth
```

Expected product data path:

```text
data/product_chunks.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API server:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

The API documentation will be available at:

```text
http://localhost:7860/docs
```

## Docker Usage

Build the Docker image:

```bash
docker build -t skinalyze-ai-api .
```

Run the container:

```bash
docker run --env-file .env -p 7860:7860 skinalyze-ai-api
```

## API Endpoints

| Method | Endpoint                      | Description                                               |
| ------ | ----------------------------- | --------------------------------------------------------- |
| GET    | `/`                           | Health check                                              |
| GET    | `/health`                     | Health check                                              |
| POST   | `/chat`                       | RAG-based skincare consultation with optional image input |
| POST   | `/analyze-image`              | Analyze uploaded skin image and recommend products        |
| POST   | `/analyze-image-base64`       | Analyze base64 skin image and recommend products          |
| POST   | `/api/classification-disease` | Classify skin disease from image and suggest products     |
| POST   | `/api/segmentation-disease`   | Segment skin lesion and return mask output                |
| POST   | `/api/face-detection`         | Detect whether an uploaded image contains a face          |
| POST   | `/api/analyze-skin-image-vlm` | Analyze skin image using a vision-language model          |

## Example Usage

Classify a skin image:

```bash
curl -X POST "http://localhost:7860/api/classification-disease" \
  -F "file=@sample.jpg" \
  -F "age=22" \
  -F "gender=male" \
  -F "allergies=fragrance"
```

Chat with the skincare assistant:

```bash
curl -X POST "http://localhost:7860/chat" \
  -F "question=I have oily and acne-prone skin. Can you suggest suitable products?"
```

## Notes

* This project provides cosmetic and skincare support, not medical diagnosis.
* Severe or unusual skin conditions should be evaluated by a qualified dermatologist.
* Product recommendations depend on the quality and completeness of the product knowledge base.
* The API requires valid model checkpoint files and a configured Gemini API key to run correctly.

## License

This project is intended for educational and research purposes.
