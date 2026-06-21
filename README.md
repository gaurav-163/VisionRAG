# Vision RAG

Vision RAG is a multimodal image Q&A app built with a FastAPI backend and a React/Vite frontend.
Upload an image, let SmolVLM2 analyze it, store the result in ChromaDB, and ask questions about the uploaded image through a Cohere-powered chat interface.

## Features

- Image upload and analysis with SmolVLM2
- Structured extraction of image description, objects, relations, and attributes
- Persistent vector storage with ChromaDB
- Cohere-based chat over analyzed image context
- React frontend with chat, upload, and gallery views
- Docker Compose setup for local development

## Tech Stack

- Backend: FastAPI, Uvicorn, Pydantic, ChromaDB, Hugging Face Transformers, PyTorch, Cohere
- Frontend: React, Vite, Axios, Tailwind CSS, Lucide React
- Storage: Local uploads directory and persistent ChromaDB volume

## Project Layout

```text
VisionRAG/
  backend/
    app/
      api/
      models/
      services/
      utils/
    uploads/
    chroma_db/
  frontend/
    src/
      components/
      hooks/
      services/
      styles/
  docker-compose.yml
```

## Prerequisites

- Python 3.11+ recommended
- Node.js 18+ recommended
- A Cohere API key
- Optional: Docker and Docker Compose

## Environment Setup

Create `backend/.env` with at least:

```env
COHERE_API_KEY=your_cohere_api_key_here
```

Optional settings supported by the backend:

```env
VLM_MODEL_NAME=HuggingFaceTB/SmolVLM2-256M-Video-Instruct
DEVICE=cpu
CHROMA_PERSIST_DIR=./chroma_db
UPLOAD_DIR=./uploads
MAX_IMAGE_SIZE_MB=10
HOST=0.0.0.0
PORT=8000
```

## Run With Docker

```bash
docker compose up --build
```

After startup:

- Frontend: http://localhost:5173
- Backend API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

## Run Locally

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

- `POST /api/upload` - upload an image for analysis
- `POST /api/chat` - ask questions about the uploaded image(s)
- `GET /api/session/{session_id}` - inspect session state
- `GET /api/images/{session_id}` - list analyzed images and stored metadata
- `GET /api/health` - health check

## How It Works

1. The frontend uploads an image to the backend.
2. The backend saves the file and runs SmolVLM2 to produce structured analysis.
3. The analysis is embedded and stored in ChromaDB.
4. When you ask a question, the backend builds context from the analyzed image metadata.
5. Cohere generates the final answer from that context.

## Notes

- The current chat flow is focused on the uploaded image context rather than comparing different images.
- Uploaded files are stored under `backend/uploads/`.
- ChromaDB data is stored under `backend/chroma_db/`.

## Troubleshooting

- If the backend reports a missing API key, verify `backend/.env` contains `COHERE_API_KEY`.
- If image analysis is slow on first run, the VLM model is being downloaded and cached.
- If you change backend settings, restart the server so the new environment variables are loaded.

## License

No license file is included in this repository yet.
