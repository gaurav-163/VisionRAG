# Vision RAG — Image Analysis & Correlation Chat

Upload images → SmolVLM2 analyzes objects & scenes → Correlations computed →
Ask questions via Cohere-powered RAG chat.

## Architecture

## Quick Start

### 1. Environment Setup

```bash
cp backend/.env.example backend/.env
# Add your COHERE_API_KEY to backend/.env

## Docker (Recommended)
```
docker-compose up --build
# Frontend: http://localhost:5173
# Backend:  http://localhost:8000/docs
```
## Local Development
```
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
## Frontend
```
cd frontend
npm install
npm run dev
```
