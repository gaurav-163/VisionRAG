# VisionRAG

Company-style mono-repo layout:

- `backend/` FastAPI service
- `frontend/` web client

## Run locally

```powershell
cd G:\VisionRAG\backend
& "G:\Python 3.11\Scripts\python.exe" -m pip install -r requirements.txt
& "G:\Python 3.11\Scripts\python.exe" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
