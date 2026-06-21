"""
API routes — thin controller layer. All logic lives in services/pipeline.
"""

import logging
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pathlib import Path
import io

from app.models.schemas import UploadResponse, ChatRequest, ChatResponse, SessionInfo
from app.services.rag_pipeline import RAGPipeline
from app.api.dependencies import get_pipeline
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    session_id: str = Form(default="default"),
    pipeline: RAGPipeline = Depends(get_pipeline)
):
    """Upload an image for analysis, depth estimation, and correlation."""
    settings = get_settings()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG, PNG, etc.)")

    content = await file.read()
    if len(content) > settings.max_image_size_bytes:
        raise HTTPException(400, f"Image exceeds {settings.max_image_size_mb}MB limit")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not process image file")

    result = pipeline.process_image(
        image=image,
        filename=file.filename or "unnamed.jpg",
        session_id=session_id
    )
    return result


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_pipeline)
):
    """Ask a question about uploaded images (intent-aware)."""
    if not request.message.strip():
        raise HTTPException(400, "Message cannot be empty")
    return pipeline.chat(request)


@router.get("/depth/{image_id}")
async def get_depth(image_id: str, gray: bool = False):
    """
    Return the depth map image for a given image_id.
    ?gray=true  → grayscale depth map (for Three.js displacement)
    ?gray=false → coloured INFERNO map (default, for display)
    """
    settings = get_settings()
    suffix = "_depth_gray.png" if gray else "_depth.png"
    depth_path = Path(settings.upload_dir) / "depth" / f"{image_id}{suffix}"

    if not depth_path.exists():
        raise HTTPException(404, f"Depth map not found for image {image_id}")

    return FileResponse(str(depth_path), media_type="image/png")


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(
    session_id: str,
    pipeline: RAGPipeline = Depends(get_pipeline)
):
    """Get session information."""
    info = pipeline.get_session_info(session_id)
    return SessionInfo(**info)


@router.get("/images/{session_id}")
async def get_images(
    session_id: str,
    pipeline: RAGPipeline = Depends(get_pipeline)
):
    """Get all images and their analyses for a session."""
    metadata_list = pipeline.store.get_all_metadata(session_id)
    return {
        "images": [m.model_dump() for m in metadata_list],
        "correlations": [
            c.model_dump()
            for c in pipeline.store.get_all_correlations(session_id)
        ]
    }


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Vision RAG + Depth Estimation"}
