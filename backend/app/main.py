"""
Application entry point — assembles middleware, routes, and startup events.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.config import get_settings
from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-load models so first request isn't slow."""
    logger.info("Starting Vision RAG service...")
    settings = get_settings()
    settings.ensure_directories()

    # Pre-load VLM model
    from app.services.vlm_service import get_vlm_service
    get_vlm_service()

    # Pre-load Cohere client
    from app.services.cohere_service import get_cohere_service
    get_cohere_service()

    logger.info("All services initialized")
    yield
    logger.info("Shutting down Vision RAG service")


app = FastAPI(
    title="Vision RAG API",
    description="Upload images, analyze with VLM, find correlations, chat with Cohere",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

# Routes
app.include_router(router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)