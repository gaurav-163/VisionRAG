"""Backend entrypoint — assembles middleware, routes, and startup events."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .api.routes import router as rag_router
from .visionrag.api.routes import router as vision_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: defer model loading to first request."""
    logger.info("Starting Vision RAG service...")
    settings = get_settings()
    settings.ensure_directories()

    # Models will be loaded on first request to avoid long startup times
    logger.info("Service ready (models will load on first request)")
    
    yield
    logger.info("Shutting down Vision RAG service")


app = FastAPI(
    title="Vision RAG API",
    description="Upload images, analyze with VLM, find correlations, chat with Cohere, and generate scene graphs",
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

# Include routers
app.include_router(rag_router, prefix="/api")
app.include_router(vision_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)