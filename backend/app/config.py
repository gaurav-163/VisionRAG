"""
Centralized configuration — single source of truth for all settings.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # API Keys
    cohere_api_key: str = ""

    # Model
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    device: str = "cpu"

    # Storage
    chroma_persist_dir: str = "./chroma_db"
    upload_dir: str = "./uploads"
    max_image_size_mb: int = 10

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = str(Path(__file__).resolve().parent.parent / ".env")
        env_file_encoding = "utf-8"

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    def ensure_directories(self):
        """Create required directories if they don't exist."""
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings