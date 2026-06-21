"""
ChromaDB vector store — single collection for image embeddings and metadata.
"""

import chromadb
from functools import lru_cache
from app.config import get_settings


@lru_cache()
def get_chroma_client() -> chromadb.PersistentClient:
    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection(name: str = "image_analysis"):
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"description": "Vision RAG image analysis embeddings"}
    )