"""
FastAPI dependencies — single place to resolve service instances.
"""

from app.services.rag_pipeline import RAGPipeline, get_rag_pipeline


def get_pipeline() -> RAGPipeline:
    return get_rag_pipeline()