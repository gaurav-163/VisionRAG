"""
Cohere LLM service — handles chat completion with RAG context.
"""

import cohere
import logging
from functools import lru_cache

from app.config import get_settings
from app.utils.prompt_templates import rag_system_prompt

logger = logging.getLogger(__name__)


class CohereService:
    """Wrapper around Cohere API for RAG-based chat."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        settings = get_settings()
        self.client = cohere.ClientV2(api_key=settings.cohere_api_key)
        logger.info("Cohere client initialized")

    def generate_response(
        self,
        user_message: str,
        context: str,
        chat_history: list[dict] | None = None
    ) -> str:
        """Generate a response using Cohere with image context."""
        system_prompt = rag_system_prompt(context)

        messages = [{"role": "system", "content": system_prompt}]

        # Add chat history
        if chat_history:
            for msg in chat_history[-10:]:  # Last 10 messages for context window
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat(
                model="command-a-03-2025",
                messages=messages
            )
            return response.message.content[0].text

        except Exception as e:
            logger.error(f"Cohere API error: {e}")
            return f"I encountered an error generating a response: {str(e)}"

    def generate_embedding(self, text: str) -> list[float]:
        """Generate text embedding for vector storage."""
        try:
            response = self.client.embed(
                texts=[text],
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"]
            )
            return response.embeddings.float_[0]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero vector as fallback
            return [0.0] * 1024

    def generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        try:
            response = self.client.embed(
                texts=[query],
                model="embed-v4.0",
                input_type="search_query",
                embedding_types=["float"]
            )
            return response.embeddings.float_[0]
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return [0.0] * 1024


@lru_cache()
def get_cohere_service() -> CohereService:
    return CohereService()