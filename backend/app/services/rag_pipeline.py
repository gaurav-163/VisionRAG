"""
RAG Pipeline — orchestrates the full flow:
  Upload:  Image → VLM Analysis → Correlation → Vector Store
  Query:   User Question → Vector Search → Context Build → Cohere LLM → Answer

This is the single entry point for both upload and chat operations.
"""

import logging
from PIL import Image

from app.models.schemas import (
    ImageAnalysis, CorrelationResult, UploadResponse,
    ChatRequest, ChatResponse, ChatMessage, MessageRole
)
from app.services.vlm_service import get_vlm_service
from app.services.correlation_service import get_correlation_service
from app.services.cohere_service import get_cohere_service
from app.services.image_store import get_image_store
from app.utils.prompt_templates import build_image_context

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrator — composes VLM, Correlation, Cohere, and ImageStore services.
    No business logic duplication; each step delegates to its service.
    """

    def __init__(self):
        self.vlm = get_vlm_service()
        self.correlator = get_correlation_service()
        self.cohere = get_cohere_service()
        self.store = get_image_store()
        # Per-session chat history
        self._chat_histories: dict[str, list[dict]] = {}

    # ── Upload Pipeline ──

    def process_image(
        self,
        image: Image.Image,
        filename: str,
        session_id: str = "default"
    ) -> UploadResponse:
        """Full upload pipeline: save → analyze → correlate → store."""

        # Step 1: Save image to disk
        image_id, filepath = self.store.save_image(image, filename)
        logger.info(f"[Pipeline] Processing image {image_id}: {filename}")

        # Step 2: Analyze with VLM
        analysis = self.vlm.analyze_image(image, image_id)
        logger.info(f"[Pipeline] VLM analysis complete for {image_id}")

        # Step 3: Store analysis + embeddings
        self.store.store_analysis(
            image_id=image_id,
            filename=filename,
            filepath=filepath,
            analysis=analysis,
            image=image,
            session_id=session_id
        )

        # Step 4: Keep the analyzed image metadata available for chat.
        metadata = self.store.get_metadata(image_id)
        if metadata:
            metadata.correlations = []
            metadata.object_counts = analysis.object_counts
            metadata.people_count = analysis.people_count
            metadata.annotated_image_path = analysis.annotated_image_path

        return UploadResponse(
            image_id=image_id,
            filename=filename,
            analysis=analysis,
            correlations=[],
            message=f"Image analyzed: {len(analysis.objects)} objects found"
        )

    # ── Chat Pipeline ──

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Full chat pipeline: search → context → LLM → response."""
        session_id = request.session_id
        user_message = request.message

        logger.info(f"[Pipeline] Chat query in session {session_id}: {user_message[:80]}...")

        # Step 1: Vector search for relevant images
        search_results = self.store.search_similar(
            query=user_message,
            n_results=5,
            session_id=session_id
        )

        # Always include the analyzed images from the current session so direct
        # questions about an uploaded image are answered from the image data
        # even when semantic search misses the best match.
        session_analyses = self.store.get_all_metadata(session_id)

        analyses_by_id: dict[str, dict] = {}
        for analysis in search_results:
            analyses_by_id[analysis["image_id"]] = analysis
        for analysis in session_analyses:
            analyses_by_id[analysis.image_id] = {
                "image_id": analysis.image_id,
                "filename": analysis.filename,
                "description": analysis.description,
                "objects": analysis.objects,
                "object_counts": analysis.object_counts,
                "people_count": analysis.people_count,
                "annotated_image_path": analysis.annotated_image_path,
                "relations": analysis.relations,
                "attributes": analysis.attributes,
            }

        analyses_for_context = list(analyses_by_id.values())

        if not analyses_for_context:
            answer = (
                "I do not have any analyzed images in this session yet. "
                "Upload an image first, then ask me what is in it."
            )
            return ChatResponse(answer=answer, referenced_images=[], correlations_used=[])

        # Step 2: Build context from the uploaded image analyses only.
        context = build_image_context(analyses=analyses_for_context)

        # Step 3: Get chat history
        history = self._chat_histories.get(session_id, [])

        # Step 4: Generate response with Cohere
        answer = self.cohere.generate_response(
            user_message=user_message,
            context=context,
            chat_history=history
        )

        # Step 5: Update chat history
        self._chat_histories.setdefault(session_id, []).extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer}
        ])

        # Step 6: Build response
        referenced_images = [r["image_id"] for r in analyses_for_context]

        return ChatResponse(
            answer=answer,
            referenced_images=referenced_images,
            correlations_used=[]
        )

    def get_session_info(self, session_id: str = "default") -> dict:
        """Get session summary."""
        image_ids = self.store.get_session_image_ids(session_id)
        return {
            "session_id": session_id,
            "image_count": len(image_ids),
            "image_ids": image_ids,
            "correlation_count": len(self.store.get_all_correlations(session_id)),
            "chat_messages": len(self._chat_histories.get(session_id, []))
        }


# Singleton
_pipeline_instance = None


def get_rag_pipeline() -> RAGPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance