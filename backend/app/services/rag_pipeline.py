"""
RAG Pipeline — orchestrates the full flow:
  Upload:  Image → VLM Analysis → Depth Estimation → Vector Store
  Query:   User Question → Intent Detection → Vector Search → Context Build → Cohere LLM → Answer
"""

import logging
from PIL import Image

from app.models.schemas import (
    ImageAnalysis, ObjectDepthInfo, CorrelationResult, UploadResponse,
    ChatRequest, ChatResponse, ChatMessage, MessageRole
)
from app.services.vlm_service import get_vlm_service
from app.services.correlation_service import get_correlation_service
from app.services.cohere_service import get_cohere_service
from app.services.image_store import get_image_store
from app.services.depth_service import get_depth_service
from app.utils.prompt_templates import build_image_context

logger = logging.getLogger(__name__)

# ── Intent keywords ──────────────────────────────────────────────────────────
INTENT_DEPTH = {"closest", "nearest", "foreground", "background", "farthest",
                "far", "near", "depth", "front", "behind", "distance"}
INTENT_COUNT = {"how many", "count", "number of", "total"}
INTENT_LOCATE = {"where", "location", "position", "left", "right", "top", "bottom", "center"}
INTENT_DESCRIBE = {"describe", "what is", "what are", "tell me about", "explain"}


def detect_intent(message: str) -> str:
    """Classify the user's query intent."""
    lower = message.lower()
    if any(k in lower for k in INTENT_DEPTH):
        return "depth"
    if any(k in lower for k in INTENT_COUNT):
        return "count"
    if any(k in lower for k in INTENT_LOCATE):
        return "locate"
    return "describe"


class RAGPipeline:
    """
    Orchestrator — composes VLM, Depth, Cohere, and ImageStore services.
    """

    def __init__(self):
        self.vlm = get_vlm_service()
        self.correlator = get_correlation_service()
        self.cohere = get_cohere_service()
        self.store = get_image_store()
        self.depth = get_depth_service()
        self._chat_histories: dict[str, list[dict]] = {}

    # ── Upload Pipeline ──────────────────────────────────────────────────────

    def process_image(
        self,
        image: Image.Image,
        filename: str,
        session_id: str = "default"
    ) -> UploadResponse:
        """Full upload pipeline: save → analyze → depth → store."""

        # Step 1: Save image to disk
        image_id, filepath = self.store.save_image(image, filename)
        logger.info(f"[Pipeline] Processing image {image_id}: {filename}")

        # Step 2: Analyze with VLM + YOLO
        analysis = self.vlm.analyze_image(image, image_id)
        logger.info(f"[Pipeline] VLM analysis complete for {image_id}")

        # Step 3: Depth estimation
        depth_result = self.depth.estimate_depth(image, image_id)
        logger.info(f"[Pipeline] Depth estimation complete for {image_id}")

        # Step 4: Score objects by depth (combine YOLO boxes + depth map)
        object_depths: list[ObjectDepthInfo] = []
        depth_array = depth_result.get("depth_array")
        if depth_array is not None:
            # Get raw detections from detection service for bounding boxes
            try:
                raw_detections = self.vlm.detector.detect_objects(image, image_id)
                detections = raw_detections.get("detections", [])
                scored = self.depth.score_objects(
                    depth_array=depth_array,
                    detections=detections,
                    image_size=image.size,
                )
                object_depths = [ObjectDepthInfo(**s) for s in scored]
            except Exception as exc:
                logger.warning(f"Object depth scoring failed: {exc}")

        # Step 5: Enrich analysis with depth data
        analysis.depth_map_path = depth_result.get("depth_map_path")
        analysis.depth_map_gray_path = depth_result.get("depth_map_gray_path")
        analysis.object_depths = object_depths

        # Step 6: Store analysis + embeddings
        self.store.store_analysis(
            image_id=image_id,
            filename=filename,
            filepath=filepath,
            analysis=analysis,
            image=image,
            session_id=session_id
        )

        metadata = self.store.get_metadata(image_id)
        if metadata:
            metadata.correlations = []
            metadata.object_counts = analysis.object_counts
            metadata.people_count = analysis.people_count
            metadata.annotated_image_path = analysis.annotated_image_path
            metadata.depth_map_path = analysis.depth_map_path
            metadata.depth_map_gray_path = analysis.depth_map_gray_path
            metadata.object_depths = object_depths

        return UploadResponse(
            image_id=image_id,
            filename=filename,
            analysis=analysis,
            correlations=[],
            message=f"Image analyzed: {len(analysis.objects)} objects found, depth map generated"
        )

    # ── Chat Pipeline ────────────────────────────────────────────────────────

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Full chat pipeline with intent-aware prompting."""
        session_id = request.session_id
        user_message = request.message

        logger.info(f"[Pipeline] Chat query in session {session_id}: {user_message[:80]}")

        # Detect intent
        intent = detect_intent(user_message)
        logger.info(f"[Pipeline] Detected intent: {intent}")

        # Vector search
        search_results = self.store.search_similar(
            query=user_message,
            n_results=5,
            session_id=session_id
        )

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
                "depth_map_path": analysis.depth_map_path,
                "object_depths": [d.model_dump() for d in analysis.object_depths],
                "relations": analysis.relations,
                "attributes": analysis.attributes,
            }

        analyses_for_context = list(analyses_by_id.values())

        if not analyses_for_context:
            return ChatResponse(
                answer="I don't have any analyzed images in this session yet. Upload an image first.",
                referenced_images=[],
                correlations_used=[]
            )

        # Build intent-specific context
        context = self._build_intent_context(analyses_for_context, intent)

        # Chat history
        history = self._chat_histories.get(session_id, [])

        # Generate response
        answer = self.cohere.generate_response(
            user_message=user_message,
            context=context,
            chat_history=history
        )

        # Update history
        self._chat_histories.setdefault(session_id, []).extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer}
        ])

        referenced_images = [r["image_id"] for r in analyses_for_context]
        return ChatResponse(
            answer=answer,
            referenced_images=referenced_images,
            correlations_used=[]
        )

    def _build_intent_context(self, analyses: list[dict], intent: str) -> str:
        """Build context string tailored to the detected intent."""
        parts = []

        for a in analyses:
            image_id = a.get("image_id", "")
            filename = a.get("filename", "")
            parts.append(f"=== Image: {filename} (ID: {image_id}) ===")
            parts.append(f"Description: {a.get('description', '')}")

            if intent == "depth":
                # Focus on depth information
                object_depths = a.get("object_depths", [])
                if object_depths:
                    parts.append("Spatial Depth (nearest to farthest):")
                    for od in object_depths:
                        name = od.get("class_name") if isinstance(od, dict) else od.class_name
                        label = od.get("depth_label") if isinstance(od, dict) else od.depth_label
                        score = od.get("depth_score") if isinstance(od, dict) else od.depth_score
                        parts.append(f"  - {name}: {label} (depth score: {score:.2f})")
                else:
                    parts.append("Depth data not available for this image.")

            elif intent == "count":
                # Focus on object counts
                counts = a.get("object_counts", {})
                if counts:
                    parts.append("Object counts:")
                    for obj, cnt in counts.items():
                        parts.append(f"  - {obj}: {cnt}")
                people = a.get("people_count")
                if people is not None:
                    parts.append(f"  - people: {people}")

            elif intent == "locate":
                # Focus on relations and positions
                relations = a.get("relations", [])
                if relations:
                    parts.append("Spatial relations:")
                    for r in relations:
                        parts.append(f"  - {r}")
                attrs = a.get("attributes", {})
                if attrs.get("people_positions"):
                    parts.append(f"People positions: {attrs['people_positions']}")

            else:
                # General describe — include everything
                objects = a.get("objects", [])
                if objects:
                    parts.append(f"Objects: {', '.join(objects)}")
                relations = a.get("relations", [])
                if relations:
                    parts.append(f"Relations: {'; '.join(relations)}")

        return "\n".join(parts)

    def get_session_info(self, session_id: str = "default") -> dict:
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
