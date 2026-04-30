"""
Image storage service — manages image files, metadata, and vector embeddings.
Single responsibility: CRUD for images + vector search.
"""

import uuid
import json
import logging
from pathlib import Path
from datetime import datetime
from PIL import Image

from app.config import get_settings
from app.models.schemas import ImageMetadata, ImageAnalysis, CorrelationResult
from app.models.database import get_collection
from app.services.cohere_service import get_cohere_service
from app.utils.image_processing import load_image

logger = logging.getLogger(__name__)


class ImageStore:
    """Manages image persistence, metadata, and vector search."""

    def __init__(self):
        self.settings = get_settings()
        self.collection = get_collection()
        self.cohere = get_cohere_service()
        # In-memory session store (replace with Redis for production)
        self._sessions: dict[str, list[str]] = {}
        self._metadata_cache: dict[str, ImageMetadata] = {}
        self._correlations: list[CorrelationResult] = []

    def generate_image_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def save_image(self, image: Image.Image, filename: str) -> tuple[str, str]:
        """Save image to disk and return (image_id, filepath)."""
        image_id = self.generate_image_id()
        ext = Path(filename).suffix or ".jpg"
        filepath = Path(self.settings.upload_dir) / f"{image_id}{ext}"
        image.save(filepath)
        logger.info(f"Saved image {image_id} to {filepath}")
        return image_id, str(filepath)

    def store_analysis(
        self,
        image_id: str,
        filename: str,
        filepath: str,
        analysis: ImageAnalysis,
        image: Image.Image,
        session_id: str = "default"
    ) -> ImageMetadata:
        """Store image analysis in vector DB and metadata cache."""
        # Create metadata
        metadata = ImageMetadata(
            image_id=image_id,
            filename=filename,
            filepath=filepath,
            width=image.width,
            height=image.height,
            description=analysis.description,
            objects=analysis.objects,
            relations=analysis.relations,
            attributes=analysis.attributes
        )

        # Generate embedding from description + objects
        embed_text = f"{analysis.description} Objects: {', '.join(analysis.objects)}"
        embedding = self.cohere.generate_embedding(embed_text)

        # Store in ChromaDB
        self.collection.upsert(
            ids=[image_id],
            embeddings=[embedding],
            documents=[embed_text],
            metadatas=[{
                "image_id": image_id,
                "filename": filename,
                "filepath": filepath,
                "description": analysis.description,
                "objects": json.dumps(analysis.objects),
                "relations": json.dumps(analysis.relations),
                "attributes": json.dumps(analysis.attributes),
                "session_id": session_id,
                "upload_time": datetime.utcnow().isoformat()
            }]
        )

        # Update caches
        self._metadata_cache[image_id] = metadata
        self._sessions.setdefault(session_id, []).append(image_id)

        logger.info(f"Stored analysis for image {image_id} in session {session_id}")
        return metadata

    @staticmethod
    def _metadata_from_chroma(doc_id: str, meta: dict) -> ImageMetadata:
        return ImageMetadata(
            image_id=doc_id,
            filename=meta.get("filename", ""),
            filepath=meta.get("filepath", ""),
            upload_time=datetime.fromisoformat(meta["upload_time"]) if meta.get("upload_time") else datetime.utcnow(),
            description=meta.get("description", ""),
            objects=json.loads(meta.get("objects", "[]")),
            relations=json.loads(meta.get("relations", "[]")),
            attributes=json.loads(meta.get("attributes", "{}")),
        )

    def store_correlations(self, correlations: list[CorrelationResult]):
        """Add new correlations to the store."""
        self._correlations.extend(correlations)

    def get_all_metadata(self, session_id: str = "default") -> list[ImageMetadata]:
        """Get all image metadata for a session."""
        image_ids = self._sessions.get(session_id, [])
        cached = [self._metadata_cache[iid] for iid in image_ids if iid in self._metadata_cache]

        if cached:
            return cached

        try:
            results = self.collection.get(where={"session_id": session_id})
        except Exception:
            return []

        metadata_list: list[ImageMetadata] = []
        if results and results.get("ids"):
            for index, doc_id in enumerate(results["ids"]):
                raw_meta = results.get("metadatas", [])
                if not raw_meta or index >= len(raw_meta):
                    continue
                meta = raw_meta[index] or {}
                metadata = self._metadata_from_chroma(doc_id, meta)
                metadata_list.append(metadata)
                self._metadata_cache[doc_id] = metadata
                if session_id not in self._sessions:
                    self._sessions[session_id] = []
                if doc_id not in self._sessions[session_id]:
                    self._sessions[session_id].append(doc_id)

        return metadata_list

    def get_metadata(self, image_id: str) -> ImageMetadata | None:
        """Get metadata for a single image."""
        metadata = self._metadata_cache.get(image_id)
        if metadata:
            return metadata

        try:
            results = self.collection.get(ids=[image_id])
        except Exception:
            return None

        if not results or not results.get("ids"):
            return None

        doc_id = results["ids"][0]
        meta_list = results.get("metadatas", [])
        if not meta_list:
            return None

        metadata = self._metadata_from_chroma(doc_id, meta_list[0] or {})
        self._metadata_cache[image_id] = metadata
        return metadata

    def search_similar(self, query: str, n_results: int = 5, session_id: str = "default") -> list[dict]:
        """Search for images similar to a text query."""
        query_embedding = self.cohere.generate_query_embedding(query)

        # Filter by session
        where_filter = {"session_id": session_id}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),
                where=where_filter if self.collection.count() > 0 else None
            )
        except Exception:
            # Fallback without filter
            if self.collection.count() == 0:
                return []
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count())
            )

        parsed = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                parsed.append({
                    "image_id": doc_id,
                    "filename": meta.get("filename", ""),
                    "description": meta.get("description", ""),
                    "objects": json.loads(meta.get("objects", "[]")),
                    "relations": json.loads(meta.get("relations", "[]")),
                    "attributes": json.loads(meta.get("attributes", "{}")),
                    "distance": results["distances"][0][i] if results["distances"] else 0
                })

        return parsed

    def get_all_correlations(self, session_id: str = "default") -> list[CorrelationResult]:
        """Get all correlations involving images in this session."""
        session_images = set(self._sessions.get(session_id, []))
        return [
            c for c in self._correlations
            if c.image_id_a in session_images or c.image_id_b in session_images
        ]

    def get_session_image_ids(self, session_id: str = "default") -> list[str]:
        return self._sessions.get(session_id, [])


# Singleton
_image_store_instance = None


def get_image_store() -> ImageStore:
    global _image_store_instance
    if _image_store_instance is None:
        _image_store_instance = ImageStore()
    return _image_store_instance