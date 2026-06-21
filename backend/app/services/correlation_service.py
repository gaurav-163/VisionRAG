"""
Correlation service — finds relationships between analyzed images.
Uses VLM for pairwise comparison + simple object overlap scoring.
"""

import logging
from itertools import combinations

from app.models.schemas import ImageAnalysis, CorrelationResult, ImageMetadata
from app.utils.prompt_templates import correlation_prompt

logger = logging.getLogger(__name__)


class CorrelationService:
    """Computes correlations between image analyses."""

    def compute_pairwise_correlation(
        self,
        analysis_a: ImageAnalysis | ImageMetadata,
        analysis_b: ImageAnalysis | ImageMetadata
    ) -> CorrelationResult:
        """Compute correlation between two image analyses."""
        shared_objects = self._find_shared_objects(
            analysis_a.objects, analysis_b.objects
        )
        similarity_score = self._compute_similarity_score(
            analysis_a, analysis_b, shared_objects
        )
        relationship = self._describe_relationship(
            analysis_a, analysis_b, shared_objects
        )

        return CorrelationResult(
            image_id_a=analysis_a.image_id,
            image_id_b=analysis_b.image_id,
            shared_objects=shared_objects,
            relationship_description=relationship,
            similarity_score=similarity_score
        )

    def compute_all_correlations(
        self,
        new_analysis: ImageAnalysis,
        existing_analyses: list[ImageMetadata]
    ) -> list[CorrelationResult]:
        """Correlate a new image against all existing images."""
        correlations = []
        for existing in existing_analyses:
            if existing.image_id == new_analysis.image_id:
                continue
            corr = self.compute_pairwise_correlation(new_analysis, existing)
            if corr.similarity_score > 0.0:  # Only keep meaningful correlations
                correlations.append(corr)

        # Sort by similarity descending
        correlations.sort(key=lambda c: c.similarity_score, reverse=True)
        return correlations

    def find_relevant_correlations(
        self,
        query: str,
        all_correlations: list[CorrelationResult]
    ) -> list[CorrelationResult]:
        """Filter correlations relevant to a user query."""
        query_lower = query.lower()
        relevant = []
        for corr in all_correlations:
            # Check if query mentions any shared objects or relationship terms
            matches = any(obj.lower() in query_lower for obj in corr.shared_objects)
            matches = matches or any(
                word in corr.relationship_description.lower()
                for word in query_lower.split()
                if len(word) > 3
            )
            if matches:
                relevant.append(corr)

        return relevant if relevant else all_correlations[:5]

    @staticmethod
    def _find_shared_objects(objects_a: list[str], objects_b: list[str]) -> list[str]:
        """Find overlapping objects with fuzzy matching."""
        set_a = {o.lower().strip() for o in objects_a}
        set_b = {o.lower().strip() for o in objects_b}

        # Exact matches
        shared = list(set_a & set_b)

        # Substring matches (e.g., "red car" and "car")
        for a in set_a:
            for b in set_b:
                if a != b and (a in b or b in a) and a not in shared and b not in shared:
                    shared.append(f"{a}/{b}")

        return shared

    @staticmethod
    def _compute_similarity_score(
        analysis_a: ImageAnalysis | ImageMetadata,
        analysis_b: ImageAnalysis | ImageMetadata,
        shared_objects: list[str]
    ) -> float:
        """Compute similarity score based on object overlap and description similarity."""
        all_objects_a = set(o.lower() for o in analysis_a.objects)
        all_objects_b = set(o.lower() for o in analysis_b.objects)
        union = all_objects_a | all_objects_b

        if not union:
            return 0.0

        # Jaccard similarity for objects
        object_score = len(shared_objects) / len(union)

        # Simple word overlap for descriptions
        words_a = set(analysis_a.description.lower().split())
        words_b = set(analysis_b.description.lower().split())
        word_union = words_a | words_b
        word_score = len(words_a & words_b) / len(word_union) if word_union else 0.0

        # Weighted combination
        return round(0.6 * object_score + 0.4 * word_score, 4)

    @staticmethod
    def _describe_relationship(
        analysis_a: ImageAnalysis | ImageMetadata,
        analysis_b: ImageAnalysis | ImageMetadata,
        shared_objects: list[str]
    ) -> str:
        """Generate human-readable relationship description."""
        if not shared_objects:
            return "These images appear to depict different subjects with no obvious shared elements."

        shared_str = ", ".join(shared_objects[:5])
        return (
            f"Both images contain: {shared_str}. "
            f"Image {analysis_a.image_id} shows: {analysis_a.description[:100]}... "
            f"Image {analysis_b.image_id} shows: {analysis_b.description[:100]}..."
        )


def get_correlation_service() -> CorrelationService:
    return CorrelationService()