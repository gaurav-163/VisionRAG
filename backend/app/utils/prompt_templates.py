"""
Prompt templates — centralized to avoid duplication across services.
"""


def image_analysis_prompt() -> str:
    return (
        "Analyze this image in detail. Provide:\n"
        "1. A comprehensive description of the scene\n"
        "2. List all identifiable objects\n"
        "3. Notable attributes (colors, positions, actions, text, emotions)\n"
        "4. Any contextual information (setting, time of day, weather, etc.)\n\n"
        "Format your response as:\n"
        "DESCRIPTION: <detailed description>\n"
        "OBJECTS: <comma-separated list of objects>\n"
        "ATTRIBUTES: <key details about the scene>\n"
        "CONTEXT: <contextual information>"
    )


def correlation_prompt(desc_a: str, desc_b: str) -> str:
    return (
        f"Compare these two image descriptions and identify correlations:\n\n"
        f"Image A: {desc_a}\n\n"
        f"Image B: {desc_b}\n\n"
        "Provide:\n"
        "1. SHARED_OBJECTS: Objects or elements present in both images\n"
        "2. RELATIONSHIP: How these images relate to each other\n"
        "3. SIMILARITY_SCORE: A score from 0.0 to 1.0 indicating overall similarity\n"
        "4. DIFFERENCES: Key differences between the images"
    )


def rag_system_prompt(context: str) -> str:
    return (
        "You are a Vision RAG assistant. You help users understand and find "
        "correlations between uploaded images. Answer questions based on the "
        "analyzed image data provided as context.\n\n"
        "Be specific, reference image IDs when relevant, and highlight "
        "correlations between images when applicable.\n\n"
        f"IMAGE CONTEXT:\n{context}"
    )


def build_image_context(analyses: list[dict], correlations: list[dict]) -> str:
    """Build context string from analyses and correlations — used by RAG pipeline."""
    parts = []

    for analysis in analyses:
        parts.append(
            f"[Image {analysis['image_id']}]\n"
            f"File: {analysis.get('filename', 'unknown')}\n"
            f"Description: {analysis['description']}\n"
            f"Objects: {', '.join(analysis.get('objects', []))}\n"
            f"Attributes: {analysis.get('attributes', {})}\n"
        )

    if correlations:
        parts.append("\n--- CORRELATIONS ---")
        for corr in correlations:
            parts.append(
                f"Images {corr['image_id_a']} <-> {corr['image_id_b']}:\n"
                f"  Shared: {', '.join(corr.get('shared_objects', []))}\n"
                f"  Relationship: {corr.get('relationship_description', 'N/A')}\n"
                f"  Similarity: {corr.get('similarity_score', 0):.2f}\n"
            )

    return "\n".join(parts)