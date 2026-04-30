"""
Prompt templates — centralized to avoid duplication across services.
"""


def image_analysis_prompt() -> str:
    return (
        "Analyze this image in detail. Provide:\n"
        "1. A comprehensive description of the scene\n"
        "2. List all identifiable objects\n"
        "3. Relationships between important objects or regions\n"
        "4. Notable attributes (colors, positions, actions, text, emotions)\n"
        "5. Any contextual information (setting, time of day, weather, etc.)\n\n"
        "Format your response as:\n"
        "DESCRIPTION: <detailed description>\n"
        "OBJECTS: <comma-separated list of objects>\n"
        "RELATIONS: <comma-separated list of object relationships>\n"
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
        "You are a multimodal Vision RAG assistant. The uploaded images have "
        "already been analyzed and the results are provided below as IMAGE "
        "CONTEXT. Use that context as the source of truth when answering. "
        "Do not say that you cannot access the images or that you only see "
        "placeholders.\n\n"
        "Be specific and reference the current image ID when relevant. "
        "Focus on the content of the uploaded image rather than comparing it "
        "to other images. If the context does not contain enough information, "
        "say exactly what is missing.\n\n"
        f"IMAGE CONTEXT:\n{context}"
    )


def build_image_context(analyses: list[dict]) -> str:
    """Build context string from analyzed images for single-image Q&A."""
    parts = []

    for analysis in analyses:
        parts.append(
            f"[Image {analysis['image_id']}]\n"
            f"File: {analysis.get('filename', 'unknown')}\n"
            f"Description: {analysis['description']}\n"
            f"Objects: {', '.join(analysis.get('objects', []))}\n"
            f"Relations: {', '.join(analysis.get('relations', []))}\n"
            f"Attributes: {analysis.get('attributes', {})}\n"
        )

    return "\n".join(parts)