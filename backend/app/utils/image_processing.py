"""
Image loading and preprocessing — shared by VLM and storage services.
"""

from PIL import Image
from pathlib import Path
import io
import base64


def load_image(source: str | bytes | Path) -> Image.Image:
    """Load image from filepath, bytes, or Path — unified loader."""
    if isinstance(source, (str, Path)):
        return Image.open(source).convert("RGB")
    elif isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")
    raise ValueError(f"Unsupported image source type: {type(source)}")


def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize keeping aspect ratio — prevents OOM on large images."""
    if max(image.size) <= max_size:
        return image
    ratio = max_size / max(image.size)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def prepare_image_for_vlm(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Resize + normalize for VLM input."""
    return resize_image(image, max_size)