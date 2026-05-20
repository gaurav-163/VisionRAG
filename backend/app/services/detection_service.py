"""
Object detection service — lightweight YOLOv8n wrapper for generic image analysis.
Returns object counts and per-detection details for downstream storage and RAG.
"""

from collections import Counter
from functools import lru_cache
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

from app.config import get_settings

logger = logging.getLogger(__name__)


class DetectionService:
    """Singleton-pattern YOLOv8n detector for lightweight CPU inference."""

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
        self.settings = get_settings()
        self.model_name = "yolov8n.pt"
        self.conf_threshold = 0.25
        self.max_image_side = 960
        self.model = None
        self.annotations_dir = Path(self.settings.upload_dir) / "annotated"
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

    def _draw_and_save_annotations(self, image: Image.Image, detections: list[dict], image_id: str) -> str:
        """Draw bounding boxes on image and save to annotations folder."""
        try:
            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)

            # Color palette for different classes
            colors = [
                "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
                "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B88B", "#ABEBC6",
            ]

            class_to_color = {}
            color_idx = 0

            for detection in detections:
                box = detection["box"]
                class_name = detection["class_name"]
                confidence = detection["confidence"]

                if class_name not in class_to_color:
                    class_to_color[class_name] = colors[color_idx % len(colors)]
                    color_idx += 1

                color = class_to_color[class_name]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # Draw label with background
                label = f"{class_name} {confidence:.2f}"
                text_bbox = draw.textbbox((x1, y1), label)
                text_width = text_bbox[2] - text_bbox[0] + 4
                text_height = text_bbox[3] - text_bbox[1] + 2

                draw.rectangle(
                    [x1, max(0, y1 - text_height - 2), x1 + text_width, y1],
                    fill=color,
                )
                draw.text((x1 + 2, max(0, y1 - text_height)), label, fill="white")

            # Save annotated image
            output_filename = f"{image_id}_annotated.jpg"
            output_path = self.annotations_dir / output_filename
            annotated.save(output_path, quality=90)

            # Return relative URL path for API response
            return f"/uploads/annotated/{output_filename}"
        except Exception as exc:
            logger.warning(f"Failed to draw annotations: {exc}")
            return None

    def _load_model(self) -> None:
        if self.model is not None:
            return

        logger.info(f"Loading YOLO detector: {self.model_name}")
        self.model = YOLO(self.model_name)

    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        longest_side = max(width, height)

        if longest_side > self.max_image_side:
            scale = self.max_image_side / float(longest_side)
            resized_size = (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            )
            rgb_image = rgb_image.resize(resized_size, Image.Resampling.LANCZOS)

        return np.asarray(rgb_image)

    def detect_objects(self, image: Image.Image, image_id: str = None) -> dict:
        """Detect generic objects and return counts, detection metadata, and annotated image path."""
        self._load_model()
        image_array = self._prepare_image(image)

        try:
            results = self.model.predict(
                source=image_array,
                conf=self.conf_threshold,
                imgsz=640,
                device=self.settings.device,
                verbose=False,
            )
        except TypeError:
            results = self.model.predict(
                source=image_array,
                conf=self.conf_threshold,
                imgsz=640,
                verbose=False,
            )

        if not results:
            return {"counts": {}, "detections": []}

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return {"counts": {}, "detections": []}

        class_names = getattr(result, "names", None) or getattr(self.model, "names", {}) or {}
        counts = Counter()
        detections: list[dict] = []

        for box in boxes:
            confidence = float(box.conf.item()) if getattr(box, "conf", None) is not None else 0.0
            if confidence < self.conf_threshold:
                continue

            class_id = int(box.cls.item()) if getattr(box, "cls", None) is not None else -1
            if isinstance(class_names, dict):
                class_name = str(class_names.get(class_id, f"class_{class_id}"))
            elif 0 <= class_id < len(class_names):
                class_name = str(class_names[class_id])
            else:
                class_name = f"class_{class_id}"

            class_name = class_name.strip()
            counts[class_name] += 1

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "box": {
                        "x1": round(float(x1), 2),
                        "y1": round(float(y1), 2),
                        "x2": round(float(x2), 2),
                        "y2": round(float(y2), 2),
                    },
                }
            )

        annotated_image_path = None
        if image_id and detections:
            annotated_image_path = self._draw_and_save_annotations(image, detections, image_id)

        return {
            "counts": dict(counts),
            "detections": detections,
            "annotated_image_path": annotated_image_path,
        }


@lru_cache()
def get_detection_service() -> DetectionService:
    return DetectionService()