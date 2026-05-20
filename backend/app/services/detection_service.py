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
        # Use a stricter confidence threshold for production-ready annotations
        self.conf_threshold = 0.5
        # IoU threshold used for NMS to reduce duplicate/overlapping detections
        self.iou_threshold = 0.45
        self.max_image_side = 960
        self.model = None
        self.annotations_dir = Path(self.settings.upload_dir) / "annotated"
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
    def _save_plotted_annotations(self, result, image_id: str) -> str | None:
        """Use YOLO's built-in plotting to render clean annotations and save image.

        This relies on `result.plot()` which produces a numpy image with plotted
        boxes and labels. We convert to RGB and save as JPEG to the annotations
        directory so the frontend can serve it from `/uploads/annotated/`.
        """
        try:
            plotted = result.plot()  # numpy array (H, W, 3), typically BGR
            if plotted is None:
                return None

            # Convert BGR -> RGB if necessary by reversing channels
            try:
                # sometimes plot already returns RGB, but reversing is safe
                rgb = plotted[..., ::-1]
            except Exception:
                rgb = plotted

            annotated_img = Image.fromarray(rgb)

            output_filename = f"{image_id}_annotated.jpg"
            output_path = self.annotations_dir / output_filename
            annotated_img.save(output_path, quality=90)
            return f"/uploads/annotated/{output_filename}"
        except Exception as exc:
            logger.warning(f"Failed to save plotted annotations: {exc}")
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
                iou=self.iou_threshold,
                imgsz=640,
                device=self.settings.device,
                verbose=False,
            )
        except TypeError:
            results = self.model.predict(
                source=image_array,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
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

        # Collect detections while enforcing confidence filtering and
        # preparing metadata for downstream components. Using the model-level
        # `conf` and `iou` parameters already reduces low-confidence and
        # duplicate boxes; we still apply a final per-box confidence filter.
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

            # box.xyxy may be a tensor-like object
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
        # Save plotted annotations using YOLO's internal plotting (better
        # handling of crowded scenes, label placement and readability).
        if image_id and detections:
            try:
                annotated_image_path = self._save_plotted_annotations(result, image_id)
            except Exception:
                # Fallback: if plotting fails, don't block detection result
                annotated_image_path = None

        return {
            "counts": dict(counts),
            "detections": detections,
            "annotated_image_path": annotated_image_path,
        }


@lru_cache()
def get_detection_service() -> DetectionService:
    return DetectionService()