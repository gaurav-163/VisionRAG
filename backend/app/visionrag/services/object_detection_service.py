"""
Object Detection Service using YOLOv8
Handles model loading, inference, and bounding box extraction.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ObjectDetectionService:
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.35, iou_threshold: float = 0.45, device: str = "cpu"):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info(f"Loading YOLOv8 model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def detect(self, image: np.ndarray) -> list[dict]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided for detection.")
        results = self.model(image, conf=self.confidence_threshold, iou=self.iou_threshold, device=self.device, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                label = result.names[class_id]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                area = int((x2 - x1) * (y2 - y1))
                detections.append({"label": label, "confidence": round(confidence, 4), "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}, "center": {"x": center_x, "y": center_y}, "area": area})
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image bytes. Ensure the file is a valid image (JPEG, PNG, BMP, etc.)")
        return image

    def get_image_dimensions(self, image: np.ndarray) -> dict:
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        return {"height": height, "width": width, "channels": channels}
