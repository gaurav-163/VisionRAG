"""
Depth Estimation Service — wraps Depth-Anything-V2-Small for monocular depth maps.
Integrated from the 3D Reconstruction project into VisionRAG.

For each uploaded image:
  - Generates a coloured (INFERNO) depth map saved to uploads/depth/
  - Scores each YOLO-detected object as foreground / mid / background
    by sampling the depth map inside its bounding box
"""

from __future__ import annotations

import io
import logging
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from app.config import get_settings

logger = logging.getLogger(__name__)

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MAX_DIM = 518  # Resize so longest edge <= 518 px for efficient CPU inference


class DepthService:
    """Singleton depth estimation service using Depth-Anything-V2-Small."""

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
        self.processor = None
        self.model = None
        self.depth_dir = Path(self.settings.upload_dir) / "depth"
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self._load_model()

    def _load_model(self) -> None:
        logger.info("Loading Depth-Anything-V2-Small model ...")
        start = time.time()
        try:
            self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            self.model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
            self.model.to(self.settings.device)
            self.model.eval()
            logger.info(f"Depth model loaded in {time.time() - start:.1f}s")
        except Exception as exc:
            logger.error(f"Failed to load depth model: {exc}")
            raise

    # ── Core inference ──────────────────────────────────────────────────────

    def _predict_depth(self, img: Image.Image) -> np.ndarray:
        """Run inference and return a float32 depth map normalised to [0, 1]."""
        orig_h, orig_w = img.height, img.width

        # Resize for efficient inference
        w, h = img.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        inputs = self.processor(images=img, return_tensors="pt").to(self.settings.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        depth = outputs.predicted_depth  # (1, H_model, W_model) or (H, W)

        # Ensure 4D for interpolation
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(1)

        # Upsample back to original resolution
        depth = F.interpolate(
            depth,
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Normalise to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        depth = (depth - d_min) / (d_max - d_min + 1e-8)
        return depth.cpu().numpy().astype(np.float32)

    # ── Public API ───────────────────────────────────────────────────────────

    def estimate_depth(self, image: Image.Image, image_id: str) -> dict:
        """
        Generate depth map for an image and save it to disk.

        Returns:
            {
                "depth_map_path": "/uploads/depth/<image_id>_depth.png",
                "depth_map_gray_path": "/uploads/depth/<image_id>_depth_gray.png",
            }
        """
        try:
            rgb = image.convert("RGB")
            depth_norm = self._predict_depth(rgb)

            depth_u8 = (depth_norm * 255).astype(np.uint8)

            # Coloured INFERNO map (nice for demo)
            colored_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
            colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
            color_img = Image.fromarray(colored_rgb)

            # Grayscale map (used for 3D displacement)
            gray_img = Image.fromarray(depth_u8, mode="L")

            # Save both
            color_path = self.depth_dir / f"{image_id}_depth.png"
            gray_path = self.depth_dir / f"{image_id}_depth_gray.png"
            color_img.save(color_path)
            gray_img.save(gray_path)

            logger.info(f"Depth maps saved for {image_id}")

            return {
                "depth_map_path": f"/uploads/depth/{image_id}_depth.png",
                "depth_map_gray_path": f"/uploads/depth/{image_id}_depth_gray.png",
                "depth_array": depth_norm,  # kept in memory for object scoring
            }

        except Exception as exc:
            logger.error(f"Depth estimation failed for {image_id}: {exc}")
            return {
                "depth_map_path": None,
                "depth_map_gray_path": None,
                "depth_array": None,
            }

    def score_objects(
        self,
        depth_array: np.ndarray,
        detections: list[dict],
        image_size: tuple[int, int],
    ) -> list[dict]:
        """
        For each YOLO detection, compute the mean depth inside its bounding box
        and assign a depth label: foreground / mid-ground / background.

        Args:
            depth_array: float32 H×W depth map normalised to [0, 1]
            detections:  list of dicts with keys class_name, box {x1,y1,x2,y2}
            image_size:  (width, height) of the ORIGINAL image

        Returns:
            List of dicts: {class_name, depth_score, depth_label}
            sorted nearest → farthest
        """
        if depth_array is None or not detections:
            return []

        orig_w, orig_h = image_size
        depth_h, depth_w = depth_array.shape

        # Scale factors in case depth map was resized
        sx = depth_w / orig_w
        sy = depth_h / orig_h

        scored = []
        for det in detections:
            box = det.get("box", {})
            x1 = int(box.get("x1", 0) * sx)
            y1 = int(box.get("y1", 0) * sy)
            x2 = int(box.get("x2", orig_w) * sx)
            y2 = int(box.get("y2", orig_h) * sy)

            # Clamp to valid range
            x1, x2 = max(0, x1), min(depth_w, x2)
            y1, y2 = max(0, y1), min(depth_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            region = depth_array[y1:y2, x1:x2]
            mean_depth = float(region.mean())

            # In Depth-Anything, LOWER value = closer (inverted disparity)
            if mean_depth < 0.35:
                label = "foreground"
            elif mean_depth < 0.65:
                label = "mid-ground"
            else:
                label = "background"

            scored.append({
                "class_name": det.get("class_name", "object"),
                "confidence": det.get("confidence", 0.0),
                "depth_score": round(mean_depth, 3),
                "depth_label": label,
            })

        # Sort nearest first
        scored.sort(key=lambda x: x["depth_score"])
        return scored


@lru_cache()
def get_depth_service() -> DepthService:
    return DepthService()
