"""
Vision Language Model service — wraps SmolVLM2 for image analysis.
Handles model loading, inference, and output parsing.
"""

import torch
import logging
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from functools import lru_cache
import cv2
import numpy as np

from app.config import get_settings
from app.models.schemas import ImageAnalysis
from app.utils.image_processing import prepare_image_for_vlm
from app.utils.prompt_templates import image_analysis_prompt

logger = logging.getLogger(__name__)


class VLMService:
    """Singleton-pattern VLM service for image analysis."""

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
        self.person_detector = cv2.HOGDescriptor()
        self.person_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._load_model()

    def _load_model(self):
        """Load VLM model and processor."""
        logger.info(f"Loading VLM model: {self.settings.vlm_model_name}")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.settings.vlm_model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.settings.vlm_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(self.settings.device)
            self.model.eval()
            logger.info("VLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}")
            raise

    def analyze_image(self, image: Image.Image, image_id: str) -> ImageAnalysis:
        """Run VLM inference on a single image and parse the output."""
        processed_image = prepare_image_for_vlm(image)
        raw_output = self._run_inference(processed_image, image_analysis_prompt())
        return self._parse_analysis(raw_output, image, image_id)

    def _run_inference(self, image: Image.Image, prompt: str) -> str:
        """Core inference method — reused for any VLM query."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )

            inputs = inputs.to(self.settings.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )

            prompt_length = inputs["input_ids"].shape[-1]
            generated_tokens = outputs[0][prompt_length:]
            decoded = self.processor.decode(generated_tokens, skip_special_tokens=True).strip()

            return decoded

        except Exception as e:
            logger.error(f"VLM inference failed: {e}")
            return f"Analysis failed: {str(e)}"

    def _parse_analysis(self, raw_output: str, image: Image.Image, image_id: str) -> ImageAnalysis:
        """Parse VLM raw text output into structured ImageAnalysis."""
        description = ""
        objects = []
        people_count = None
        relations = []
        attributes = {}

        lines = raw_output.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if line.upper().startswith("DESCRIPTION:"):
                current_section = "description"
                description = line.split(":", 1)[1].strip()
            elif line.upper().startswith("OBJECTS:"):
                current_section = "objects"
                obj_text = line.split(":", 1)[1].strip()
                objects = [o.strip() for o in obj_text.split(",") if o.strip()]
            elif line.upper().startswith("PEOPLE_COUNT:") or line.upper().startswith("PERSON_COUNT:"):
                current_section = "people_count"
                count_text = line.split(":", 1)[1].strip()
                try:
                    people_count = int("".join(ch for ch in count_text if ch.isdigit()))
                except ValueError:
                    people_count = None
            elif line.upper().startswith("RELATIONS:"):
                current_section = "relations"
                rel_text = line.split(":", 1)[1].strip()
                relations = [r.strip() for r in rel_text.split(",") if r.strip()]
            elif line.upper().startswith("ATTRIBUTES:"):
                current_section = "attributes"
                attr_text = line.split(":", 1)[1].strip()
                attributes["details"] = attr_text
            elif line.upper().startswith("CONTEXT:"):
                current_section = "context"
                attributes["context"] = line.split(":", 1)[1].strip()
            elif current_section == "description" and line:
                description += " " + line
            elif current_section == "objects" and line:
                objects.extend([o.strip() for o in line.split(",") if o.strip()])
            elif current_section == "relations" and line:
                relations.extend([r.strip() for r in line.split(",") if r.strip()])

        # Fallback if parsing didn't yield structured data
        if not description:
            description = raw_output[:500]
        if not objects:
            objects = self._extract_objects_fallback(raw_output)

        detected_people = self._count_people_with_hog(image)
        if detected_people is not None:
            people_count = detected_people
        elif people_count is None:
            people_count = self._infer_people_count(objects, description, raw_output)

        # Filter objects for consistency with description
        objects = self._filter_inconsistent_objects(objects, description)

        return ImageAnalysis(
            image_id=image_id,
            description=description,
            objects=objects,
            people_count=people_count,
            relations=relations,
            attributes=attributes,
            raw_vlm_output=raw_output
        )

    @staticmethod
    def _extract_objects_fallback(text: str) -> list[str]:
        """Simple noun extraction as fallback when VLM doesn't follow format.
        Prioritizes common structural/visible objects, excludes ambiguous animals.
        """
        common_objects = [
            "person", "people", "worker", "car", "truck", "vehicle", "tree", "building",
            "table", "chair", "phone", "computer", "book", "food", "sky", "water",
            "road", "house", "flower", "sign", "door", "window", "light", "mountain",
            "grass", "cloud", "sun", "equipment", "machine", "tool", "tank", "structure",
            "wall", "floor", "ceiling", "text", "screen", "monitor", "desk", "vehicle"
        ]
        text_lower = text.lower()
        return [obj for obj in common_objects if obj in text_lower]

    @staticmethod
    def _filter_inconsistent_objects(objects: list[str], description: str) -> list[str]:
        """Remove objects that are inconsistent with the description.
        Objects must either appear in the description or make contextual sense.
        """
        if not description:
            return objects

        description_lower = description.lower()
        filtered = []

        # Objects that are often hallucinated and shouldn't appear without being in description
        high_hallucination_risk = {
            "cat", "dog", "bird", "animal", "pet", "fish", "rabbit", "squirrel",
            "horse", "cow", "sheep", "pig", "lion", "tiger", "bear", "monkey"
        }

        for obj in objects:
            obj_lower = obj.lower()
            # If it's a risky object, only keep if mentioned in description
            if any(risky in obj_lower for risky in high_hallucination_risk):
                if obj_lower in description_lower or any(risky in description_lower for risky in high_hallucination_risk):
                    filtered.append(obj)
                # Silently drop hallucinated animals not in description
            else:
                # Keep non-animal objects
                filtered.append(obj)

        return filtered

    @staticmethod
    def _infer_people_count(objects: list[str], description: str, raw_output: str) -> int | None:
        """Best-effort fallback for people count when the model omits it."""
        combined_text = f"{description}\n{raw_output}".lower()

        if "no people" in combined_text or "no person" in combined_text:
            return 0

        import re

        count_patterns = [
            r"(?:people|persons|person|workers|crowd)\s*(?:of|:)?\s*(\d+)",
            r"(\d+)\s*(?:people|persons|person|workers)",
        ]
        for pattern in count_patterns:
            match = re.search(pattern, combined_text)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue

        return None

    def _count_people_with_hog(self, image: Image.Image) -> int | None:
        """Count visible people using OpenCV's built-in HOG detector.

        This avoids downloading model weights and works offline.
        Returns None when detection cannot be performed.
        """
        try:
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size

            # Resize large frames to improve speed and consistency.
            max_width = 900
            if width > max_width:
                scale = max_width / float(width)
                rgb_image = rgb_image.resize((max_width, int(height * scale)))

            image_array = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            boxes, weights = self.person_detector.detectMultiScale(
                gray,
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.05,
                useMeanshiftGrouping=False,
            )

            if len(boxes) == 0:
                return 0

            if weights is None or len(weights) == 0:
                return int(len(boxes))

            valid_boxes = 0
            for weight in weights.flatten():
                if float(weight) >= 0.3:
                    valid_boxes += 1

            return valid_boxes if valid_boxes > 0 else int(len(boxes))
        except Exception as exc:
            logger.warning(f"HOG person detector unavailable, falling back to VLM count: {exc}")
            return None


@lru_cache()
def get_vlm_service() -> VLMService:
    return VLMService()