"""
Vision Language Model service — wraps SmolVLM2 for image analysis.
Handles model loading, inference, and output parsing.
"""

import torch
import logging
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from functools import lru_cache

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
        return self._parse_analysis(raw_output, image_id)

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

    def _parse_analysis(self, raw_output: str, image_id: str) -> ImageAnalysis:
        """Parse VLM raw text output into structured ImageAnalysis."""
        description = ""
        objects = []
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

        return ImageAnalysis(
            image_id=image_id,
            description=description,
            objects=objects,
            relations=relations,
            attributes=attributes,
            raw_vlm_output=raw_output
        )

    @staticmethod
    def _extract_objects_fallback(text: str) -> list[str]:
        """Simple noun extraction as fallback when VLM doesn't follow format."""
        common_objects = [
            "person", "car", "dog", "cat", "tree", "building", "table",
            "chair", "phone", "computer", "book", "food", "sky", "water",
            "road", "house", "flower", "animal", "bird", "sign", "door",
            "window", "light", "mountain", "grass", "cloud", "sun"
        ]
        text_lower = text.lower()
        return [obj for obj in common_objects if obj in text_lower]


@lru_cache()
def get_vlm_service() -> VLMService:
    return VLMService()