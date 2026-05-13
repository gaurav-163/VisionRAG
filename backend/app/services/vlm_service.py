"""

Vision Language Model service — wraps SmolVLM2 for image analysis.
Handles model loading, inference, and output parsing.
"""

import torch
import logging
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from functools import lru_cache

from ..config import get_settings
from ..models.schemas import ImageAnalysis
from ..utils.image_processing import prepare_image_for_vlm
from ..utils.prompt_templates import image_analysis_prompt

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
                dtype=torch.float32,
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
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=input_text,
                images=[image],
                return_tensors="pt"
            ).to(self.settings.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )

            decoded = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated portion (after the prompt)
            if input_text in decoded:
                decoded = decoded[len(input_text):].strip()

            return decoded

        except Exception as e:
            logger.error(f"VLM inference failed: {e}")
            return f"Analysis failed: {str(e)}"

    def _parse_analysis(self, raw_output: str, image_id: str) -> ImageAnalysis:
        """Parse VLM raw text output into structured ImageAnalysis."""
        description = ""
        objects = []
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

        # Fallback if parsing didn't yield structured data
        # If the model returned template placeholders (e.g. '<detailed description>'),
        # treat them as empty so fallback will use the raw output instead.
        placeholders = ["<detailed description>", "<comma-separated list of objects>", "<key details about the scene>", "<contextual information>"]
        desc_is_placeholder = any(p in description.lower() for p in [ph.lower() for ph in placeholders])
        if not description or desc_is_placeholder:
            description = raw_output[:500]
        if not objects:
            objects = self._extract_objects_fallback(raw_output)

        return ImageAnalysis(
            image_id=image_id,
            description=description,
            objects=objects,
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