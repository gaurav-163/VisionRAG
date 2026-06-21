"""
Vision Language Model service — wraps SmolVLM2 for image analysis.
Detector-first pipeline: YOLO provides authoritative people counts and primary object signals,
while the VLM is used for scene description, relationships, and human activity reasoning.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import logging
import re

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from app.config import get_settings
from app.models.schemas import ImageAnalysis
from app.services.detection_service import get_detection_service
from app.utils.image_processing import prepare_image_for_vlm
from app.utils.object_normalization import HUMAN_LABELS, normalize_object_counts
from app.utils.prompt_templates import image_analysis_prompt

logger = logging.getLogger(__name__)

PERSON_REGEX = re.compile(r"\b(person|people|man|woman|boy|girl|human|worker|staff)\b", re.IGNORECASE)
COUNT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

COUNT_PATTERNS = (
    re.compile(r"\b(?:count|total|people|person|humans?|workers?|staff)\D{0,12}(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\D{0,12}(?:people|persons?|humans?|workers?|staff)\b", re.IGNORECASE),
    re.compile(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\D{0,12}(?:people|persons?|humans?|workers?|staff)\b",
        re.IGNORECASE,
    ),
)

SECTION_HEADERS = {
    "description": re.compile(r"^DESCRIPTION:\s*(.*)$", re.IGNORECASE),
    "visible_people": re.compile(r"^VISIBLE_PEOPLE:\s*(.*)$", re.IGNORECASE),
    "objects": re.compile(r"^OBJECTS:\s*(.*)$", re.IGNORECASE),
    "relations": re.compile(r"^RELATIONS:\s*(.*)$", re.IGNORECASE),
    "attributes": re.compile(r"^ATTRIBUTES:\s*(.*)$", re.IGNORECASE),
}

FALLBACK_OBJECT_LABELS = (
    "person",
    "chair",
    "table",
    "desk",
    "bench",
    "sofa",
    "couch",
    "bed",
    "car",
    "truck",
    "bus",
    "van",
    "motorcycle",
    "bicycle",
    "train",
    "plane",
    "boat",
    "door",
    "window",
    "wall",
    "floor",
    "ceiling",
    "road",
    "street",
    "sidewalk",
    "building",
    "house",
    "office",
    "store",
    "shop",
    "tree",
    "plant",
    "grass",
    "sign",
    "screen",
    "monitor",
    "phone",
    "laptop",
    "computer",
    "keyboard",
    "camera",
    "book",
    "bag",
    "backpack",
    "bottle",
    "cup",
    "glass",
    "plate",
    "bowl",
    "box",
    "package",
    "helmet",
    "hat",
    "shirt",
    "jacket",
    "shoe",
    "vehicle",
)

OBJECT_EXTRACTION_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(label) for label in sorted(set(FALLBACK_OBJECT_LABELS), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


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
        self.detector = get_detection_service()
        self._last_detection_results: dict[str, dict] = {}
        self._load_model()

    def _load_model(self) -> None:
        """Load VLM model and processor with device-aware dtype selection."""
        logger.info("Loading VLM model: %s", self.settings.vlm_model_name)
        dtype = torch.float16 if "cuda" in str(self.settings.device).lower() else torch.float32

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.settings.vlm_model_name,
                trust_remote_code=True,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.settings.vlm_model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(self.settings.device)
            self.model.eval()
            logger.info("VLM model loaded successfully")
        except Exception as exc:
            logger.error("Failed to load VLM model: %s", exc)
            raise

    def analyze_image(self, image: Image.Image, image_id: str) -> ImageAnalysis:
        """Run detector-first analysis and parse the VLM output into structured data."""
        detection_result = self._safe_detect(image, image_id)
        processed_image = prepare_image_for_vlm(image)
        raw_output = self._run_inference(processed_image, image_analysis_prompt())
        analysis = self._parse_analysis(raw_output, image_id, detection_result)
        # Cache raw detector output (with correct inference-space boxes) so
        # downstream services (e.g. depth scoring) don't need to re-run YOLO.
        self._last_detection_results[image_id] = detection_result
        return analysis

    def get_detection_result(self, image_id: str) -> dict | None:
        """Retrieve the raw YOLO detection result cached for an image_id."""
        return self._last_detection_results.get(image_id)

    def _safe_detect(self, image: Image.Image, image_id: str) -> dict:
        try:
            return self.detector.detect_objects(image, image_id)
        except Exception as exc:
            logger.warning("YOLO detection failed, falling back to VLM parsing: %s", exc)
            return {"counts": {}, "detections": [], "annotated_image_path": None}

    def _run_inference(self, image: Image.Image, prompt: str) -> str:
        """Core inference method used for image understanding queries."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.settings.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )

            prompt_length = inputs["input_ids"].shape[-1]
            generated_tokens = outputs[0][prompt_length:]
            return self.processor.decode(generated_tokens, skip_special_tokens=True).strip()
        except Exception as exc:
            logger.error("VLM inference failed: %s", exc)
            return f"Analysis failed: {exc}"

    def _parse_analysis(self, raw_output: str, image_id: str, detection_result: dict) -> ImageAnalysis:
        """Parse VLM output and merge it with detector output."""
        sections = self._parse_sections(raw_output)
        description = self._join_lines(sections.get("description", []))
        visible_people_text = self._join_lines(sections.get("visible_people", []))
        objects = self._parse_objects_section(sections.get("objects", []))
        relations = self._parse_list_section(sections.get("relations", []))
        attributes = self._parse_attributes_section(sections.get("attributes", []))

        detector_counts = normalize_object_counts(detection_result.get("counts", {}))
        detector_objects = list(detector_counts.keys())
        detected_people_count = detector_counts.get("person")

        people_info = self._extract_people_info(visible_people_text, raw_output)
        people_count, people_source = self._normalize_people(detected_people_count, people_info, raw_output)

        vlm_objects = [self._normalize_object_label(obj) for obj in objects]
        merged_objects = self._merge_objects(detector_objects, vlm_objects)

        object_counts = dict(detector_counts)
        vlm_counts = normalize_object_counts(dict(Counter(vlm_objects)))
        if not object_counts and vlm_counts:
            object_counts = vlm_counts
        else:
            for label, count in vlm_counts.items():
                object_counts.setdefault(label, count)

        if not merged_objects and object_counts:
            merged_objects = list(object_counts.keys())
        if not merged_objects:
            merged_objects = self._extract_objects_fallback(raw_output)
        if not merged_objects and object_counts:
            merged_objects = list(object_counts.keys())

        if not description:
            description = raw_output[:500].strip()

        attributes["contains_people"] = bool(people_count and people_count > 0)
        attributes["human_activity"] = ", ".join(people_info["activities"])
        if people_info["positions"]:
            attributes["people_positions"] = people_info["positions"]
        attributes["people_visibility_uncertain"] = people_info["uncertainty"]
        attributes["people_count_source"] = people_source

        return ImageAnalysis(
            image_id=image_id,
            description=description,
            objects=merged_objects,
            object_counts=object_counts,
            people_count=people_count,
            annotated_image_path=detection_result.get("annotated_image_path"),
            relations=relations,
            attributes=attributes,
            raw_vlm_output=raw_output,
        )

    def _parse_sections(self, raw_output: str) -> dict[str, list[str]]:
        sections = {key: [] for key in SECTION_HEADERS}
        current_section = None

        for raw_line in raw_output.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            matched_section = None
            matched_text = ""
            for section_name, pattern in SECTION_HEADERS.items():
                match = pattern.match(line)
                if match:
                    matched_section = section_name
                    matched_text = match.group(1).strip()
                    break

            if matched_section:
                current_section = matched_section
                if matched_text:
                    sections[current_section].append(matched_text)
                continue

            if current_section:
                sections[current_section].append(line)

        return sections

    @staticmethod
    def _join_lines(lines: list[str]) -> str:
        return " ".join(part.strip() for part in lines if part.strip()).strip()

    @staticmethod
    def _split_items(text: str) -> list[str]:
        if not text:
            return []
        items = []
        for raw_item in re.split(r"[,;\n]", text):
            item = raw_item.strip().strip("-*")
            if item:
                items.append(item)
        return items

    def _parse_objects_section(self, lines: list[str]) -> list[str]:
        text = self._join_lines(lines)
        objects = self._split_items(text)
        if objects:
            return objects
        return self._extract_objects_fallback(text)

    def _parse_list_section(self, lines: list[str]) -> list[str]:
        return self._split_items(self._join_lines(lines))

    def _parse_attributes_section(self, lines: list[str]) -> dict:
        text = self._join_lines(lines)
        attributes: dict[str, str] = {}

        if not text:
            return attributes

        attributes["details"] = text

        for line in lines:
            normalized = line.strip().lstrip("-*").strip()
            if ":" in normalized:
                key, value = normalized.split(":", 1)
                attributes[key.strip().lower().replace(" ", "_")] = value.strip()

        return attributes

    def _merge_objects(self, detector_objects: list[str], vlm_objects: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()

        for label in detector_objects + vlm_objects:
            normalized = self._normalize_object_label(label)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)

        return merged

    @staticmethod
    def _normalize_object_label(label: str) -> str:
        normalized = str(label or "").strip().lower()
        return "person" if normalized in HUMAN_LABELS else normalized

    def _normalize_people(
        self,
        detector_people_count: int | None,
        people_info: dict,
        raw_output: str,
    ) -> tuple[int | None, str]:
        if detector_people_count is not None:
            return int(detector_people_count), "detector"

        parsed_count = people_info.get("count")
        if parsed_count is not None:
            return int(parsed_count), "vlm_visible_people"

        fallback_count = self._safe_parse_count(raw_output)
        if fallback_count is not None:
            return fallback_count, "regex_fallback"

        return 0, "regex_fallback"

    def _extract_people_info(self, visible_people_text: str, raw_output: str) -> dict:
        combined_text = f"{visible_people_text}\n{raw_output}"
        info = {
            "count": None,
            "positions": "",
            "activities": [],
            "uncertainty": False,
        }

        if visible_people_text:
            count = self._safe_parse_count(visible_people_text)
            if count is not None:
                info["count"] = count

            positions = self._extract_field_value(visible_people_text, "positions")
            activities = self._extract_field_value(visible_people_text, "activities")
            if positions:
                info["positions"] = positions
            if activities:
                info["activities"] = self._split_items(activities)

            if re.search(r"\b(uncertain|unclear|hard to tell|cannot tell|partially visible|occluded)\b", visible_people_text, re.IGNORECASE):
                info["uncertainty"] = True

        if info["count"] is None:
            info["count"] = self._safe_parse_count(combined_text)

        if not info["activities"]:
            activity_text = self._extract_field_value(combined_text, "activities")
            if activity_text:
                info["activities"] = self._split_items(activity_text)

        if not info["positions"]:
            positions_text = self._extract_field_value(combined_text, "positions")
            if positions_text:
                info["positions"] = positions_text

        if info["count"] is None and PERSON_REGEX.search(combined_text):
            info["count"] = max(1, len(set(match.group(0).lower() for match in PERSON_REGEX.finditer(combined_text))))

        return info

    @staticmethod
    def _extract_field_value(text: str, field_name: str) -> str:
        pattern = re.compile(rf"\b{re.escape(field_name)}\b\s*[:\-]?\s*(.*)$", re.IGNORECASE)
        values = []
        for line in text.splitlines():
            match = pattern.search(line.strip())
            if match:
                value = match.group(1).strip().lstrip("-*").strip()
                if value:
                    values.append(value)
        return " ".join(values).strip()

    def _safe_parse_count(self, text: str) -> int | None:
        if not text:
            return None

        lowered = text.lower()
        if any(phrase in lowered for phrase in ("no people", "no person", "nobody", "none visible", "no humans")):
            return 0

        for pattern in COUNT_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue

            value = match.group(1).lower()
            if value.isdigit():
                return int(value)
            if value in COUNT_WORDS:
                return COUNT_WORDS[value]

        person_mentions = PERSON_REGEX.findall(text)
        if person_mentions:
            return min(max(len(person_mentions), 1), 10)

        return None

    @staticmethod
    def _extract_objects_fallback(text: str) -> list[str]:
        """Fast fallback extractor for malformed VLM output."""
        found: list[str] = []
        seen: set[str] = set()

        for match in OBJECT_EXTRACTION_PATTERN.finditer(text.lower()):
            label = match.group(0).strip().lower()
            if label in seen:
                continue
            seen.add(label)
            found.append(label)

        return found


@lru_cache()
def get_vlm_service() -> VLMService:
    return VLMService()
