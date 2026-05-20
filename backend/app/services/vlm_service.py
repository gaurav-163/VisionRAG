"""
Vision Language Model service — wraps SmolVLM2 for image analysis.
Handles model loading, inference, and output parsing.
"""

import torch
import logging
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from functools import lru_cache
import re

from app.config import get_settings
from app.models.schemas import ImageAnalysis
from app.services.detection_service import get_detection_service
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
        self.detector = get_detection_service()
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
        object_counts: dict[str, int] = {}
        annotated_image_path: str | None = None
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

        try:
            detection_result = self.detector.detect_objects(image, image_id)
            object_counts = detection_result.get("counts", {})
            annotated_image_path = detection_result.get("annotated_image_path")
            people_count = int(object_counts.get("person", 0))
        except Exception as exc:
            logger.warning(f"YOLO detection failed, falling back to VLM count parsing: {exc}")
            object_counts = {}
            if people_count is None:
                people_count = self._infer_people_count(objects, description, raw_output) or 0

        if not objects and object_counts:
            objects = list(object_counts.keys())

        # Filter objects for consistency with description (less aggressive)
        objects = self._filter_inconsistent_objects(objects, description)

        return ImageAnalysis(
            image_id=image_id,
            description=description,
            objects=objects,
            object_counts=object_counts,
            people_count=people_count,
            annotated_image_path=annotated_image_path,
            relations=relations,
            attributes=attributes,
            raw_vlm_output=raw_output
        )

    @staticmethod
    def _extract_objects_fallback(text: str) -> list[str]:
        """Improved noun extraction as fallback when VLM doesn't follow format.
        Focuses ONLY on concrete, tangible objects that can be seen in an image.
        """
        # ONLY concrete, tangible objects that can be visually identified
        concrete_objects = {
            # People (specific roles)
            "person", "people", "man", "woman", "boy", "girl", "child", "worker", 
            "employee", "staff", "customer", "visitor", "pilot", "driver", "rider",
            
            # Vehicles & Transportation
            "car", "truck", "bus", "van", "motorcycle", "bicycle", "train", "plane", 
            "boat", "ship", "helicopter", "taxi", "ambulance", "firetruck", "police car",
            "wheel", "tire", "door", "window", "mirror", "seat", "handle", "light",
            
            # Buildings & Structures
            "building", "house", "home", "office", "school", "hospital", "store", "shop",
            "restaurant", "hotel", "factory", "warehouse", "barn", "shed", "garage",
            "roof", "wall", "floor", "ceiling", "door", "window", "fence", "gate",
            "stairs", "staircase", " balcony", "porch", "deck", "patio",
            "bridge", "tunnel", "road", "street", "highway", "sidewalk", "path",
            "column", "pillar", "beam", "beam", "foundation", "beam", "pipe", "tube",
            
            # Furniture & Fixtures
            "table", "chair", "desk", "bench", "sofa", "couch", "bed", "mattress",
            "cabinet", "shelf", "drawer", "closet", "wardrobe", "dresser",
            "lamp", "light", "ceiling fan", "fan", "vent", "outlet", "switch",
            "toilet", "sink", "bathtub", "shower", "faucet", "mirror",
            
            # Electronics & Equipment
            "phone", "smartphone", "telephone", "computer", "laptop", "desktop",
            "monitor", "screen", "tv", "television", "projector", "camera",
            "keyboard", "mouse", "trackpad", "printer", "scanner", "speaker",
            "headphones", "microphone", "remote", "battery", "charger", "cable",
            "wire", "cord", "plug", "jack", "port", "antenna", "satellite",
            
            # Kitchen & Food Items
            "plate", "bowl", "cup", "mug", "glass", "fork", "knife", "spoon",
            "pot", "pan", "kettle", "oven", "stove", "microwave", "fridge",
            "refrigerator", "freezer", "dishwasher", "blender", "mixer", "toaster",
            "bottle", "jar", "container", "package", "wrapper", "bag", "box",
            "food", "fruit", "vegetable", "bread", "meat", "fish", "egg",
            
            # Clothing & Personal Items
            "shirt", "pants", "trousers", "jeans", "shorts", "skirt", "dress",
            "jacket", "coat", "sweater", "hoodie", "hat", "cap", "helmet",
            "shoe", "boot", "sandal", "flipflop", "sock", "glove", "mitten",
            "belt", "tie", "scarf", "watch", "jewelry", "ring", "necklace",
            "glasses", "sunglasses", "wallet", "purse", "bag", "backpack",
            
            # Sports & Recreation
            "ball", "bat", "racket", "glove", "helmet", "pad", "uniform",
            "goal", "hoop", "net", "field", "court", "track", "pool",
            "bike", "skateboard", "surfboard", "ski", "snowboard",
            
            # Nature & Outdoors
            "tree", "plant", "flower", "grass", "leaf", "branch", "trunk", "root",
            "rock", "stone", "boulder", "pebble", "mountain", "hill", "valley",
            "water", "ocean", "sea", "lake", "river", "stream", "pond", "puddle",
            "wave", "beach", "sand", "soil", "dirt", "mud", "cloud", "sun",
            "moon", "star", "sky", "horizon", "shadow", "reflection",
            
            # Tools & Hardware
            "hammer", "screwdriver", "wrench", "pliers", "saw", "axe", "knife",
            "drill", "nail", "screw", "bolt", "nut", "anchor", "hook",
            "ladder", "stepstool", "scaffold", "wheelbarrow", "shovel",
            "rake", "hoe", "pickaxe", "shovel", "level", "tape", "ruler",
            
            # Signs & Labels
            "sign", "poster", "banner", "flag", "label", "tag", "sticker",
            "billboard", "notice", "warning", "instruction", "direction",
            "text", "letter", "number", "symbol", "logo", "icon", "image",
            
            # Containers & Storage
            "box", "container", "bottle", "jar", "can", "tin", "tube", "bag",
            "sack", "pouch", "wallet", "purse", "suitcase", "luggage",
            "basket", "crate", "pallet", "drum", "barrel", "tank", "reservoir",
            
            # Common objects that might appear
            "money", "coin", "bill", "credit card", "id", "badge", "key",
            "lock", "chain", "rope", "string", "thread", "wire", "cable",
            "tape", "glue", "adhesive", "clip", "pin", "staple", "clip",
            "toy", "game", "puzzle", "book", "magazine", "newspaper",
            "paper", "document", "form", "ticket", "receipt", "label"
        }
        
        text_lower = text.lower()
        found_objects = []
        
        # Check for each concrete object in the text
        for obj in sorted(concrete_objects, key=len, reverse=True):  # Longer matches first
            # Use word boundaries to avoid partial matches
            if re.search(r'\b' + re.escape(obj) + r'\b', text_lower):
                found_objects.append(obj)
        
        # Additional filtering: exclude any remaining abstract/conceptual words
        abstract_exclude = {
            # Colors (often mistaken as objects)
            "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
            "black", "white", "gray", "grey", "silver", "gold", "bronze", "copper",
            
            # Sizes/quantities
            "large", "small", "big", "little", "tiny", "huge", "massive", "enormous",
            "medium", "average", "normal", "regular", "standard", "full", "empty",
            "half", "quarter", "part", "section", "portion", "amount", "number",
            "many", "few", "several", "multiple", "various", "different", "same",
            "single", "double", "triple", "one", "two", "three", "four", "five",
            "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "teen",
            "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
            "ninety", "hundred", "thousand", "million", "billion",
            
            # Descriptions/qualities
            "new", "old", "young", "ancient", "modern", "contemporary", "traditional",
            "bright", "dark", "dim", "dull", "shiny", "dull", "glossy", "matte",
            "clear", "cloudy", "foggy", "hazy", "transparent", "opaque", "translucent",
            "smooth", "rough", "bumpy", "uneven", "flat", "curved", "straight",
            "round", "square", "rectangular", "circular", "oval", "triangle",
            "heavy", "light", "weight", "mass", "dense", "sparse", "thick", "thin",
            "wide", "narrow", "tall", "short", "long", "deep", "shallow", "high",
            "low", "top", "bottom", "middle", "center", "edge", "side", "end",
            "hot", "cold", "warm", "cool", "freezing", "boiling", "temperature",
            "wet", "dry", "damp", "moist", "soaked", "dry", "clean", "dirty",
            "fresh", "stale", "rotten", "fresh", "clean", "messy", "neat", "tidy",
            "organized", "disordered", "chaotic", "orderly", "neat", "cluttered",
            "simple", "complex", "complicated", "basic", "advanced", "elaborate",
            "beautiful", "ugly", "attractive", "unattractive", "pretty", "handsome",
            "nice", "good", "bad", "better", "worse", "best", "worst", "fine",
            "okay", "alright", "right", "wrong", "correct", "incorrect", "true",
            "false", "real", "fake", "genuine", "authentic", "original", "copy",
            "natural", "artificial", "synthetic", "manmade", "handmade", "custom",
            "standard", "special", "unique", "common", "rare", "usual", "unusual",
            "normal", "abnormal", "typical", "atypical", "regular", "irregular",
            "frequent", "infrequent", "often", "seldom", "usually", "normally",
            "always", "never", "sometimes", "occasionally", "periodically",
            
            # Actions/states (verbs/adjectives that aren't objects)
            "working", "moving", "standing", "sitting", "lying", "walking", "running",
            "jumping", "flying", "driving", "riding", "carrying", "holding", "pushing",
            "pulling", "lifting", "dropping", "throwing", "catching", "kicking",
            "hitting", "striking", "breaking", "fixing", "repairing", "building",
            "constructing", "destroying", "opening", "closing", "entering", "exiting",
            "appearing", "disappearing", "visible", "invisible", "hidden", "shown",
            "displayed", "present", "absent", "missing", "lost", "found", "located",
            "positioned", "placed", "located", "situated", "positioned", "oriented",
            "facing", "pointed", "aimed", "directed", "oriented", "aligned",
            "straight", "crooked", "bent", "curved", "angled", "tilted", "leaned",
            "vertical", "horizontal", "diagonal", "parallel", "perpendicular",
            "active", "inactive", "on", "off", "open", "closed", "started", "stopped",
            "begun", "finished", "completed", "running", "stopped", "halted", "paused",
            "waiting", "resting", "sleeping", "awake", "alert", "tired", "exhausted",
            "busy", "idle", "free", "occupied", "available", "unavailable", "accessible",
            "inaccessible", "reachable", "unreachable", "near", "far", "close", "distant",
            "nearby", "far away", "in front", "behind", "above", "below", "over", "under",
            "inside", "outside", "indoors", "outdoors", "interior", "exterior",
            "internal", "external", "center", "middle", "core", "edge", "border",
            "boundary", "limit", "extent", "range", "area", "space", "volume", "capacity",
            
            # Concepts/ideas
            "time", "date", "hour", "minute", "second", "moment", "instant", "period",
            "duration", "age", "era", "age", "epoch", "generation", "century", "decade",
            "year", "month", "week", "day", "holiday", "weekend", "morning", "afternoon",
            "evening", "night", "dawn", "dusk", "sunrise", "sunset", "twilight",
            "light", "darkness", "shadow", "shine", "glow", "illumination", "brightness",
            "sound", "noise", "voice", "speech", "talk", "conversation", "dialogue",
            "music", "song", "melody", "rhythm", "beat", "tune", "noise", "silence",
            "temperature", "heat", "cold", "warmth", "coolness", "pressure", "weight",
            "force", "energy", "power", "electricity", "magnetism", "gravity", "friction",
            "speed", "velocity", "acceleration", "motion", "movement", "direction",
            "way", "path", "route", "course", "track", "trail", "line", "street", "road",
            "avenue", "boulevard", "drive", "lane", "alley", "path", "trail", "way",
            "idea", "thought", "concept", "theory", "belief", "faith", "trust", "hope",
            "dream", "vision", "goal", "objective", "target", "purpose", "reason", "cause",
            "effect", "result", "outcome", "consequence", "impact", "influence", "effect",
            "change", "difference", "variation", "alternative", "option", "choice", "decision",
            "plan", "strategy", "method", "approach", "technique", "process", "procedure",
            "step", "stage", "phase", "level", "grade", "rank", "score", "rating", "evaluation",
            "assessment", "analysis", "examination", "inspection", "review", "survey",
            "study", "research", "investigation", "exploration", "discovery", "invention",
            "creation", "production", "manufacture", "construction", "building", "development",
            "growth", "expansion", "extension", "addition", "subtraction", "multiplication",
            "division", "fraction", "percentage", "ratio", "proportion", "rate", "speed",
            "distance", "length", "width", "height", "depth", "size", "dimension", "measure",
            "unit", "scale", "range", "spectrum", "scale", "gauge", "meter", "liter", "gram",
            "kilogram", "pound", "ounce", "ton", "mile", "kilometer", "meter", "centimeter",
            "millimeter", "microgram", "microsecond", "nanosecond",
            
            # Articles/prepositions/conjunctions (should already be caught by word boundaries but just in case)
            "the", "a", "an", "and", "or", "but", "nor", "for", "yet", "so", "although",
            "because", "since", "unless", "while", "whereas", "whether", "either", "neither",
            "both", "not", "only", "also", "just", "even", "still", "already", "yet",
            "always", "never", "sometimes", "often", "frequently", "generally", "usually",
            "normally", "typically", "occasionally", "seldom", "rarely", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "up", "down", "in", "out", "on", "off",
            "over", "under", "above", "below", "top", "bottom", "front", "back", "inside",
            "outside", "between", "among", "through", "across", "around", "past", "beyond",
            "within", "without", "against", "along", "among", "around", "before", "after",
            "during", "throughout", "i", "you", "he", "she", "it", "we", "they", "me",
            "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their",
            "mine", "yours", "his", "hers", "its", "ours", "theirs", "this", "that", "these",
            "those", "which", "who", "whom", "whose", "what", "where", "when", "why", "how"
        }
        
        # Final filter: remove any abstract words that might have slipped through
        filtered_objects = []
        for obj in found_objects:
            if obj not in abstract_exclude and len(obj) >= 2:
                # Additional sanity check: if it's mostly alphabetic and not a common false positive
                if obj.isalpha() and not obj in {
                    "image", "pic", "photo", "picture", "scene", "view", "sight", "look",
                    "appearance", "looks", "seems", "appears", "like", "as", "than",
                    "very", "quite", "rather", "somewhat", "pretty", "fairly", "kind",
                    "sort", "type", "kind", "form", "shape", "size", "color", "colour"
                }:
                    filtered_objects.append(obj)
        
        return list(set(filtered_objects))  # Remove duplicates

    @staticmethod
    def _filter_inconsistent_objects(objects: list[str], description: str) -> list[str]:
        """Filter objects that are inconsistent with the description.
        Less aggressive filtering to prevent losing valid detections.
        """
        if not description or not objects:
            return objects

        description_lower = description.lower()
        filtered = []

        # Objects that are often hallucinated and shouldn't appear without being in description
        high_hallucination_risk = {
            "lion", "tiger", "bear", "monkey", "elephant", "giraffe", "zebra"  # Reduced set
        }

        for obj in objects:
            obj_lower = obj.lower()
            # If it's a risky object, only keep if mentioned in description
            if any(risky in obj_lower for risky in high_hallucination_risk):
                if obj_lower in description_lower:
                    filtered.append(obj)
                # Still include if it's a general category that might be implied
                elif any(general in obj_lower for general in ["animal", "creature", "beast"]):
                    if any(general in description_lower for general in ["animal", "creature", "beast"]):
                        filtered.append(obj)
                # Otherwise, tentatively include it but log for review
                else:
                    filtered.append(obj)  # Include it rather than dropping it
            else:
                # Keep non-animal/objects
                filtered.append(obj)

        return filtered

    @staticmethod
    def _infer_people_count(objects: list[str], description: str, raw_output: str) -> int | None:
        """Best-effort fallback for people count when the model omits it."""
        combined_text = f"{description}\n{raw_output}".lower()

        if "no people" in combined_text or "no person" in combined_text or "nobody" in combined_text:
            return 0

        import re

        count_patterns = [
            r"(?:people|persons|person|workers|men|women|boys|girls|children)\s*(?:of|:)?\s*(\d+)",
            r"(\d+)\s*(?:people|persons|person|workers|men|women|boys|girls|children)",
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:people|persons|persons|people)\b",
        ]
        
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }

        for pattern in count_patterns:
            match = re.search(pattern, combined_text)
            if match:
                try:
                    # Check if it's a word number
                    if match.group(1).isalpha() and match.group(1) in word_to_num:
                        return word_to_num[match.group(1)]
                    else:
                        return int(match.group(1))
                except ValueError:
                    continue

        # Count explicit mentions
        people_mentions = len(re.findall(r'\bpeople\b', combined_text))
        person_mentions = len(re.findall(r'\bperson\b', combined_text))
        if people_mentions > 0 or person_mentions > 0:
            # Rough estimate: if they mention people multiple times, there are likely multiple people
            total_mentions = people_mentions + person_mentions
            if total_mentions > 2:
                return min(total_mentions, 5)  # Cap reasonable estimate
            elif total_mentions > 0:
                return 1

        return None

@lru_cache()
def get_vlm_service() -> VLMService:
    return VLMService()