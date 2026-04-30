"""
Pydantic models — reused across API, services, and storage layers.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ── Image Schemas ──

class ImageMetadata(BaseModel):
    image_id: str
    filename: str
    filepath: str
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    width: int | None = None
    height: int | None = None
    description: str = ""
    objects: list[str] = Field(default_factory=list)
    attributes: dict = Field(default_factory=dict)
    correlations: list[dict] = Field(default_factory=list)


class ImageAnalysis(BaseModel):
    image_id: str
    description: str
    objects: list[str]
    attributes: dict
    raw_vlm_output: str


class CorrelationResult(BaseModel):
    image_id_a: str
    image_id_b: str
    shared_objects: list[str]
    relationship_description: str
    similarity_score: float


# ── Chat Schemas ──

class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    image_refs: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    referenced_images: list[str] = Field(default_factory=list)
    correlations_used: list[CorrelationResult] = Field(default_factory=list)


# ── Upload Schemas ──

class UploadResponse(BaseModel):
    image_id: str
    filename: str
    analysis: ImageAnalysis
    correlations: list[CorrelationResult] = Field(default_factory=list)
    message: str = "Image uploaded and analyzed successfully"


class SessionInfo(BaseModel):
    session_id: str
    image_count: int
    image_ids: list[str]