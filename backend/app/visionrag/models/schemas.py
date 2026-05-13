"""
Pydantic schemas for API request/response validation.
"""

from typing import Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int = Field(..., description="Left edge X coordinate")
    y1: int = Field(..., description="Top edge Y coordinate")
    x2: int = Field(..., description="Right edge X coordinate")
    y2: int = Field(..., description="Bottom edge Y coordinate")


class Center(BaseModel):
    x: int = Field(..., description="Centroid X coordinate")
    y: int = Field(..., description="Centroid Y coordinate")


class Detection(BaseModel):
    label: str = Field(..., description="Object class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: BoundingBox = Field(..., description="Bounding box pixel coordinates")
    center: Center = Field(..., description="Bounding box centroid")
    area: int = Field(..., description="Bounding box area in pixels")


class SceneGraphNode(BaseModel):
    id: str = Field(..., description="Unique node identifier")
    label: str = Field(..., description="Object class label")
    confidence: float = Field(..., description="Detection confidence")
    bbox: BoundingBox
    center: Center
    area: int


class SceneGraphEdge(BaseModel):
    subject: str = Field(..., description="Subject node ID")
    subject_label: str = Field(..., description="Subject class label")
    relationship: str = Field(..., description="Spatial relationship type")
    object: str = Field(..., description="Object node ID")
    object_label: str = Field(..., description="Object class label")


class Triplet(BaseModel):
    subject: str = Field(..., description="Subject class label")
    relationship: str = Field(..., description="Spatial relationship")
    object: str = Field(..., description="Object class label")


class SceneGraphSummary(BaseModel):
    total_objects: int
    unique_categories: list[str]
    total_relationships: int
    relationship_distribution: dict[str, int]
    most_connected_object: Optional[str]
    scene_complexity: str


class SceneGraph(BaseModel):
    nodes: list[SceneGraphNode]
    edges: list[SceneGraphEdge]
    triplets: list[Triplet]
    summary: SceneGraphSummary


class ImageDimensions(BaseModel):
    height: int
    width: int
    channels: int


class VisionAnalysisResponse(BaseModel):
    status: str = Field(default="success")
    image_dimensions: ImageDimensions
    detections: list[Detection]
    scene_graph: SceneGraph
    detection_count: int
    message: str = Field(default="Analysis complete")


class ErrorResponse(BaseModel):
    status: str = Field(default="error")
    message: str
    detail: Optional[str] = None
