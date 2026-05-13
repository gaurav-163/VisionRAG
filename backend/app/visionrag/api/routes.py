"""
Vision API Router
Exposes endpoints for object detection, scene graph generation,
and visualization for the VisionRAG pipeline.
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response

from ..models.schemas import VisionAnalysisResponse
from ..services.object_detection_service import ObjectDetectionService
from ..services.scene_graph_service import SceneGraphService
from ..utils.visualization import VisualizationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/vision", tags=["Vision Analysis"])

_detection_service: ObjectDetectionService = None
_scene_graph_service: SceneGraphService = None
_visualization_service: VisualizationService = None


def get_detection_service() -> ObjectDetectionService:
    global _detection_service
    if _detection_service is None:
        _detection_service = ObjectDetectionService(
            model_path="yolov8n.pt",
            confidence_threshold=0.35,
            device="cpu",
        )
    return _detection_service


def get_scene_graph_service() -> SceneGraphService:
    global _scene_graph_service
    if _scene_graph_service is None:
        _scene_graph_service = SceneGraphService()
    return _scene_graph_service


def get_visualization_service() -> VisualizationService:
    global _visualization_service
    if _visualization_service is None:
        _visualization_service = VisualizationService()
    return _visualization_service


@router.post(
    "/analyze",
    response_model=VisionAnalysisResponse,
    summary="Analyze image for objects and spatial relationships",
)
async def analyze_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP)"),
    detection_svc: ObjectDetectionService = Depends(get_detection_service),
    scene_graph_svc: SceneGraphService = Depends(get_scene_graph_service),
):
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. Accepted: {', '.join(allowed_types)}",
        )

    try:
        image_bytes = await file.read()
        image = detection_svc.decode_image(image_bytes)
        dimensions = detection_svc.get_image_dimensions(image)
        detections = detection_svc.detect(image)
        scene_graph = scene_graph_svc.build_scene_graph(
            detections=detections,
            image_width=dimensions["width"],
            image_height=dimensions["height"],
        )

        return VisionAnalysisResponse(
            status="success",
            image_dimensions=dimensions,
            detections=detections,
            scene_graph=scene_graph,
            detection_count=len(detections),
            message=f"Detected {len(detections)} objects with {scene_graph['summary']['total_relationships']} spatial relationships.",
        )
    except ValueError as e:
        logger.warning(f"Image decode error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.post(
    "/visualize/detections",
    summary="Return annotated image with bounding boxes",
    response_class=Response,
)
async def visualize_detections(
    file: UploadFile = File(...),
    detection_svc: ObjectDetectionService = Depends(get_detection_service),
    viz_svc: VisualizationService = Depends(get_visualization_service),
):
    try:
        image_bytes = await file.read()
        image = detection_svc.decode_image(image_bytes)
        detections = detection_svc.detect(image)
        annotated = viz_svc.draw_detections(image, detections)
        output_bytes = viz_svc.image_to_bytes(annotated, format=".jpg")
        return Response(content=output_bytes, media_type="image/jpeg", headers={"X-Detection-Count": str(len(detections))})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/visualize/scene-graph",
    summary="Return scene graph as PNG diagram",
    response_class=Response,
)
async def visualize_scene_graph(
    file: UploadFile = File(...),
    detection_svc: ObjectDetectionService = Depends(get_detection_service),
    scene_graph_svc: SceneGraphService = Depends(get_scene_graph_service),
    viz_svc: VisualizationService = Depends(get_visualization_service),
):
    try:
        image_bytes = await file.read()
        image = detection_svc.decode_image(image_bytes)
        dimensions = detection_svc.get_image_dimensions(image)
        detections = detection_svc.detect(image)
        scene_graph = scene_graph_svc.build_scene_graph(
            detections=detections,
            image_width=dimensions["width"],
            image_height=dimensions["height"],
        )
        graph_png = viz_svc.draw_scene_graph(scene_graph)
        return Response(content=graph_png, media_type="image/png", headers={"X-Node-Count": str(len(scene_graph["nodes"]))})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", summary="Health check for vision module")
async def health_check(
    detection_svc: ObjectDetectionService = Depends(get_detection_service),
):
    return {
        "status": "healthy",
        "model_loaded": detection_svc.model is not None,
        "model_path": detection_svc.model_path,
        "device": detection_svc.device,
        "confidence_threshold": detection_svc.confidence_threshold,
    }
