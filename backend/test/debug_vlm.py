#!/usr/bin/env python3
"""
Debug script to test VLM service people counting on a sample image.
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Load environment variables from backend/.env
from dotenv import load_dotenv
load_dotenv(backend_dir / ".env")

from PIL import Image
from app.services.vlm_service import get_vlm_service
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_people_counting():
    # Find an image to test
    uploads_dir = backend_dir / "uploads"
    
    # Get first available image
    images = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        logger.error("No images found in uploads directory")
        return
    
    test_image_path = uploads_dir / images[0]
    logger.info(f"Testing with image: {test_image_path}")
    
    # Load image
    try:
        image = Image.open(test_image_path).convert("RGB")
        logger.info(f"Image loaded: {image.size}")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return
    
    # Get VLM service
    vlm_service = get_vlm_service()
    
    # Test HOG detection directly
    logger.info("Testing HOG people detection...")
    try:
        hog_count = vlm_service._count_people_with_hog(image)
        logger.info(f"HOG detection result: {hog_count}")
    except Exception as e:
        logger.error(f"HOG detection failed: {e}")
    
    # Test full analysis
    logger.info("Running full VLM analysis...")
    try:
        # Use a simple image ID for testing
        analysis = vlm_service.analyze_image(image, "test_debug")
        logger.info(f"Analysis complete:")
        logger.info(f"  Description: {analysis.description[:100]}...")
        logger.info(f"  Objects: {analysis.objects}")
        logger.info(f"  People count: {analysis.people_count}")
        logger.info(f"  Relations: {analysis.relations}")
        logger.info(f"  Attributes keys: {list(analysis.attributes.keys())}")
        logger.info(f"  Raw output preview: {analysis.raw_vlm_output[:200]}...")
    except Exception as e:
        logger.error(f"Full analysis failed: {e}", exc_info=True)

if __name__ == "__main__":
    test_people_counting()