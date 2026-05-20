#!/usr/bin/env python3
"""
Debug script to test VLM service people counting on multiple sample images.
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

def test_multiple_images():
    # Find images to test
    uploads_dir = backend_dir / "uploads"
    
    # Get first few available images
    images = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        logger.error("No images found in uploads directory")
        return
    
    logger.info(f"Found {len(images)} images. Testing first 3...")
    
    # Get VLM service
    vlm_service = get_vlm_service()
    
    for i, image_name in enumerate(images[:3]):
        test_image_path = uploads_dir / image_name
        logger.info(f"\n=== Testing image {i+1}: {image_name} ===")
        
        # Load image
        try:
            image = Image.open(test_image_path).convert("RGB")
            logger.info(f"Image loaded: {image.size}")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            continue
        
        # Test HOG detection directly
        logger.info("Testing HOG people detection...")
        try:
            hog_count = vlm_service._count_people_with_hog(image)
            logger.info(f"HOG detection result: {hog_count}")
        except Exception as e:
            logger.error(f"HOG detection failed: {e}")
            hog_count = None
        
        # Test full analysis
        logger.info("Running full VLM analysis...")
        try:
            # Use the image name as ID for testing
            analysis = vlm_service.analyze_image(image, f"test_{i}")
            logger.info(f"Analysis complete:")
            logger.info(f"  Description: {analysis.description[:100]}...")
            logger.info(f"  Objects: {analysis.objects}")
            logger.info(f"  People count: {analysis.people_count}")
            logger.info(f"  Relations: {analysis.relations}")
            if analysis.attributes:
                logger.info(f"  Attributes keys: {list(analysis.attributes.keys())}")
            logger.info(f"  Raw output preview: {analysis.raw_vlm_output[:150]}...")
        except Exception as e:
            logger.error(f"Full analysis failed: {e}", exc_info=True)

if __name__ == "__main__":
    test_multiple_images()