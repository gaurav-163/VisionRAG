#!/usr/bin/env python3
"""
Simple test to verify VLM service improvements
"""

import sys
import os
from pathlib import Path

# Set up paths
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(backend_dir / ".env")

from PIL import Image
from app.services.vlm_service import get_vlm_service

def test_vlm_improvements():
    print("Testing VLM service improvements...")
    
    # Get VLM service
    vlm_service = get_vlm_service()
    print("✓ VLM service loaded successfully")
    
    # Test with an image
    uploads_dir = backend_dir / "uploads"
    image_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("✗ No images found in uploads directory")
        return
    
    # Test first image
    test_image_path = uploads_dir / image_files[0]
    print(f"Testing with image: {image_files[0]}")
    
    try:
        # Load image
        image = Image.open(test_image_path).convert("RGB")
        print(f"✓ Image loaded: {image.size}")
        
        # Run analysis
        analysis = vlm_service.analyze_image(image, "test_001")
        
        print(f"✓ Analysis completed:")
        print(f"  - Description: {analysis.description[:80]}...")
        print(f"  - Objects detected: {analysis.objects}")
        print(f"  - People count: {analysis.people_count}")
        print(f"  - Relations: {analysis.relations}")
        print(f"  - Attributes keys: {list(analysis.attributes.keys()) if analysis.attributes else 'None'}")
        
        # Verify we're getting reasonable results
        if analysis.objects:
            print("✓ Object detection working")
        else:
            print("⚠ No objects detected")
            
        if analysis.people_count is not None:
            print("✓ People counting working")
        else:
            print("⚠ People count not determined")
            
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vlm_improvements()