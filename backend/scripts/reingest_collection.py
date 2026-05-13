"""Re-ingest images into ChromaDB by running the upload pipeline for each file.

This script will iterate over all files in `settings.upload_dir`, open each image,
and call the RAG pipeline `process_image` which saves the image, runs VLM analysis,
and upserts embeddings/metadata into ChromaDB. Use when you want to rebuild the
collection after a dimension-change or collection recreation.

WARNING: Running this will create new image IDs and may duplicate entries if the
collection was not deleted. Prefer to delete the `image_analysis` collection first
or run this after using the collection-recreation flow in the app.
"""
import logging
from pathlib import Path
from PIL import Image
import argparse

from ..app.services.rag_pipeline import get_rag_pipeline
from ..app.config import get_settings

logger = logging.getLogger("reingest")
logging.basicConfig(level=logging.INFO)


def main(session_id: str = "default"):
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    pipeline = get_rag_pipeline()

    if not upload_dir.exists():
        logger.error("Upload directory does not exist: %s", upload_dir)
        return

    files = sorted([p for p in upload_dir.iterdir() if p.is_file()])
    logger.info("Found %d files in %s", len(files), upload_dir)

    for p in files:
        try:
            logger.info("Processing %s", p.name)
            with Image.open(p) as img:
                pipeline.process_image(img, p.name, session_id=session_id)
        except Exception as e:
            logger.exception("Failed to process %s: %s", p.name, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-ingest images into ChromaDB")
    parser.add_argument("--session", "-s", default="default", help="Session ID to use for ingested images")
    args = parser.parse_args()
    main(session_id=args.session)
