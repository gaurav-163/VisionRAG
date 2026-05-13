"""
VisionRAG FastAPI application entry point.
"""

from fastapi import FastAPI

from .api.routes import router


app = FastAPI(
    title="VisionRAG",
    version="0.1.0",
    description="Vision analysis API for object detection and scene graphs.",
)

app.include_router(router)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "VisionRAG API is running",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.visionrag.main:app", host="0.0.0.0", port=8000, reload=True)
