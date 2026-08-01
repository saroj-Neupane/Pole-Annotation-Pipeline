"""Main ASGI application for the Pole Annotation product app."""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from deploy.config import get_settings
from deploy.model_loader import is_models_loaded, load_all
from deploy.routers import inference, internal, pages
from deploy.security import setup_security_middleware
from deploy.shared import (
    STATIC_DIR,
    TEMPLATES_DIR,
    get_inference_executor,
)
from deploy.utils import load_html_template

logger = logging.getLogger(__name__)
_inference_executor = get_inference_executor()

@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    try:
        if not settings.skip_startup_model_load:
            load_all()
        else:
            logger.info("Skipping startup model load because SKIP_STARTUP_MODEL_LOAD is enabled.")
        yield
    finally:
        _inference_executor.shutdown(wait=True, cancel_futures=False)
        logger.info("Inference thread pool executor shut down.")


app = FastAPI(
    title="Pole Annotation",
    description="Pole Annotation product application",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

setup_security_middleware(app)

app.include_router(pages.router)
app.include_router(internal.router)
app.include_router(inference.router)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "metrics").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/favicon.ico")
async def favicon() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "logos" / "Pole_Annotation_Logo.png",
        media_type="image/png",
    )


@app.get("/health", response_class=HTMLResponse)
async def health_page() -> str:
    return load_html_template("health_page.html", TEMPLATES_DIR)


@app.get("/api/health-status")
async def health_check() -> dict[str, bool | str]:
    return {
        "status": "healthy" if is_models_loaded() else "degraded",
        "models_loaded": is_models_loaded(),
        "runtime": "onnxruntime-cpu",
    }


if __name__ == "__main__":
    uvicorn.run(
        "deploy.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
        reload_includes=["**/*.py", "**/*.html"],
    )
