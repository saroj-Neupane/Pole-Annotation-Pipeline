"""Shared state and paths for the deploy module."""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

_inference_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("INFERENCE_WORKERS", "4")),
    thread_name_prefix="inference",
)


def get_inference_executor() -> ThreadPoolExecutor:
    return _inference_executor


TEMPLATES_DIR = DEPLOY_DIR / "templates"
MIDSPAN_IMAGES_DIR = PROJECT_ROOT / "inference" / "midspan" / "images"
SPANS_DIR = PROJECT_ROOT / "inference" / "spans"
POLE_IMAGES_DIR = PROJECT_ROOT / "inference" / "pole" / "images"
LOGOS_DIR = DEPLOY_DIR / "static" / "logos"
STATIC_DIR = DEPLOY_DIR / "static"
