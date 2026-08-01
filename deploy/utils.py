"""Utility functions for deployment routes and templates."""
import base64
import re
import logging
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)


def image_to_base64_data_url(image_bgr: np.ndarray) -> str:
    """Encode a BGR image as a JPEG base64 data URL."""
    _, buf = cv2.imencode(".jpg", image_bgr)
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"


# HTML template cache
_template_cache: Dict[str, str] = {}


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


async def validate_upload_file(
    file: UploadFile,
    max_size_mb: int = 50,
    allowed_types: list = None
) -> None:
    """
    Validate uploaded file for security.

    Args:
        file: The uploaded file
        max_size_mb: Maximum file size in megabytes
        allowed_types: List of allowed MIME types (e.g., ['image/jpeg', 'image/png'])

    Raises:
        HTTPException: If validation fails
    """
    if allowed_types is None:
        allowed_types = ['image/jpeg', 'image/png', 'image/webp', 'image/tiff']

    # Validate content type
    if not file.content_type or file.content_type not in allowed_types:
        logger.warning(f"File upload rejected: invalid content type {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
        )

    # Validate filename (prevent path traversal)
    if file.filename and (".." in file.filename or "/" in file.filename or "\\" in file.filename):
        logger.warning(f"File upload rejected: suspicious filename {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Validate file size
    max_bytes = max_size_mb * 1024 * 1024
    file_size = 0
    while True:
        chunk = await file.read(1024 * 1024)  # Read in 1MB chunks
        if not chunk:
            break
        file_size += len(chunk)
        if file_size > max_bytes:
            logger.warning(f"File upload rejected: file too large ({file_size} bytes > {max_bytes} bytes)")
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {max_size_mb}MB"
            )

    # Reset file pointer for later reading
    await file.seek(0)


def serve_image_file(image_dir: Path, filename: str) -> FileResponse:
    """
    Serve an image file from a directory.
    Prevents path traversal by resolving paths and ensuring result stays under image_dir.

    Args:
        image_dir: Directory containing the image
        filename: Name of the image file (path components stripped for security)

    Returns:
        FileResponse with the image

    Raises:
        HTTPException: If image not found or path escapes directory
    """
    base = image_dir.resolve()
    safe_name = Path(filename).name
    if not safe_name or safe_name.startswith("."):
        raise HTTPException(status_code=404, detail="Image not found")
    full_path = (image_dir / safe_name).resolve()
    try:
        if not full_path.is_relative_to(base) or not full_path.is_file():
            raise HTTPException(status_code=404, detail="Image not found")
    except (ValueError, AttributeError):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(full_path)


def load_html_template(template_name: str, deploy_dir: Path) -> str:
    """
    Load an HTML template file with caching.
    In development mode, always reload from disk to pick up changes.

    Args:
        template_name: Name of the template file
        deploy_dir: Directory containing templates

    Returns:
        HTML content as string
    """
    from deploy.config import get_settings
    cache_templates = get_settings().is_production

    template_path = deploy_dir / template_name
    cache_key = str(template_path)

    # Check cache first (for production only; dev always reloads to pick up edits)
    if cache_key in _template_cache and cache_templates:
        return _template_cache[cache_key]
    
    # Load from disk and cache (cache used only in production)
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    _template_cache[cache_key] = content
    return content
