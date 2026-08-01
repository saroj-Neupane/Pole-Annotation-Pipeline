"""Demo image management utilities.

Handles loading demo images from local storage for use in landing page
and demo endpoints.
"""

import os
from pathlib import Path
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_demo_images_dir(image_type: str) -> Path:
    """
    Get the demo images directory for a given type.

    Args:
        image_type: 'pole' or 'midspan'

    Returns:
        Path to the demo images directory

    Raises:
        ValueError: If image_type is invalid
    """
    if image_type not in ('pole', 'midspan'):
        raise ValueError(f"Invalid image_type: {image_type}. Must be 'pole' or 'midspan'")

    project_root = Path(__file__).parent.parent
    return project_root / "inference" / image_type / "images"


def list_demo_images(image_type: str) -> List[Path]:
    """
    List all demo images of a given type.

    Args:
        image_type: 'pole' or 'midspan'

    Returns:
        List of image paths, sorted alphabetically
    """
    demo_dir = get_demo_images_dir(image_type)

    if not demo_dir.exists():
        logger.warning(f"Demo images directory not found: {demo_dir}")
        return []

    images = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        images.extend(sorted(demo_dir.glob(ext)))

    return sorted(images)


def list_demo_images_for_api(image_type: str) -> list:
    """
    List demo images in API format: [{"filename": str, "path": str}, ...].
    Used by /api/images/list endpoint.
    """
    paths = list_demo_images(image_type)
    return [
        {"filename": p.name, "path": f"/api/images/{image_type}/{p.name}"}
        for p in paths
    ]


def count_demo_images() -> dict:
    """
    Count available demo images by type.

    Returns:
        Dict with counts: {'pole': int, 'midspan': int}
    """
    return {
        'pole': len(list_demo_images('pole')),
        'midspan': len(list_demo_images('midspan')),
    }


def demo_images_available() -> bool:
    """Check if any demo images are available."""
    counts = count_demo_images()
    return counts['pole'] > 0 or counts['midspan'] > 0


def get_demo_image_info() -> dict:
    """
    Get information about available demo images.

    Returns:
        Dict with counts and list of available types
    """
    counts = count_demo_images()
    available_types = [k for k, v in counts.items() if v > 0]

    return {
        'total': sum(counts.values()),
        'by_type': counts,
        'available_types': available_types,
    }
