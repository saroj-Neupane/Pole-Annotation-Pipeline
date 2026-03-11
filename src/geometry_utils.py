"""
Geometry utilities for overlap and spatial checks.
"""

from typing import Optional, Tuple


def bbox_perfect_overlap(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int],
    tolerance: int = 2,
) -> Optional[Tuple[int, int, int, int]]:
    """Check if two bounding boxes perfectly overlap (within tolerance).

    Args:
        bbox1: (x1, y1, x2, y2) for first bbox
        bbox2: (x1, y1, x2, y2) for second bbox
        tolerance: Pixel tolerance for perfect match (default: 2 pixels)

    Returns:
        Intersection bbox (x1, y1, x2, y2) if they perfectly overlap, None otherwise
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    if (
        abs(x1_1 - x1_2) <= tolerance
        and abs(y1_1 - y1_2) <= tolerance
        and abs(x2_1 - x2_2) <= tolerance
        and abs(y2_1 - y2_2) <= tolerance
    ):
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        if x1_int < x2_int and y1_int < y2_int:
            return (x1_int, y1_int, x2_int, y2_int)
    return None


def line_perfect_overlap(
    line1: Tuple[int, int, int],
    line2: Tuple[int, int, int],
    tolerance: int = 2,
) -> Optional[Tuple[int, int, int]]:
    """Check if two horizontal lines perfectly overlap (within tolerance).

    Args:
        line1: (x_start, y, x_end) for first line
        line2: (x_start, y, x_end) for second line
        tolerance: Pixel tolerance for perfect match (default: 2 pixels)

    Returns:
        Overlapping segment (x_start, y, x_end) if they perfectly overlap, None otherwise
    """
    x1_start, y1, x1_end = line1
    x2_start, y2, x2_end = line2

    if abs(y1 - y2) > tolerance:
        return None

    if abs(x1_start - x2_start) <= tolerance and abs(x1_end - x2_end) <= tolerance:
        x_start = max(x1_start, x2_start)
        x_end = min(x1_end, x2_end)
        if x_start < x_end:
            return (x_start, y1, x_end)
    return None
