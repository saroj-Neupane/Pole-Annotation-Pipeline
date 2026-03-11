"""
Bounding box calculations for poles, rulers, and equipment.

Provides utilities for calculating symmetric bounding boxes around vertical structures
and equipment based on keypoint coordinates and physical dimensions.
"""

from typing import Dict, Optional

import numpy as np

from .config import (
    ATTACHMENT_BBOX_HEIGHT_FEET,
    ATTACHMENT_BBOX_WIDTH_FEET,
    RISER_BBOX_HEIGHT_FEET,
    RISER_BBOX_WIDTH_FEET,
    SECONDARY_DRIP_LOOP_BBOX_HEIGHT_FEET,
    SECONDARY_DRIP_LOOP_BBOX_WIDTH_FEET,
    STREET_LIGHT_BBOX_HEIGHT_FEET,
    STREET_LIGHT_BBOX_WIDTH_FEET,
    TRANSFORMER_BBOX_HEIGHT_FEET,
    TRANSFORMER_BBOX_WIDTH_FEET,
)
from .height_calculations import calculate_ground_coordinates, calculate_ruler_top_coordinates


def calculate_symmetric_bounding_box(
    start_coords: Dict,
    end_coords: Dict,
    width_pct: float,
    extend_top_pct: float = 0.05,
    extend_bottom_pct: float = 0.03,
) -> Optional[Dict]:
    """Calculate axis-aligned bounding box symmetric about line from start to end.

    Args:
        start_coords: Starting point dict with percentX, percentY
        end_coords: Ending point dict with percentX, percentY
        width_pct: Box width as percentage of line length (0.0-1.0)
        extend_top_pct: Extension above top point as fraction of height (default 0.05 = 5%)
        extend_bottom_pct: Extension below bottom point as fraction of height (default 0.03 = 3%)

    Returns:
        Bounding box dict with left, right, top, bottom, center_x, center_y, width, height, etc.
    """
    start_x = start_coords['percentX']
    start_y = start_coords['percentY']
    end_x = end_coords['percentX']
    end_y = end_coords['percentY']

    # Calculate the line vector
    line_dx = end_x - start_x
    line_dy = end_y - start_y
    line_length = np.sqrt(line_dx**2 + line_dy**2)

    if line_length == 0:
        return None

    # Normalize the line vector
    line_unit_x = line_dx / line_length
    line_unit_y = line_dy / line_length

    # Calculate perpendicular vector (rotated 90 degrees)
    perp_x = -line_unit_y
    perp_y = line_unit_x

    # Calculate box width
    rect_width = line_length * width_pct
    half_width = rect_width / 2

    # Calculate the four corners of the rectangle
    start_left_x = start_x + perp_x * half_width
    start_left_y = start_y + perp_y * half_width
    start_right_x = start_x - perp_x * half_width
    start_right_y = start_y - perp_y * half_width

    end_left_x = end_x + perp_x * half_width
    end_left_y = end_y + perp_y * half_width
    end_right_x = end_x - perp_x * half_width
    end_right_y = end_y - perp_y * half_width

    # Calculate bounding box boundaries
    all_x = [start_left_x, start_right_x, end_left_x, end_right_x]
    all_y = [start_left_y, start_right_y, end_left_y, end_right_y]

    left_x = min(all_x)
    right_x = max(all_x)
    top_y = min(all_y)
    bottom_y = max(all_y)

    # Extend bounding box
    vertical_height = bottom_y - top_y
    top_y = max(0.0, top_y - vertical_height * extend_top_pct)
    bottom_y = min(100.0, bottom_y + vertical_height * extend_bottom_pct)

    final_height = bottom_y - top_y
    center_x = (left_x + right_x) / 2
    center_y = (top_y + bottom_y) / 2

    return {
        'left': left_x,
        'right': right_x,
        'top': top_y,
        'bottom': bottom_y,
        'center_x': center_x,
        'center_y': center_y,
        'width': rect_width,
        'height': final_height,
        'start_x': start_x,
        'start_y': start_y,
        'end_x': end_x,
        'end_y': end_y,
    }


def calculate_bounding_box(height_measurements: Dict) -> Optional[Dict]:
    """Calculate pole bounding box using ground (0ft) and pole_top coordinates.

    Uses 20% width ratio for ~5:1 aspect ratio (height:width).
    """
    ground_coords = calculate_ground_coordinates(height_measurements)
    if not ground_coords:
        return None

    pole_top_coords = None
    for key, measurement in height_measurements.items():
        if str(key).startswith('pole_top_'):
            pole_top_coords = (measurement['percentX'], measurement['percentY'])
            break

    if not pole_top_coords:
        return None

    return calculate_symmetric_bounding_box(
        {'percentX': ground_coords['percentX'], 'percentY': ground_coords['percentY']},
        {'percentX': pole_top_coords[0], 'percentY': pole_top_coords[1]},
        width_pct=0.20,
        extend_top_pct=0.05,
        extend_bottom_pct=0.03,
    )


def calculate_ruler_bounding_box(height_measurements: Dict) -> Optional[Dict]:
    """Calculate ruler bounding box using ground (0ft) and ruler_top (17ft) coordinates.

    Uses 10% width ratio for ~10:1 aspect ratio (height:width).
    """
    ground_coords = calculate_ground_coordinates(height_measurements)
    if not ground_coords:
        return None

    ruler_top_coords = calculate_ruler_top_coordinates(height_measurements)
    if not ruler_top_coords:
        return None

    return calculate_symmetric_bounding_box(
        {'percentX': ground_coords['percentX'], 'percentY': ground_coords['percentY']},
        {'percentX': ruler_top_coords['percentX'], 'percentY': ruler_top_coords['percentY']},
        width_pct=0.10,
        extend_top_pct=0.05,
        extend_bottom_pct=0.03,
    )


def calculate_riser_bounding_box(
    riser_coords: Dict,
    ground_coords: Dict,
    ppi: float,
    img_w: int,
    img_h: int,
) -> Optional[Dict]:
    """Riser bbox (H x W) centered on riser point. Size from config. No padding."""
    if not riser_coords or not ppi or ppi <= 0:
        return None

    box_w_px = RISER_BBOX_WIDTH_FEET * 12.0 * ppi
    box_h_px = RISER_BBOX_HEIGHT_FEET * 12.0 * ppi

    cx_px = riser_coords['percentX'] / 100.0 * img_w
    cy_px = riser_coords['percentY'] / 100.0 * img_h

    top_px = cy_px - box_h_px / 2
    bottom_px = cy_px + box_h_px / 2

    return {
        'left': max(0.0, (cx_px - box_w_px / 2) / img_w * 100.0),
        'right': min(100.0, (cx_px + box_w_px / 2) / img_w * 100.0),
        'top': max(0.0, top_px / img_h * 100.0),
        'bottom': min(100.0, bottom_px / img_h * 100.0),
    }


def calculate_transformer_bounding_box(
    top_coords: Dict,
    bottom_coords: Dict,
    ppi: float,
    img_w: int,
    img_h: int,
) -> Optional[Dict]:
    """Transformer bbox centred on keypoints. Size from config."""
    if not top_coords or not bottom_coords or not ppi or ppi <= 0:
        return None

    height_px = TRANSFORMER_BBOX_HEIGHT_FEET * 12.0 * ppi
    width_px = TRANSFORMER_BBOX_WIDTH_FEET * 12.0 * ppi
    half_w_pct = (width_px / 2.0 / img_w) * 100.0
    half_h_pct = (height_px / 2.0 / img_h) * 100.0

    cx = (top_coords['percentX'] + bottom_coords['percentX']) / 2.0
    cy = (top_coords['percentY'] + bottom_coords['percentY']) / 2.0

    left = cx - half_w_pct
    right = cx + half_w_pct
    top = cy - half_h_pct
    bottom = cy + half_h_pct

    # Expand if either keypoint falls outside
    for pt in (top_coords, bottom_coords):
        left = min(left, pt['percentX'])
        right = max(right, pt['percentX'])
        top = min(top, pt['percentY'])
        bottom = max(bottom, pt['percentY'])

    return {
        'left': max(0.0, left),
        'right': min(100.0, right),
        'top': max(0.0, top),
        'bottom': min(100.0, bottom),
    }


def calculate_street_light_bounding_box(
    upper_coords: Optional[Dict],
    lower_coords: Optional[Dict],
    ppi: float,
    img_w: int,
    img_h: int,
    drip_loop_coords: Optional[Dict] = None,
) -> Optional[Dict]:
    """Street light bbox from config. Supports upper, lower, and optional drip_loop keypoints."""
    if (not upper_coords and not lower_coords) or not ppi or ppi <= 0:
        return None

    height_px = STREET_LIGHT_BBOX_HEIGHT_FEET * 12.0 * ppi
    width_px = STREET_LIGHT_BBOX_WIDTH_FEET * 12.0 * ppi
    half_w_pct = (width_px / 2.0 / img_w) * 100.0
    half_h_pct = (height_px / 2.0 / img_h) * 100.0

    if upper_coords and lower_coords:
        cx = (upper_coords['percentX'] + lower_coords['percentX']) / 2.0
        cy = (upper_coords['percentY'] + lower_coords['percentY']) / 2.0
        left = cx - half_w_pct
        right = cx + half_w_pct
        top = cy - half_h_pct
        bottom = cy + half_h_pct
        pts = (upper_coords, lower_coords)
        if drip_loop_coords:
            pts = pts + (drip_loop_coords,)
    elif lower_coords:
        # Only lower (bottom bracket): anchor at lower, extend 8 ft upward (smaller percentY)
        cx = lower_coords['percentX']
        top = lower_coords['percentY'] - half_h_pct
        bottom = lower_coords['percentY'] + half_h_pct * 0.1  # small padding below
        left = cx - half_w_pct
        right = cx + half_w_pct
        pts = (lower_coords,) + ((drip_loop_coords,) if drip_loop_coords else ())
    else:
        # Only upper (top bracket): anchor at upper, extend 8 ft downward
        cx = upper_coords['percentX']
        top = upper_coords['percentY'] - half_h_pct * 0.1  # small padding above
        bottom = upper_coords['percentY'] + half_h_pct
        left = cx - half_w_pct
        right = cx + half_w_pct
        pts = (upper_coords,) + ((drip_loop_coords,) if drip_loop_coords else ())

    for pt in pts:
        left = min(left, pt['percentX'])
        right = max(right, pt['percentX'])
        top = min(top, pt['percentY'])
        bottom = max(bottom, pt['percentY'])

    return {
        'left': max(0.0, left),
        'right': min(100.0, right),
        'top': max(0.0, top),
        'bottom': min(100.0, bottom),
    }


def calculate_secondary_drip_loop_bounding_box(
    coords: Dict,
    ppi: float,
    img_w: int,
    img_h: int,
) -> Optional[Dict]:
    """Secondary drip loop bbox centered on keypoint. Size from config."""
    if not coords or not ppi or ppi <= 0:
        return None

    box_w_px = SECONDARY_DRIP_LOOP_BBOX_WIDTH_FEET * 12.0 * ppi
    box_h_px = SECONDARY_DRIP_LOOP_BBOX_HEIGHT_FEET * 12.0 * ppi

    cx_px = coords['percentX'] / 100.0 * img_w
    cy_px = coords['percentY'] / 100.0 * img_h

    top_px = cy_px - box_h_px / 2
    bottom_px = cy_px + box_h_px / 2

    return {
        'left': max(0.0, (cx_px - box_w_px / 2) / img_w * 100.0),
        'right': min(100.0, (cx_px + box_w_px / 2) / img_w * 100.0),
        'top': max(0.0, top_px / img_h * 100.0),
        'bottom': min(100.0, bottom_px / img_h * 100.0),
    }


def calculate_attachment_bounding_box(
    coords: Dict,
    ppi: float,
    img_w: int,
    img_h: int,
    height_feet: Optional[float] = None,
    width_feet: Optional[float] = None,
) -> Optional[Dict]:
    """Attachment bbox centered on keypoint.

    Default: comm (2ft H x 4ft W). For down_guy, pass height_feet=4, width_feet=2.
    """
    if not coords or not ppi or ppi <= 0:
        return None

    h_ft = height_feet if height_feet is not None else ATTACHMENT_BBOX_HEIGHT_FEET
    w_ft = width_feet if width_feet is not None else ATTACHMENT_BBOX_WIDTH_FEET
    height_px = h_ft * 12.0 * ppi
    width_px = w_ft * 12.0 * ppi
    half_w_pct = (width_px / 2.0 / img_w) * 100.0
    half_h_pct = (height_px / 2.0 / img_h) * 100.0

    cx = coords['percentX']
    cy = coords['percentY']

    return {
        'left': max(0.0, cx - half_w_pct),
        'right': min(100.0, cx + half_w_pct),
        'top': max(0.0, cy - half_h_pct),
        'bottom': min(100.0, cy + half_h_pct),
    }
