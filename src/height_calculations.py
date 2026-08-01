"""
Height and coordinate calculations for ruler and pole measurements.

Provides utilities for:
- Linear interpolation of coordinates at unmeasured heights
- PPI (pixels per inch) calculation from height measurements
- Ground and ruler top coordinate calculation
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union


def linear_interpolate_height(target_height: float, height_measurements: Dict) -> Optional[Dict]:
    """Use linear interpolation to find coordinates at a specific height using available measurements.

    Args:
        target_height: Target height in feet to calculate coordinates for
        height_measurements: Dict with height measurements (keys are heights in feet)

    Returns:
        Dict with percentX, percentY or None if calculation not possible
    """
    # Get all available height measurements (excluding pole_top and equipment)
    available_heights = []
    for key, measurement in height_measurements.items():
        if isinstance(key, (int, float)) and key > 0:
            available_heights.append((key, measurement['percentX'], measurement['percentY']))

    if len(available_heights) < 2:
        return None

    # Sort by height
    available_heights.sort(key=lambda x: x[0])

    # Find the two closest heights for interpolation
    if target_height <= available_heights[0][0]:
        # Target is below the lowest measurement - extrapolate downward
        h1, x1, y1 = available_heights[0]
        h2, x2, y2 = available_heights[1]
    elif target_height >= available_heights[-1][0]:
        # Target is above the highest measurement - extrapolate upward
        h1, x1, y1 = available_heights[-2]
        h2, x2, y2 = available_heights[-1]
    else:
        # Target is between measurements - find the two closest ones
        for i in range(len(available_heights) - 1):
            h1, x1, y1 = available_heights[i]
            h2, x2, y2 = available_heights[i + 1]
            if h1 <= target_height <= h2:
                break

    # Linear interpolation
    if h2 != h1:
        interpolated_y = y1 + (target_height - h1) * (y2 - y1) / (h2 - h1)
        interpolated_x = x1 + (target_height - h1) * (x2 - x1) / (h2 - h1)
        return {'percentX': interpolated_x, 'percentY': interpolated_y}

    return None


def calculate_ground_coordinates(height_measurements: Dict) -> Optional[Dict]:
    """Calculate 0 ft coordinates using linear interpolation."""
    return linear_interpolate_height(0.0, height_measurements)


def calculate_ruler_top_coordinates(height_measurements: Dict) -> Optional[Dict]:
    """Calculate ruler_top coordinates at 17.0 ft using linear interpolation."""
    return linear_interpolate_height(17.0, height_measurements)


def calculate_ppi_from_measurements(height_measurements: Dict, image_height_px: float) -> Optional[float]:
    """Calculate PPI (pixels per inch) using available height measurements.

    Calculates PPI by averaging multiple consecutive height measurement pairs for better accuracy.
    Uses pairs: 2.5-6.5, 6.5-10.5, 10.5-14.5, 14.5-16.5 feet.
    This is more accurate than using a single pair because it averages out measurement errors.

    Args:
        height_measurements: Dict with height measurements (keys are heights in feet, values are dicts with percentX, percentY)
        image_height_px: Height of the image in pixels

    Returns:
        PPI value (pixels per inch) or None if calculation is not possible
    """
    # Get all available height measurements (excluding pole_top and equipment)
    available_heights = []
    for key, measurement in height_measurements.items():
        if isinstance(key, (int, float)) and key > 0:
            if isinstance(measurement, dict) and 'percentY' in measurement:
                available_heights.append((key, measurement['percentY']))
            elif isinstance(measurement, (list, tuple)) and len(measurement) >= 2:
                # Handle tuple format: (percentX, percentY)
                available_heights.append((key, measurement[1]))

    if len(available_heights) < 2:
        return None

    # Sort by height
    available_heights.sort(key=lambda x: x[0])

    # Calculate PPI for each consecutive pair and average them
    ppi_values = []
    for i in range(len(available_heights) - 1):
        h1, y1_percent = available_heights[i]
        h2, y2_percent = available_heights[i + 1]

        # Calculate height difference in inches
        height_diff_feet = h2 - h1
        height_diff_inches = height_diff_feet * 12.0

        if height_diff_inches <= 0:
            continue

        # Calculate pixel distance between the two points
        y_diff_percent = abs(y2_percent - y1_percent)
        pixel_distance = y_diff_percent / 100.0 * image_height_px

        if pixel_distance <= 0:
            continue

        # Calculate PPI for this pair
        ppi_pair = pixel_distance / height_diff_inches
        ppi_values.append(ppi_pair)

    if len(ppi_values) == 0:
        return None

    # Return average PPI
    return sum(ppi_values) / len(ppi_values)


# --------------------------------------------------------------------------- #
# Projection model (single source of truth for pixel -> height in inches)
#
# A single PPI scalar assumes the percentY -> height curve is a straight line, but
# the ruler is a camera projection, so the true curve is projective (rational). The
# canonical fit lives in src/ruler_height_model.py; the helpers below are the ONE
# place that loads a photo's ruler anchors and turns a pixel position (or a pair of
# them) into a physically-consistent height in inches. Both the wire tracer and the
# keypoint/PCK metrics go through here, so height is computed identically everywhere.
# --------------------------------------------------------------------------- #

# The 5 real Katapult ruler anchors (feet). 0.0 (ground) / 17.0 (ruler top) are
# extrapolated by extract_height and excluded — the projective fit is validated on
# exactly these (matches the SDK's iter_photo_calibration_anchors).
RULER_ANCHOR_FEET = (2.5, 6.5, 10.5, 14.5, 16.5)

_HEIGHT_FIT_CACHE: Dict[str, object] = {}


def load_height_anchors(label_path: Union[str, Path]) -> List[Tuple[float, float]]:
    """Parse the 5 ruler anchors from a ``*_location.txt`` as ``(percentY, inches)``.

    Anchor lines are ``height_ft, percentX, percentY``; only the real
    :data:`RULER_ANCHOR_FEET` rows are kept. Returns ``[]`` when none are present.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    anchors: List[Tuple[float, float]] = []
    for line in label_path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            ft = float(parts[0])
            py = float(parts[2])
        except ValueError:
            continue
        if ft in RULER_ANCHOR_FEET:
            anchors.append((py, ft * 12.0))  # (percentY 0-100, inches)
    return anchors


def fit_height_from_location_file(label_path: Union[str, Path]):
    """Cached projective ``percentY -> inches`` :class:`HeightFit` for a photo, or None.

    Central entry point: parses the ruler anchors and fits the canonical projective
    model (:func:`src.ruler_height_model.fit_photo_height`). Returns ``None`` when the
    photo has no fittable ruler (callers fall back to a PPI scalar / percent band).
    """
    from src.ruler_height_model import fit_photo_height

    key = str(label_path)
    if key in _HEIGHT_FIT_CACHE:
        return _HEIGHT_FIT_CACHE[key]
    anchors = load_height_anchors(label_path)
    fit = fit_photo_height(anchors) if anchors else None
    _HEIGHT_FIT_CACHE[key] = fit
    return fit


def vertical_error_inches(
    fit,
    y1_px: float,
    y2_px: float,
    image_height_px: float,
    ppi: Optional[float] = None,
) -> Optional[float]:
    """Vertical distance in inches between two pixel-Y positions.

    Preferred path: the projection model — converts each pixel-Y to percentY
    (``y / image_height_px * 100``), reads its height in inches via ``fit``, and
    returns ``|h1 - h2|``. This captures the within-photo perspective nonlinearity a
    single PPI scalar cannot. Falls back to ``|y1 - y2| / ppi`` when there is no fit
    (or the fit is non-physical at a point), and to raw pixels as a last resort.
    """
    if fit is not None and image_height_px and image_height_px > 0:
        from src.ruler_height_model import height_in_at

        h1 = height_in_at(fit, y1_px / image_height_px * 100.0)
        h2 = height_in_at(fit, y2_px / image_height_px * 100.0)
        if h1 is not None and h2 is not None:
            return abs(h1 - h2)
    if ppi and ppi > 0:
        return abs(y1_px - y2_px) / ppi
    return abs(y1_px - y2_px)
