"""
Height and coordinate calculations for ruler and pole measurements.

Provides utilities for:
- Linear interpolation of coordinates at unmeasured heights
- PPI (pixels per inch) calculation from height measurements
- Ground and ruler top coordinate calculation
"""

from typing import Dict, Optional


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
