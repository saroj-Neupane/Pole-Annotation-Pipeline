"""
Calculate and update Pole Class and Pole Height based on calibration data.

Workflow:
1. After calibration, get pole_top_height from photo calibration data
2. Calculate pole_length = ceil(pole_top_height + 6) to nearest 5-ft increment
3. Get GLC ("Measured Groundline Circumference") from node attributes
4. Use pole_length + GLC to lookup pole_class and pole_height
5. Update node attributes with results
"""

import math
from typing import Optional, Dict, Any, Tuple

try:
    from src.pole_class_lookup import get_pole_class, get_pole_class_with_confidence
except ImportError:
    from pole_class_lookup import get_pole_class, get_pole_class_with_confidence


def round_to_nearest_5(value: float) -> int:
    """Round up to next 5-foot increment (slightly higher).

    Examples:
        28.5 → 30 (next standard up from 28.5)
        29.9 → 30
        30.0 → 30
        30.1 → 35
        34.9 → 35
    """
    return int(math.ceil(value / 5) * 5)


def calculate_pole_length_from_top(pole_top_height_ft: float) -> int:
    """
    Calculate pole length from pole top height.

    Pole top is at height (pole_length - 6 feet), so:
    pole_length = pole_top_height + 6 feet
    Then round to nearest 5-ft standard (20, 25, 30, ..., 125)

    Args:
        pole_top_height_ft: Height of pole top in feet (from calibration)

    Returns:
        Pole length in feet (20, 25, 30, 35, ..., 125)
    """
    calculated_length = pole_top_height_ft + 6.0
    standard_length = round_to_nearest_5(calculated_length)

    # Clamp to valid range
    standard_length = max(20, min(125, standard_length))

    return standard_length


def get_pole_class_and_height(
    pole_length_ft: int,
    glc_in: float,
    tolerance: float = 0.5
) -> Dict[str, Any]:
    """
    Look up pole class and retrieve corresponding standard height.

    Args:
        pole_length_ft: Pole length in feet
        glc_in: Ground Line Circumference in inches
        tolerance: Tolerance for fuzzy matching (inches)

    Returns:
        {
            'pole_class': str or None,
            'pole_height': str or None,
            'glc_diff': float or None,
            'confidence': dict (from get_pole_class_with_confidence)
        }
    """
    # Get pole class with confidence
    confidence_result = get_pole_class_with_confidence(
        height_ft=pole_length_ft,
        glc_in=glc_in,
        tolerance=tolerance
    )

    pole_class = confidence_result.get('pole_class')
    glc_diff = confidence_result.get('diff')

    # Get standard pole height from JSON
    # For now, we'll return the pole_length as the pole_height
    # (since pole_height in standards is same as pole_length)
    pole_height = str(pole_length_ft) if pole_class else None

    return {
        'pole_class': pole_class,
        'pole_height': pole_height,
        'glc_diff': glc_diff,
        'confidence': confidence_result
    }


def prepare_pole_attributes(
    pole_top_height_ft: float,
    glc_in: float,
    tolerance: float = 0.5
) -> Optional[Dict[str, str]]:
    """
    Prepare pole class and height attributes for upload.

    Args:
        pole_top_height_ft: Height of pole top from calibration (feet)
        glc_in: Ground Line Circumference from node attributes (inches)
        tolerance: Tolerance for GLC matching (inches)

    Returns:
        {
            'Pole Class': 'X',
            'Pole Height': 'XXX',
            'pole_length_calculated': int,
            'glc_diff': float
        }
        or None if lookup failed
    """
    # Calculate pole length from top
    pole_length_ft = calculate_pole_length_from_top(pole_top_height_ft)

    # Look up pole class and height
    result = get_pole_class_and_height(pole_length_ft, glc_in, tolerance)

    if result['pole_class'] is None:
        return None

    return {
        'Pole Class': result['pole_class'],
        'Pole Height': result['pole_height'],
        'pole_length_calculated': pole_length_ft,
        'glc_diff': result['glc_diff'],
        'lookup_confidence': result['confidence']
    }


if __name__ == '__main__':
    # Test examples
    print("=" * 70)
    print("POLE CLASS CALCULATOR - Test Examples")
    print("=" * 70)

    test_cases = [
        (28.5, 31.51, "Pole top 28.5 ft, GLC 31.51\" → expects 35 ft pole, Class 4"),
        (44.0, 42.0, "Pole top 44.0 ft, GLC 42.0\" → expects 50 ft pole, Class 2"),
        (69.0, 55.51, "Pole top 69.0 ft, GLC 55.51\" → expects 75 ft pole, Class H1"),
        (64.0, 66.51, "Pole top 64.0 ft, GLC 66.51\" → expects 70 ft pole, Class H5"),
    ]

    for pole_top, glc, desc in test_cases:
        print(f"\n{desc}")
        pole_len = calculate_pole_length_from_top(pole_top)
        print(f"  Calculated pole length: {pole_top} + 6 = {pole_top + 6:.1f} → {pole_len} ft")

        attrs = prepare_pole_attributes(pole_top, glc)
        if attrs:
            print(f"  Pole Class: {attrs['Pole Class']}")
            print(f"  Pole Height: {attrs['Pole Height']} ft")
            print(f"  GLC difference: {attrs['glc_diff']:.2f}\"")
            conf = attrs['lookup_confidence']
            if conf['candidates']:
                top3 = conf['candidates'][:3]
                print(f"  Top candidates: {top3}")
        else:
            print(f"  ❌ No match found")
