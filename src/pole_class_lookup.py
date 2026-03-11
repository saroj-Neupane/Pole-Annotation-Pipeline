"""
Pole class lookup from Ground Line Circumference (GLC) and Height.

Uses ANSI O5.1 standard dimensions for Douglas Fir and Southern Pine utility poles.

Usage:
    pole_class = get_pole_class(height_ft=35, glc_in=31.51)
    # Returns: '4'

    result = get_pole_class_with_confidence(height_ft=35, glc_in=31.51)
    # Returns: {
    #     'pole_class': '4',
    #     'diff': 0.0,
    #     'candidates': [('4', 0.0), ('3', 2.48), ('5', 2.51), ...]
    # }
"""

import json
from pathlib import Path
from typing import Optional


def load_pole_data():
    """Load pole class height JSON data into memory."""
    json_path = Path(__file__).parent.parent / "deploy" / "Pole_Class_Height.json"

    with open(json_path) as f:
        data = json.load(f)

    return data['poles']  # Return just the poles data, skip metadata


# Lazy load data on first call
_POLE_DATA = None
_METADATA = None

def _get_data():
    global _POLE_DATA
    if _POLE_DATA is None:
        json_path = Path(__file__).parent.parent / "deploy" / "Pole_Class_Height.json"
        with open(json_path) as f:
            data = json.load(f)
        _POLE_DATA = data['poles']
    return _POLE_DATA


def get_metadata():
    """Get metadata about the pole standards."""
    global _METADATA
    if _METADATA is None:
        json_path = Path(__file__).parent.parent / "deploy" / "Pole_Class_Height.json"
        with open(json_path) as f:
            data = json.load(f)
        _METADATA = data['metadata']
    return _METADATA


def get_pole_class(height_ft: int, glc_in: float, tolerance: float = 0.5) -> Optional[str]:
    """
    Get pole class from height and Ground Line Circumference.

    Args:
        height_ft: Pole length in feet (20, 25, 30, ..., 125)
        glc_in: Ground Line Circumference in inches
        tolerance: Tolerance for fuzzy matching (inches). Default 0.5 handles minor measurement variations.

    Returns:
        Pole class string ('1', '2', '3', '4', '5', '6', '7', 'H1', 'H2', etc.) or None if not found

    Example:
        >>> get_pole_class(height_ft=35, glc_in=31.51)
        '4'
    """
    data = _get_data()
    height_str = str(height_ft)

    if height_str not in data or 'Circum.' not in data[height_str]:
        return None

    circumferences = data[height_str]['Circum.']

    # Try exact match first
    for pole_class, value in circumferences.items():
        if abs(value - glc_in) < 0.01:  # Exact match (within floating point error)
            return pole_class

    # Always return the closest match (no tolerance limit)
    best_match = None
    best_diff = float('inf')

    for pole_class, value in circumferences.items():
        diff = abs(value - glc_in)
        if diff < best_diff:
            best_diff = diff
            best_match = pole_class

    return best_match


def get_pole_class_with_confidence(height_ft: int, glc_in: float, tolerance: float = 0.5) -> dict:
    """
    Get pole class with confidence score.

    Returns:
        {
            'pole_class': str or None,
            'diff': float (absolute difference from reference),
            'candidates': [(pole_class, diff), ...] (sorted by diff)
        }

    Example:
        >>> get_pole_class_with_confidence(height_ft=35, glc_in=31.51)
        {
            'pole_class': '4',
            'diff': 0.0,
            'candidates': [('4', 0.0), ('3', 1.48), ('5', 2.49), ...]
        }
    """
    data = _get_data()
    height_str = str(height_ft)

    if height_str not in data or 'Circum.' not in data[height_str]:
        return {'pole_class': None, 'diff': None, 'candidates': []}

    circumferences = data[height_str]['Circum.']

    # Calculate differences for all classes
    candidates = []
    for pole_class, value in circumferences.items():
        diff = abs(value - glc_in)
        candidates.append((pole_class, diff))

    # Sort by difference
    candidates.sort(key=lambda x: x[1])

    # Always use the closest match (no tolerance limit)
    best_class = None
    best_diff = None
    if candidates:
        best_class = candidates[0][0]
        best_diff = candidates[0][1]

    return {
        'pole_class': best_class,
        'diff': best_diff,
        'candidates': candidates
    }


if __name__ == '__main__':
    # Show metadata
    metadata = get_metadata()
    print("=" * 70)
    print("POLE CLASS LOOKUP - ANSI O5.1 Standard")
    print("=" * 70)
    print(f"\nSource: {metadata['source']}")
    print(f"Measurement Point: {metadata['measurement_point']}")
    print(f"Units: {metadata['units']}")
    print(f"Total Entries: {metadata['total_entries']}")
    print(f"Pole Classes: {', '.join(metadata['pole_classes'])}")
    print(f"Heights (ft): {', '.join(map(str, metadata['heights_ft']))}\n")

    # Test examples
    print("=" * 70)
    print("TEST CASES")
    print("=" * 70)

    tests = [
        (35, 31.51, "Class 4"),
        (50, 42.0, "Class 2 (approx)"),
        (75, 55.51, "Class H1"),
        (100, 65.5, "Class H2"),
    ]

    for height, glc, description in tests:
        result = get_pole_class(height, glc)
        full = get_pole_class_with_confidence(height, glc)
        print(f"\nHeight: {height} ft, GLC: {glc} in ({description})")
        print(f"  Matched Class: {result}")
        if full['candidates']:
            top3 = full['candidates'][:3]
            print(f"  Top 3 matches: {top3}")
