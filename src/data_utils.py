"""
Data processing utilities for dataset preparation.
"""

import numpy as np
import cv2
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Callable, Any, Iterable
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


def _parallel_map(items: Iterable, fn: Callable, workers: int, desc: str = None, verbose: bool = False) -> List[Any]:
    """Process items in parallel with fn(item). Returns list of results."""
    items = list(items)
    if workers <= 1 or len(items) == 0:
        return [fn(x) for x in tqdm(items, desc=desc, disable=not verbose)]
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_idx = {ex.submit(fn, x): i for i, x in enumerate(items)}
        for future in tqdm(as_completed(future_to_idx), total=len(items), desc=desc, disable=not verbose):
            results[future_to_idx[future]] = future.result()
    return results


from .config import (
    DATASETS_DIR,
    FROZEN_MANIFEST_FILENAME,
    SPLIT_MANIFEST_PATH,
    SPLIT_MANIFEST_RANDOM_STATE,
    EQUIPMENT_CLASSES,
    EQUIPMENT_CLASS_NAMES,
    ATTACHMENT_CLASSES,
    ATTACHMENT_CLASS_NAMES,
    RISER_NUM_KEYPOINTS,
    RISER_KEYPOINT_NAMES,
    TRANSFORMER_NUM_KEYPOINTS,
    TRANSFORMER_KEYPOINT_NAMES,
    STREET_LIGHT_NUM_KEYPOINTS,
    STREET_LIGHT_KEYPOINT_NAMES,
    SECONDARY_DRIP_LOOP_NUM_KEYPOINTS,
    SECONDARY_DRIP_LOOP_KEYPOINT_NAMES,
    RISER_BBOX_HEIGHT_FEET,
    RISER_BBOX_WIDTH_FEET,
    KEYPOINT_NAMES,
)


def load_frozen_manifest(dataset_dir: Path, strict: bool = False) -> Optional[Dict]:
    """
    Load frozen manifest for a dataset.

    Args:
        dataset_dir: Path to dataset directory
        strict: If True, raise FileNotFoundError when manifest doesn't exist.
                If False, return None when not found.

    Returns:
        Manifest dictionary, or None if not found (when strict=False)
    """
    manifest_path = dataset_dir / FROZEN_MANIFEST_FILENAME
    if not manifest_path.exists():
        if strict:
            raise FileNotFoundError(
                f"Frozen manifest not found: {manifest_path}\n"
                f"Please run: python scripts/freeze_validation_test_sets.py --dataset {dataset_dir.name}"
            )
        return None
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except Exception as e:
        if strict:
            raise
        print(f"❌ Error loading manifest: {e}")
        return None


# -----------------------------------------------------------------------------
# Master split manifest (single source of truth for train/val/test across datasets)
# -----------------------------------------------------------------------------


def create_split_manifest(
    pole_photos_dir: Path,
    pole_labels_dir: Path,
    midspan_photos_dir: Path,
    midspan_labels_dir: Path,
    output_path: Path = SPLIT_MANIFEST_PATH,
    random_state: int = SPLIT_MANIFEST_RANDOM_STATE,
) -> Dict:
    """
    Create master split manifest. Pole uses equipment criteria (pole bbox, is_photo_labeled).
    Midspan uses has-location-file. Each domain split 80/10/10 independently.
    """
    from datetime import datetime

    pole_photos_dir = Path(pole_photos_dir)
    pole_labels_dir = Path(pole_labels_dir)
    midspan_photos_dir = Path(midspan_photos_dir)
    midspan_labels_dir = Path(midspan_labels_dir)

    from PIL import Image
    pole_stems = []
    for photo_path in sorted(pole_photos_dir.glob("*.jpg")):
        label_path = pole_labels_dir / f"{photo_path.stem}_location.txt"
        if not label_path.exists():
            continue
        if not is_photo_labeled(label_path):
            continue
        try:
            with Image.open(photo_path) as im:
                w, h = im.size
        except Exception:
            continue
        if load_pole_bbox_from_location_file(label_path, w, h) is None:
            continue
        pole_stems.append(photo_path.stem)

    midspan_stems = [
        p.stem for p in sorted(midspan_photos_dir.glob("*.jpg"))
        if (midspan_labels_dir / f"{p.stem}_location.txt").exists()
    ]

    def _split(stems: List[str]) -> Tuple[List[str], List[str], List[str]]:
        if not stems:
            return [], [], []
        train, temp = train_test_split(stems, test_size=0.2, random_state=random_state)
        val, test = train_test_split(temp, test_size=0.5, random_state=random_state)
        return train, val, test

    pole_train, pole_val, pole_test = _split(pole_stems)
    midspan_train, midspan_val, midspan_test = _split(midspan_stems)

    manifest = {
        "version": 1,
        "created": datetime.now().isoformat(),
        "random_state": random_state,
        "pole": {"train": pole_train, "val": pole_val, "test": pole_test},
        "midspan": {"train": midspan_train, "val": midspan_val, "test": midspan_test},
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_split_manifest(path: Path = SPLIT_MANIFEST_PATH) -> Optional[Dict]:
    """Load master split manifest. Returns None if not found."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def update_split_manifest(
    pole_photos_dir: Path,
    pole_labels_dir: Path,
    midspan_photos_dir: Path,
    midspan_labels_dir: Path,
    output_path: Path = SPLIT_MANIFEST_PATH,
    manifest: Optional[Dict] = None,
) -> Dict:
    """
    Update split manifest by freezing val/test and adding new eligible samples to train.
    Stems already in manifest (any split) are skipped. New stems from data/ go to train only.
    """
    from datetime import datetime
    from PIL import Image

    pole_photos_dir = Path(pole_photos_dir)
    pole_labels_dir = Path(pole_labels_dir)
    midspan_photos_dir = Path(midspan_photos_dir)
    midspan_labels_dir = Path(midspan_labels_dir)
    output_path = Path(output_path)

    manifest = manifest or load_split_manifest(output_path)
    if not manifest:
        raise FileNotFoundError(
            f"split_manifest.json not found at {output_path}. "
            "Run prepare_dataset.py once without --freeze-splits to create it."
        )

    # Frozen stems: all that are already in manifest (val/test stay unchanged)
    pole_existing = set()
    for split in ["train", "val", "test"]:
        pole_existing.update(manifest.get("pole", {}).get(split, []))
    midspan_existing = set()
    for split in ["train", "val", "test"]:
        midspan_existing.update(manifest.get("midspan", {}).get(split, []))

    # Collect eligible pole stems (same logic as create_split_manifest)
    pole_stems = []
    for photo_path in sorted(pole_photos_dir.glob("*.jpg")):
        label_path = pole_labels_dir / f"{photo_path.stem}_location.txt"
        if not label_path.exists():
            continue
        if not is_photo_labeled(label_path):
            continue
        try:
            with Image.open(photo_path) as im:
                w, h = im.size
        except Exception:
            continue
        if load_pole_bbox_from_location_file(label_path, w, h) is None:
            continue
        pole_stems.append(photo_path.stem)

    midspan_stems = [
        p.stem for p in sorted(midspan_photos_dir.glob("*.jpg"))
        if (midspan_labels_dir / f"{p.stem}_location.txt").exists()
    ]

    # New stems = eligible in data but not in manifest
    pole_new = [s for s in pole_stems if s not in pole_existing]
    midspan_new = [s for s in midspan_stems if s not in midspan_existing]

    # Append new stems to train; val and test unchanged
    pole_train = list(manifest.get("pole", {}).get("train", [])) + pole_new
    pole_val = manifest.get("pole", {}).get("val", [])
    pole_test = manifest.get("pole", {}).get("test", [])
    midspan_train = list(manifest.get("midspan", {}).get("train", [])) + midspan_new
    midspan_val = manifest.get("midspan", {}).get("val", [])
    midspan_test = manifest.get("midspan", {}).get("test", [])

    updated = {
        "version": manifest.get("version", 1),
        "created": manifest.get("created", ""),
        "updated": datetime.now().isoformat(),
        "random_state": manifest.get("random_state", SPLIT_MANIFEST_RANDOM_STATE),
        "pole": {"train": pole_train, "val": pole_val, "test": pole_test},
        "midspan": {"train": midspan_train, "val": midspan_val, "test": midspan_test},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(updated, f, indent=2)
    return updated


def get_pole_split_map(manifest: Dict) -> Dict[str, str]:
    """Return {stem: 'train'|'val'|'test'} for pole photos."""
    m = manifest.get("pole", {})
    out = {}
    for split in ["train", "val", "test"]:
        for stem in m.get(split, []):
            out[stem] = split
    return out


def get_midspan_split_map(manifest: Dict) -> Dict[str, str]:
    """Return {stem: 'train'|'val'|'test'} for midspan photos."""
    m = manifest.get("midspan", {})
    out = {}
    for split in ["train", "val", "test"]:
        for stem in m.get(split, []):
            out[stem] = split
    return out


def get_manifest_test_stems(manifest: Dict, domain: str = "pole") -> set:
    """Return set of test stems for E2E evaluation (never-seen data)."""
    return set(manifest.get(domain, {}).get("test", []))


def get_manifest_val_stems(manifest: Dict, domain: str = "pole") -> set:
    """Return set of val stems for inference demos (random / end-to-end prediction)."""
    return set(manifest.get(domain, {}).get("val", []))


def get_e2e_test_images(domain: str = "equipment") -> List[Path]:
    """
    Canonical helper for E2E test images (manifest-filtered, never-seen during training).
    If E2E_USE_TEST_SPLIT_ONLY, returns only test-split images; raises if manifest or
    test stems are missing. Use this everywhere for consistent unseen evaluation.
    """
    from .config import (
        E2E_USE_TEST_SPLIT_ONLY,
        EQUIPMENT_E2E_IMAGES_DIR,
        ATTACHMENT_E2E_IMAGES_DIR,
        EQUIPMENT_DATASET_DIR,
        ATTACHMENT_DATASET_DIR,
    )

    images_dir = Path(EQUIPMENT_E2E_IMAGES_DIR) if domain == "equipment" else Path(ATTACHMENT_E2E_IMAGES_DIR)
    all_images = sorted(images_dir.glob("*.jpg"))
    if not E2E_USE_TEST_SPLIT_ONLY:
        return all_images

    manifest = load_split_manifest()
    if manifest:
        test_stems = get_manifest_test_stems(manifest, "pole")
    else:
        test_dir = (EQUIPMENT_DATASET_DIR if domain == "equipment" else ATTACHMENT_DATASET_DIR) / "images" / "test"
        test_stems = {p.stem for p in test_dir.glob("*.jpg")} if test_dir.exists() else set()

    if not test_stems:
        raise RuntimeError(
            f"E2E test-only mode is enabled but no test stems were found for '{domain}'. "
            "Ensure split manifest exists (preferred) or dataset test split is prepared."
        )
    filtered = [p for p in all_images if p.stem in test_stems]
    if not filtered:
        raise RuntimeError(
            f"E2E test-only mode is enabled for '{domain}', but no files in {images_dir} matched test stems. "
            "Check image source dir and split manifest consistency."
        )
    return filtered


def get_e2e_val_images(domain: str = "equipment") -> List[Path]:
    """
    Return val-split images for inference demos (random prediction, end-to-end).
    Uses manifest val stems. Raises if manifest or val stems missing.
    """
    from .config import EQUIPMENT_E2E_IMAGES_DIR, ATTACHMENT_E2E_IMAGES_DIR

    images_dir = Path(EQUIPMENT_E2E_IMAGES_DIR) if domain == "equipment" else Path(ATTACHMENT_E2E_IMAGES_DIR)
    all_images = sorted(images_dir.glob("*.jpg"))

    manifest = load_split_manifest()
    if not manifest:
        raise RuntimeError("Split manifest not found. Generate manifest for val-based inference demos.")
    val_stems = get_manifest_val_stems(manifest, "pole")
    if not val_stems:
        raise RuntimeError("No pole val stems in split manifest.")
    filtered = [p for p in all_images if p.stem in val_stems]
    if not filtered:
        raise RuntimeError(
            f"No files in {images_dir} matched manifest val stems. Check manifest and image dir."
        )
    return filtered


def parse_label_file(label_path: Path) -> Tuple[Optional[List[float]], Optional[List[float]], Dict[str, Tuple[float, float]], Optional[float]]:
    """
    Parse label file to extract pole bbox, ruler bbox, keypoints, and PPI.
    
    Args:
        label_path: Path to the label file
        
    Returns:
        Tuple of (pole_bbox, ruler_bbox, keypoints_dict, ppi)
        - pole_bbox: [left, right, top, bottom] in percent coordinates, or None
        - ruler_bbox: [left, right, top, bottom] in percent coordinates, or None
        - keypoints: Dict mapping height strings to (x, y) tuples in percent coordinates
        - ppi: Pixels per inch value, or None
    """
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    keypoints = {}
    pole_bbox = None
    ruler_bbox = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('#') and ',' in line and 'Left,Right,Top,Bottom' not in line:
            try:
                bbox_data = line.replace('#', '').strip().split(',')
                if len(bbox_data) >= 7:
                    # Check previous lines for type
                    for j in range(i-1, -1, -1):
                        prev_line = lines[j].strip()
                        if prev_line and not prev_line.startswith('#'):
                            break
                        if 'Pole bounding box' in prev_line:
                            pole_bbox = [float(x) for x in bbox_data[:4]]  # left, right, top, bottom
                            break
                        elif 'Ruler bounding box' in prev_line:
                            ruler_bbox = [float(x) for x in bbox_data[:4]]
                            break
            except Exception:
                pass
        elif not line.startswith('#') and ',' in line:
            parts = line.strip().split(',')
            if len(parts) == 3 and parts[0] and parts[1] and parts[2]:
                try:
                    height = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    keypoints[height] = (x, y)
                except Exception:
                    pass
    
    # Parse PPI from location file
    ppi = None
    for line in lines:
        line = line.strip()
        if line.startswith('# PPI='):
            try:
                ppi_str = line.split('=')[1].strip()
                ppi = float(ppi_str)
            except Exception:
                pass
    
    return pole_bbox, ruler_bbox, keypoints, ppi


def keypoint_in_bbox(kp_x_percent: float, kp_y_percent: float, 
                     bbox_left: float, bbox_right: float, 
                     bbox_top: float, bbox_bottom: float) -> bool:
    """
    Check if keypoint (in percent coordinates) is within bbox (in percent coordinates).
    
    Args:
        kp_x_percent: Keypoint x coordinate in percent
        kp_y_percent: Keypoint y coordinate in percent
        bbox_left: Bounding box left edge in percent
        bbox_right: Bounding box right edge in percent
        bbox_top: Bounding box top edge in percent
        bbox_bottom: Bounding box bottom edge in percent
        
    Returns:
        True if keypoint is within bbox, False otherwise
    """
    return (bbox_left <= kp_x_percent <= bbox_right and 
            bbox_top <= kp_y_percent <= bbox_bottom)


def check_dataset_complete(dataset_dir: Path, photo_files: Optional[List[Path]] = None,
                          train_files: Optional[List[Path]] = None,
                          val_files: Optional[List[Path]] = None,
                          test_files: Optional[List[Path]] = None,
                          manifest: Optional[Dict] = None,
                          domain: str = "pole") -> bool:
    """
    Check if all expected files exist in the dataset.
    
    This function supports three calling patterns:
    1. With photo_files, train_files, val_files, test_files: Checks exact counts
    2. With manifest: Verifies all manifest stems exist in dataset (catches new train samples)
    3. Without files/manifest: Checks that train/val/test splits have matching images/labels
    
    Args:
        dataset_dir: Path to dataset directory
        photo_files: Optional list of all photo files (for exact count checking)
        train_files: Optional list of training files
        val_files: Optional list of validation files
        test_files: Optional list of test files
        manifest: Optional split manifest; when provided, verifies all stems exist in dataset
        domain: 'pole' or 'midspan' when manifest is provided
        
    Returns:
        True if dataset is complete, False otherwise
    """
    # Check if dataset directory exists
    if not dataset_dir.exists():
        return False

    # If manifest provided, verify all stems exist (supports incremental manifest updates)
    if manifest is not None:
        m = manifest.get(domain, {})
        for split in ["train", "val", "test"]:
            stems = set(m.get(split, []))
            img_dir = dataset_dir / "images" / split
            lbl_dir = dataset_dir / "labels" / split
            for stem in stems:
                if not (img_dir / f"{stem}.jpg").exists() or not (lbl_dir / f"{stem}.txt").exists():
                    return False
        return True
    
    # Check images and labels for train, val, and test
    train_images = len(list((dataset_dir / "images" / "train").glob("*.jpg")))
    train_labels = len(list((dataset_dir / "labels" / "train").glob("*.txt")))
    val_images = len(list((dataset_dir / "images" / "val").glob("*.jpg")))
    val_labels = len(list((dataset_dir / "labels" / "val").glob("*.txt")))
    test_images = len(list((dataset_dir / "images" / "test").glob("*.jpg")))
    test_labels = len(list((dataset_dir / "labels" / "test").glob("*.txt")))
    
    # If file lists provided, check exact counts
    if train_files is not None and val_files is not None and test_files is not None:
        train_count = len(train_files)
        val_count = len(val_files)
        test_count = len(test_files)
        
        return (train_images >= train_count * 0.5 and train_labels >= train_count * 0.5 and
                val_images >= val_count * 0.5 and val_labels >= val_count * 0.5 and
                test_images >= test_count * 0.5 and test_labels >= test_count * 0.5)
    else:
        # Dataset is complete if we have a reasonable number of files in each split
        # (at least 10 files per split, or if we have files, check that images and labels match)
        has_train = train_images > 0 and train_labels > 0 and train_images == train_labels
        has_val = val_images > 0 and val_labels > 0 and val_images == val_labels
        has_test = test_images > 0 and test_labels > 0 and test_images == test_labels
        
        # Consider complete if we have files in all three splits with matching images/labels
        return has_train and has_val and has_test


def is_photo_labeled(label_path: Path) -> bool:
    """
    Check if a photo has been labeled (has equipment or attachment markers).
    
    A photo is considered labeled if the location file contains equipment
    (riser, transformer, street_light) or attachments (comm, down_guy).
    Photos with only pole_top and height measurements are considered unlabeled.
    
    Args:
        label_path: Path to *_location.txt file
        
    Returns:
        True if photo has equipment or attachment labels, False otherwise
    """
    with open(label_path, 'r') as f:
        content = f.read()
    
    # Check for equipment markers
    equipment_markers = ['riser', 'transformer', 'street_light', 'secondary_drip_loop']
    for marker in equipment_markers:
        if f'\n{marker}' in content or content.startswith(marker):
            return True
    
    # Check for attachment markers
    attachment_markers = ['comm', 'down_guy', 'primary', 'secondary', 'neutral', 'guy']
    for marker in attachment_markers:
        if f'\n{marker}' in content or content.startswith(marker):
            return True
    
    return False


def _pct_to_pixels(pct_x: float, pct_y: float, img_width: int, img_height: int) -> Tuple[float, float]:
    """Convert percentage coordinates (0-100) to pixel coordinates."""
    return (pct_x / 100.0) * img_width, (pct_y / 100.0) * img_height


def _load_bbox_from_location_file(location_path: Path, img_width: int, img_height: int,
                                   section_marker: str) -> Optional[Tuple[int, int, int, int]]:
    """Load bbox from location file for a given section marker (e.g., 'Pole bounding box')."""
    if not location_path.exists():
        return None

    with open(location_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if section_marker in line and 'percentage coordinates' in line.lower():
            if i + 1 < len(lines):
                data_line = lines[i + 1].strip()
                if data_line.startswith('#'):
                    try:
                        bbox_data = data_line.replace('#', '').strip().split(',')
                        if len(bbox_data) >= 4:
                            left_pct, right_pct = float(bbox_data[0]), float(bbox_data[1])
                            top_pct, bottom_pct = float(bbox_data[2]), float(bbox_data[3])
                            x1 = int(_pct_to_pixels(left_pct, 0, img_width, img_height)[0])
                            y1 = int(_pct_to_pixels(0, top_pct, img_width, img_height)[1])
                            x2 = int(_pct_to_pixels(right_pct, 0, img_width, img_height)[0])
                            y2 = int(_pct_to_pixels(0, bottom_pct, img_width, img_height)[1])
                            return (x1, y1, x2, y2)
                    except (ValueError, IndexError):
                        continue
    return None


def load_yolo_label(label_path: Path) -> Optional[List[float]]:
    """Load YOLO format label file."""
    if not label_path.exists():
        return None
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
    if not lines:
        return None
    parts = lines[0].split()
    if len(parts) < 5:
        return None
    return [float(x) for x in parts[:5]]


def load_ruler_marking_keypoints(label_path: Path, crop_width: float, crop_height: float) -> Optional[Dict[float, Dict[str, float]]]:
    """Load ruler marking keypoints from YOLO pose format."""
    if not label_path.exists():
        return None
    
    keypoint_heights = [2.5, 6.5, 10.5, 14.5, 16.5]
    
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
    
    if not lines:
        return None
    
    parts = lines[0].split()
    if len(parts) < 11:
        return None
    
    keypoints = {}
    num_keypoints = (len(parts) - 5) // 3
    
    for i in range(min(num_keypoints, len(keypoint_heights))):
        kp_idx = 5 + i * 3
        if kp_idx + 2 < len(parts):
            kp_x_norm = float(parts[kp_idx])
            kp_y_norm = float(parts[kp_idx + 1])
            kp_v = float(parts[kp_idx + 2])
            
            if kp_v > 0:
                height = keypoint_heights[i]
                keypoints[height] = {
                    'x': kp_x_norm * crop_width,
                    'y': kp_y_norm * crop_height
                }
    
    return keypoints if keypoints else None


def load_pole_top_keypoint(label_path: Path, crop_width: float, crop_height: float) -> Optional[Dict[str, float]]:
    """Load pole top keypoint from YOLO pose format."""
    if not label_path.exists():
        return None
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
    if not lines:
        return None
    parts = lines[0].split()
    if len(parts) >= 8:
        kp_x_norm = float(parts[5])
        kp_y_norm = float(parts[6])
        vis = float(parts[7])
        if vis > 0:
            return {
                'x': kp_x_norm * crop_width,
                'y': kp_y_norm * crop_height
            }
    return None


def load_pole_top_from_location_file(location_path: Path, img_width: int, img_height: int) -> Optional[Dict[str, float]]:
    """Load pole top keypoint from location file (percentage coordinates -> global pixel coordinates)."""
    if not location_path.exists():
        return None
    with open(location_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or ',' not in line:
                continue
            parts = line.split(',')
            if len(parts) >= 3 and parts[0].strip() == 'pole_top':
                try:
                    x_global, y_global = _pct_to_pixels(float(parts[1]), float(parts[2]), img_width, img_height)
                    return {'x': x_global, 'y': y_global}
                except (ValueError, IndexError):
                    continue
    return None


def load_ruler_marking_keypoints_from_location_file(location_path: Path, img_width: int, img_height: int) -> Optional[Dict[float, Dict[str, float]]]:
    """Load ruler marking keypoints from location file (percentage coordinates -> global pixel coordinates)."""
    if not location_path.exists():
        return None

    expected_heights = [2.5, 6.5, 10.5, 14.5, 16.5]
    keypoints = {}

    with open(location_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or ',' not in line:
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    height = float(parts[0].strip())
                    if height in expected_heights:
                        x_global, y_global = _pct_to_pixels(float(parts[1]), float(parts[2]), img_width, img_height)
                        keypoints[height] = {'x': x_global, 'y': y_global}
                except (ValueError, IndexError):
                    continue

    return keypoints if keypoints else None


def load_pole_bbox_from_location_file(location_path: Path, img_width: int, img_height: int) -> Optional[Tuple[int, int, int, int]]:
    """Load pole bounding box from location file (percentage coordinates -> global pixel coordinates)."""
    return _load_bbox_from_location_file(location_path, img_width, img_height, 'Pole bounding box')


def load_ruler_bbox_from_location_file(location_path: Path, img_width: int, img_height: int) -> Optional[Tuple[int, int, int, int]]:
    """Load ruler bounding box from location file (percentage coordinates -> global pixel coordinates)."""
    return _load_bbox_from_location_file(location_path, img_width, img_height, 'Ruler bounding box')


def load_pole_top_ppi(label_path: Path) -> Optional[float]:
    """Load PPI from pole top label file comment."""
    if not label_path.exists():
        return None
    with open(label_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith('# PPI='):
                try:
                    return float(line.split('=')[1])
                except:
                    return None
    return None


def load_ppi_from_label(label_path: Path) -> Optional[float]:
    """Load PPI from label file comment. Alias for load_pole_top_ppi."""
    return load_pole_top_ppi(label_path)


def calculate_ppi_from_keypoints(keypoints: Dict[float, Dict[str, float]]) -> Optional[float]:
    """
    Calculate Pixels-per-Inch (PPI) from ruler marking keypoints using linear regression.
    
    This function implements Algorithm 3 from the research paper:
    - Fits linear regression on the five ruler marking keypoints (2.5, 6.5, 10.5, 14.5, 16.5 feet)
    - Extracts the slope (pixels per foot) from the linear model
    - Converts to pixels per inch by dividing by 12
    
    Args:
        keypoints: Dictionary mapping height (float) to {'x': x_coord, 'y': y_coord}
                  Expected keys: 2.5, 6.5, 10.5, 14.5, 16.5 (feet)
    
    Returns:
        PPI value (pixels per inch) if calculation succeeds, None otherwise
    """
    expected_heights = [2.5, 6.5, 10.5, 14.5, 16.5]
    
    # Extract heights and Y-coordinates for available keypoints
    heights = []
    y_coords = []
    
    for height in expected_heights:
        if height in keypoints and 'y' in keypoints[height]:
            heights.append(height)
            y_coords.append(keypoints[height]['y'])
    
    # Need at least 2 keypoints for linear regression
    if len(heights) < 2:
        return None
    
    try:
        # Fit linear regression: y = a * h + b, where a is pixels per foot
        y_coeffs = np.polyfit(heights, y_coords, 1)
        slope = y_coeffs[0]  # pixels per foot
        
        # Convert to pixels per inch: PPI = slope / 12
        ppi = slope / 12.0
        
        return ppi if ppi > 0 else None
    except Exception:
        return None


def load_ground_truth_keypoints(image_path: Path, keypoint_names: List[str]) -> List[Dict]:
    """
    Load ground truth ruler marking keypoints from training dataset label file (YOLO format).
    
    Args:
        image_path: Path to ruler crop image
        keypoint_names: List of keypoint names (e.g., ['2.5', '6.5', '10.5', '14.5', '16.5'])
        
    Returns:
        List of keypoint dictionaries with 'name', 'x', 'y', 'conf', 'ppi' keys
    """
    import cv2
    
    # Load from training dataset label file (YOLO format with normalized coordinates)
    label_file_path = DATASETS_DIR / 'ruler_marking_detection' / 'labels' / 'val' / f"{image_path.stem}.txt"
    
    if not label_file_path.exists():
        return []

    # Load the ruler crop image to get dimensions
    orig_image = cv2.imread(str(image_path))
    if orig_image is None:
        return []
    orig_h, orig_w = orig_image.shape[:2]

    # Load keypoints from YOLO format label file
    gt_points = []
    
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the label line (non-comment line)
    label_line = None
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            label_line = line
            break
    
    if not label_line:
        return []
    
    # Parse YOLO format: class x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
    parts = label_line.split()
    num_keypoints = len(keypoint_names)
    if len(parts) >= 5 + num_keypoints * 3:
        # Keypoints start at index 5 (after class + bbox)
        for i in range(num_keypoints):
            kp_idx = 5 + i * 3
            if kp_idx + 2 < len(parts):
                try:
                    # YOLO format: normalized coordinates (0-1) relative to image
                    x_norm = float(parts[kp_idx])
                    y_norm = float(parts[kp_idx + 1])
                    visibility = float(parts[kp_idx + 2])
                    
                    # Only include visible keypoints
                    if visibility >= 2:  # Visible keypoint
                        # Convert normalized coordinates to pixel coordinates
                        x_px = x_norm * orig_w
                        y_px = y_norm * orig_h
                        
                        gt_points.append({
                            'name': keypoint_names[i],
                            'x': x_px,
                            'y': y_px,
                            'conf': 1.0,  # GT has full confidence
                            'ppi': 0.0  # PPI not available from YOLO labels
                        })
                except (ValueError, IndexError):
                    continue
    
    return gt_points


def load_ground_truth_pole_top(image_path: Path) -> Optional[Dict]:
    """Load ground truth pole top keypoint from Pole Top Detection dataset label file."""
    import cv2

    orig_image = cv2.imread(str(image_path))
    if orig_image is None:
        return None
    orig_h, orig_w = orig_image.shape[:2]

    # Determine split from path
    if '/test/' in str(image_path):
        split = 'test'
    elif '/train/' in str(image_path):
        split = 'train'
    else:
        split = 'val'

    label_path = DATASETS_DIR / 'pole_top_detection' / 'labels' / split / f"{image_path.stem}.txt"
    if not label_path.exists():
        return None

    with open(label_path, 'r') as f:
        ppi = 0.0
        for line in f:
            line = line.strip()
            if line.startswith('# PPI='):
                try:
                    ppi = float(line.split('=')[1])
                except:
                    pass
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 8:
                try:
                    if int(parts[7]) >= 2:  # Visible
                        x_full = float(parts[5]) * orig_w
                        y_full = float(parts[6]) * orig_h
                        return {'name': 'pole_top', 'x': x_full, 'y': y_full, 'conf': 1.0, 'ppi': ppi}
                except (ValueError, IndexError):
                    continue

    return None

# ============================================================================
# Equipment Detection Data Utilities
# ============================================================================


def parse_equipment_from_label_file(label_path: Path) -> List[Dict]:
    """
    Parse equipment bounding boxes from a location label file.

    Equipment bbox lines have format:
        riser1_bbox,Left,Right,Top,Bottom  (percentage coordinates 0-100)

    Args:
        label_path: Path to the *_location.txt file

    Returns:
        List of dicts with keys: 'class_id', 'class_name', 'left', 'right', 'top', 'bottom'
        (all coordinates in percentage 0-100)
    """
    if not label_path.exists():
        return []

    equipment = []

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(',')
            if len(parts) < 5:
                continue

            name = parts[0].strip()
            if not name.endswith('_bbox'):
                continue

            try:
                left = float(parts[1])
                right = float(parts[2])
                top = float(parts[3])
                bottom = float(parts[4])
            except (ValueError, IndexError):
                continue

            # Determine class from name prefix
            if name.startswith('riser'):
                class_name = 'riser'
            elif name.startswith('transformer'):
                class_name = 'transformer'
            elif name.startswith('street_light'):
                class_name = 'street_light'
            elif name.startswith('secondary_drip_loop'):
                class_name = 'secondary_drip_loop'
            else:
                continue

            equipment.append({
                'class_id': EQUIPMENT_CLASSES[class_name],
                'class_name': class_name,
                'left': left,
                'right': right,
                'top': top,
                'bottom': bottom,
            })

    return equipment


def equipment_bbox_to_yolo(left: float, right: float, top: float, bottom: float) -> Tuple[float, float, float, float]:
    """
    Convert equipment bbox from percentage (0-100) left/right/top/bottom
    to YOLO normalized (0-1) x_center/y_center/width/height.

    Returns:
        (x_center, y_center, width, height) all in 0-1 range
    """
    x_center = (left + right) / 200.0
    y_center = (top + bottom) / 200.0
    width = (right - left) / 100.0
    height = (bottom - top) / 100.0
    return x_center, y_center, width, height


def parse_attachments_from_label_file(label_path: Path) -> List[Dict]:
    """
    Parse attachment bounding boxes (comm, down_guy) from a location label file.

    Attachment bbox lines have format:
        comm1_bbox,Left,Right,Top,Bottom  (percentage coordinates 0-100)
        down_guy1_bbox,Left,Right,Top,Bottom

    Returns:
        List of dicts with keys: 'class_id', 'class_name', 'left', 'right', 'top', 'bottom'
    """
    if not label_path.exists():
        return []

    attachments = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 5:
                continue
            name = parts[0].strip()
            if not name.endswith('_bbox'):
                continue
            try:
                left, right, top, bottom = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except (ValueError, IndexError):
                continue
            if name.startswith('comm'):
                class_name = 'comm'
            elif name.startswith('down_guy'):
                class_name = 'down_guy'
            elif name.startswith('primary'):
                class_name = 'primary'
            elif name.startswith('secondary_drip'):
                continue  # equipment, not attachment
            elif name.startswith('secondary'):
                class_name = 'secondary'
            elif name.startswith('open_secondary') or name.startswith('neutral'):
                class_name = 'neutral'
            elif name.startswith('power_guy') or (name.startswith('guy') and not name.startswith('guying')):
                class_name = 'guy'
            else:
                continue
            attachments.append({
                'class_id': ATTACHMENT_CLASSES[class_name],
                'class_name': class_name,
                'left': left,
                'right': right,
                'top': top,
                'bottom': bottom,
            })
    return attachments


def parse_attachments_with_keypoints(label_path: Path) -> List[Dict]:
    """
    Parse attachment bboxes and center keypoints from a location file.

    Returns list of dicts with: class_id, class_name, left, right, top, bottom, center (px%, py%).
    Center comes from comm1, down_guy1 lines (Measurement,PercentX,PercentY).
    """
    if not label_path.exists():
        return []

    bboxes = {}
    centers = {}
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            name = parts[0].strip()
            if name.endswith('_bbox') and len(parts) >= 5:
                try:
                    prefix = name[:-5]
                    bboxes[prefix] = {
                        'left': float(parts[1]), 'right': float(parts[2]),
                        'top': float(parts[3]), 'bottom': float(parts[4]),
                    }
                except (ValueError, IndexError):
                    pass
            elif (name.startswith('comm') or name.startswith('down_guy') or name.startswith('primary')
                  or name.startswith('secondary') or name.startswith('open_secondary') or name.startswith('neutral')
                  or name.startswith('power_guy') or name.startswith('guy')) \
                 and not name.endswith('_bbox') and not name.startswith('secondary_drip') and not name.startswith('guying') and len(parts) >= 3:
                try:
                    centers[name] = (float(parts[1]), float(parts[2]))
                except (ValueError, IndexError):
                    pass

    results = []
    for prefix, bbox in bboxes.items():
        if prefix.startswith('comm'):
            class_name = 'comm'
        elif prefix.startswith('down_guy'):
            class_name = 'down_guy'
        elif prefix.startswith('primary'):
            class_name = 'primary'
        elif prefix.startswith('secondary_drip'):
            continue
        elif prefix.startswith('secondary'):
            class_name = 'secondary'
        elif prefix.startswith('open_secondary') or prefix.startswith('neutral'):
            class_name = 'neutral'
        elif prefix.startswith('power_guy') or (prefix.startswith('guy') and not prefix.startswith('guying')):
            class_name = 'guy'
        else:
            continue
        center = centers.get(prefix)
        results.append({
            'class_id': ATTACHMENT_CLASSES[class_name],
            'class_name': class_name,
            'left': bbox['left'], 'right': bbox['right'],
            'top': bbox['top'], 'bottom': bbox['bottom'],
            'center': center,
        })
    return results


# Per-type keypoint counts and names for separate HRNet models
# (Imported from config.py)

def riser_attachment_bbox(
    attachment_pct: Tuple[float, float],
    ppi: float,
    img_w: int,
    img_h: int,
    height_feet: float = RISER_BBOX_HEIGHT_FEET,
    width_feet: float = RISER_BBOX_WIDTH_FEET,
) -> Dict[str, float]:
    """Compute riser bbox (H x W) centered on attachment point. No padding.

    Args:
        attachment_pct: (percent_x, percent_y) in 0-100 range.
        ppi: Pixels per inch for the image.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        height_feet: Box height in feet (default from config).
        width_feet: Box width in feet (default from config).

    Returns:
        Dict with 'left', 'right', 'top', 'bottom' in percentage coordinates (0-100).
    """
    box_w_px = width_feet * 12.0 * ppi
    box_h_px = height_feet * 12.0 * ppi

    cx_px = attachment_pct[0] / 100.0 * img_w
    cy_px = attachment_pct[1] / 100.0 * img_h

    top_px = cy_px - box_h_px / 2
    bottom_px = cy_px + box_h_px / 2

    left = max(0.0, (cx_px - box_w_px / 2) / img_w * 100.0)
    right = min(100.0, (cx_px + box_w_px / 2) / img_w * 100.0)
    top = max(0.0, top_px / img_h * 100.0)
    bottom = min(100.0, bottom_px / img_h * 100.0)

    return {'left': left, 'right': right, 'top': top, 'bottom': bottom}


def parse_equipment_with_keypoints(label_path: Path) -> List[Dict]:
    """
    Parse equipment bboxes AND their associated keypoints from a location file.

    Returns a list of equipment instances, each with bbox and up to 2 keypoints
    (top/primary, bottom/secondary) in percentage coordinates (0-100).

    Keypoint mapping:
      - riser:        kp0 = riser point,        kp1 = None, kp2 = None (1 keypoint)
      - transformer:  kp0 = top_bolt,            kp1 = bottom, kp2 = None (2 keypoints)
      - street_light: kp0 = upper bracket,       kp1 = lower bracket, kp2 = drip_loop (3 keypoints)
    """
    if not label_path.exists():
        return []

    # First pass: collect all equipment keypoints and bboxes
    bboxes = {}       # e.g. 'riser1' -> {left, right, top, bottom}
    keypoints = {}    # e.g. 'riser1' -> (px, py)  or  'transformer1_top' -> (px, py)

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            name = parts[0].strip()

            if name.endswith('_bbox') and len(parts) >= 5:
                try:
                    prefix = name[:-5]  # strip '_bbox'
                    bboxes[prefix] = {
                        'left': float(parts[1]), 'right': float(parts[2]),
                        'top': float(parts[3]), 'bottom': float(parts[4]),
                    }
                except (ValueError, IndexError):
                    continue
            elif len(parts) >= 3:
                if name.startswith(('riser', 'transformer', 'street_light', 'secondary_drip_loop')):
                    try:
                        keypoints[name] = (float(parts[1]), float(parts[2]))
                    except (ValueError, IndexError):
                        continue

    # Second pass: pair bboxes with their keypoints
    results = []
    for prefix, bbox in bboxes.items():
        # Determine class
        if prefix.startswith('riser'):
            class_name = 'riser'
            # Single keypoint: the riser attachment point
            kp0 = keypoints.get(prefix)  # e.g. 'riser1'
            kp1 = None
        elif prefix.startswith('transformer'):
            class_name = 'transformer'
            kp0 = keypoints.get(f'{prefix}_top')
            kp1 = keypoints.get(f'{prefix}_bottom')
        elif prefix.startswith('street_light'):
            class_name = 'street_light'
            kp0 = keypoints.get(f'{prefix}_upper')
            kp1 = keypoints.get(f'{prefix}_lower')
            kp2 = keypoints.get(f'{prefix}_drip_loop')
        elif prefix.startswith('secondary_drip_loop'):
            class_name = 'secondary_drip_loop'
            kp0 = keypoints.get(prefix)
            kp1 = None
            kp2 = None
        else:
            continue

        # Must have at least one keypoint (street_light needs upper or lower; drip_loop is optional)
        if kp0 is None and kp1 is None:
            continue

        result = {
            'class_id': EQUIPMENT_CLASSES[class_name],
            'class_name': class_name,
            'bbox': bbox,
            'kp0': kp0,
            'kp1': kp1,
        }
        if class_name == 'street_light':
            result['kp2'] = kp2
        results.append(result)

    return results


def _compute_pole_upper70_2x5_crop(
    img: np.ndarray,
    pole_bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[np.ndarray, int, int, int, int, int, int]]:
    """
    Crop to pole bbox, upper 70%, with horizontal expansion for 2:5 aspect ratio.
    Returns (crop, x1_new, y1, x2_new, crop_y2, crop_w_actual, crop_h_actual) or None.
    """
    x1, y1, x2, y2 = pole_bbox
    crop_h_full = y2 - y1
    crop_h = int(crop_h_full * 0.7)
    if crop_h < 10 or (x2 - x1) < 10:
        return None
    target_width = int(crop_h * (2 / 5))
    center_x = (x1 + x2) / 2
    x1_new = max(0, int(center_x - target_width / 2))
    x2_new = min(img_w, int(center_x + target_width / 2))
    if x2_new - x1_new < 10:
        return None
    crop = img[y1 : y1 + crop_h, x1_new:x2_new]
    crop_h_actual, crop_w_actual = crop.shape[:2]
    crop_y2 = y1 + crop_h
    return crop, x1_new, y1, x2_new, crop_y2, crop_w_actual, crop_h_actual


def prepare_equipment_detection_dataset(photos_dir: Path, labels_dir: Path, dataset_dir: Path, verbose: bool = False, workers: int = 1, max_neg_ratio: float = 0.2) -> None:
    """
    Prepare equipment detection dataset (Riser, Transformer, Street Light) for YOLO training.

    Crops each image to the pole bounding box, then takes the upper 70% of that
    crop (where equipment typically appears). Expands the bbox width so the crop
    has 2:5 aspect ratio (no padding—pure crop from source). Only equipment fully contained
    within this region are included. Negative examples (labeled images with no
    equipment in the crop) are included with empty label files, capped at
    max_neg_ratio of positive examples per split. Unlabeled photos
    (no equipment or attachment markers) are skipped.

    Args:
        photos_dir: Path to directory with photos (*.jpg)
        labels_dir: Path to directory with *_location.txt label files
        dataset_dir: Output directory for prepared dataset

    Note:
        Delete the existing dataset directory before re-running if you change
        the preparation logic (e.g. crop strategy).
    """
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    photos_dir = Path(photos_dir)
    labels_dir = Path(labels_dir)
    dataset_dir = Path(dataset_dir)
    manifest = load_split_manifest()

    if check_dataset_complete(dataset_dir, manifest=manifest, domain="pole") if manifest else check_dataset_complete(dataset_dir):
        if verbose:
            print(f"✓ Equipment detection dataset already prepared at {dataset_dir}")
            for split in ['train', 'val', 'test']:
                n_img = len(list((dataset_dir / "images" / split).glob("*.jpg")))
                n_lbl = len(list((dataset_dir / "labels" / split).glob("*.txt")))
                print(f"  {split}: {n_img} images, {n_lbl} labels")
        return

    # Find all photos with pole bbox (required for cropping)
    from PIL import Image
    photo_files = []
    equipment_cache = {}
    photos_skipped_no_pole = 0
    photos_skipped_unlabeled = 0
    photos_with_equipment = 0

    for photo_path in sorted(photos_dir.glob("*.jpg")):
        label_path = labels_dir / f"{photo_path.stem}_location.txt"
        if not label_path.exists():
            continue
        
        # Skip unlabeled photos (only have pole_top/height markers, no equipment/attachments)
        if not is_photo_labeled(label_path):
            photos_skipped_unlabeled += 1
            continue
        try:
            with Image.open(photo_path) as im:
                img_w, img_h = im.size
        except Exception:
            continue
        pole_bbox = load_pole_bbox_from_location_file(label_path, img_w, img_h)
        if pole_bbox is None:
            photos_skipped_no_pole += 1
            continue
        equipment = parse_equipment_from_label_file(label_path)
        equipment_cache[photo_path.stem] = equipment
        if equipment:
            photos_with_equipment += 1
        photo_files.append(photo_path)

    if verbose:
        print(f"Found {len(photo_files)} pole photos with pole bbox")
        print(f"  Skipped {photos_skipped_no_pole} (no pole bbox)")
        print(f"  Skipped {photos_skipped_unlabeled} (unlabeled - no equipment/attachment markers)")
        print(f"  {photos_with_equipment} with equipment")
    class_counts = {name: 0 for name in EQUIPMENT_CLASS_NAMES}
    for photo_path in photo_files:
        for eq in equipment_cache.get(photo_path.stem, []):
            class_counts[eq['class_name']] += 1
    if verbose:
        print(f"Class distribution: {class_counts}")
        print(f"Total equipment instances: {sum(class_counts.values())}")

    # Create directory structure
    for split in ['train', 'val', 'test']:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    if manifest:
        pole_split_map = get_pole_split_map(manifest)
        split_map = {f: pole_split_map.get(f.stem, 'train') for f in photo_files}
        if verbose:
            train_count = sum(1 for s in split_map.values() if s == 'train')
            val_count = sum(1 for s in split_map.values() if s == 'val')
            test_count = sum(1 for s in split_map.values() if s == 'test')
            print(f"Split (manifest): {train_count} train / {val_count} val / {test_count} test")
    else:
        train_files, temp_files = train_test_split(photo_files, test_size=0.2, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        split_map = {f: 'train' for f in train_files}
        split_map.update({f: 'val' for f in val_files})
        split_map.update({f: 'test' for f in test_files})
        if verbose:
            print(f"Split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")
    if verbose:
        print(f"Strategy: Crop to pole bbox, upper 70%, expand width for 2:5 ratio, include negatives")

    # Pre-build riser keypoint lookup and ppi cache (parse once per photo)
    riser_kp_lookup = {}
    ppi_cache = {}
    for photo_path in photo_files:
        label_path = labels_dir / f"{photo_path.stem}_location.txt"
        eq_with_kp = parse_equipment_with_keypoints(label_path)
        ppi_cache[photo_path.stem] = load_ppi_from_label(label_path)
        riser_idx = 0
        for eq in eq_with_kp:
            if eq['class_name'] == 'riser' and eq['kp0'] is not None:
                riser_kp_lookup[(photo_path.stem, riser_idx)] = eq['kp0']
                riser_idx += 1

    # Generate YOLO dataset
    processed = 0
    skipped = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    class_counts = {name: 0 for name in EQUIPMENT_CLASS_NAMES}
    riser_bbox_replaced = 0

    def _process_one_eq(photo_path):
        label_path = labels_dir / f"{photo_path.stem}_location.txt"
        img = cv2.imread(str(photo_path))
        if img is None:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0}, {n: 0 for n in EQUIPMENT_CLASS_NAMES}, 0)

        img_h, img_w = img.shape[:2]
        pole_bbox = load_pole_bbox_from_location_file(label_path, img_w, img_h)
        if pole_bbox is None:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0}, {n: 0 for n in EQUIPMENT_CLASS_NAMES}, 0)

        crop_result = _compute_pole_upper70_2x5_crop(img, pole_bbox, img_w, img_h)
        if crop_result is None:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0}, {n: 0 for n in EQUIPMENT_CLASS_NAMES}, 0)

        crop, x1_new, y1, x2_new, crop_y2, crop_w_actual, crop_h_actual = crop_result
        ppi = ppi_cache.get(photo_path.stem)
        split = split_map[photo_path]
        img_dst = dataset_dir / "images" / split / photo_path.name
        lbl_dst = dataset_dir / "labels" / split / f"{photo_path.stem}.txt"

        equipment = equipment_cache.get(photo_path.stem, [])
        lines = []
        riser_idx = 0
        _class_counts = {n: 0 for n in EQUIPMENT_CLASS_NAMES}
        _riser_replaced = 0

        for eq in equipment:
            l_px = eq['left'] / 100.0 * img_w
            r_px = eq['right'] / 100.0 * img_w
            t_px = eq['top'] / 100.0 * img_h
            b_px = eq['bottom'] / 100.0 * img_h
            if l_px >= x1_new and r_px <= x2_new and t_px >= y1 and b_px <= crop_y2:
                if eq['class_name'] == 'riser':
                    kp = riser_kp_lookup.get((photo_path.stem, riser_idx))
                    if kp is not None and ppi is not None:
                        new_bbox = riser_attachment_bbox(kp, ppi, img_w, img_h)
                        l_px_new = new_bbox['left'] / 100.0 * img_w
                        r_px_new = new_bbox['right'] / 100.0 * img_w
                        t_px_new = new_bbox['top'] / 100.0 * img_h
                        b_px_new = new_bbox['bottom'] / 100.0 * img_h
                        if l_px_new >= x1_new and r_px_new <= x2_new and t_px_new >= y1 and b_px_new <= crop_y2:
                            l_px, r_px, t_px, b_px = l_px_new, r_px_new, t_px_new, b_px_new
                            _riser_replaced += 1
                    riser_idx += 1
                left_crop = (l_px - x1_new) / crop_w_actual * 100.0
                right_crop = (r_px - x1_new) / crop_w_actual * 100.0
                top_crop = (t_px - y1) / crop_h_actual * 100.0
                bottom_crop = (b_px - y1) / crop_h_actual * 100.0
                cx, cy, w, h = equipment_bbox_to_yolo(left_crop, right_crop, top_crop, bottom_crop)
                lines.append(f"{eq['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                _class_counts[eq['class_name']] += 1
            elif eq['class_name'] == 'riser':
                riser_idx += 1

        _split = {'train': 0, 'val': 0, 'test': 0}
        _split[split] = 1

        if not lines:
            cv2.imwrite(str(img_dst), crop)
            lbl_dst.write_text("")
            return (1, 0, _split, _class_counts, _riser_replaced)

        cv2.imwrite(str(img_dst), crop)
        lbl_dst.write_text('\n'.join(lines) + '\n')
        return (1, 0, _split, _class_counts, _riser_replaced)

    results = _parallel_map(photo_files, _process_one_eq, workers, desc="Preparing equipment dataset", verbose=verbose)
    for _processed, _skipped, _split, _class, _riser in results:
        processed += _processed
        skipped += _skipped
        for k in split_counts:
            split_counts[k] += _split.get(k, 0)
        for k in class_counts:
            class_counts[k] += _class.get(k, 0)
        riser_bbox_replaced += _riser

    # Downsample negative examples (empty label files) per split
    import random as _random
    _random.seed(42)
    neg_removed = 0
    for split in ['train', 'val', 'test']:
        lbl_dir = dataset_dir / "labels" / split
        img_dir = dataset_dir / "images" / split
        all_labels = list(lbl_dir.glob("*.txt"))
        positives = [p for p in all_labels if p.stat().st_size > 0]
        negatives = [p for p in all_labels if p.stat().st_size == 0]
        max_neg = int(len(positives) * max_neg_ratio)
        if len(negatives) > max_neg:
            _random.shuffle(negatives)
            to_remove = negatives[max_neg:]
            for lbl_path in to_remove:
                img_path = img_dir / f"{lbl_path.stem}.jpg"
                lbl_path.unlink(missing_ok=True)
                img_path.unlink(missing_ok=True)
            neg_removed += len(to_remove)
            split_counts[split] -= len(to_remove)
            if verbose:
                print(f"  {split}: kept {max_neg}/{len(negatives)} negatives ({len(to_remove)} removed)")

    # Write data.yaml
    yaml_content = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(EQUIPMENT_CLASS_NAMES)}\n"
        f"names: {EQUIPMENT_CLASS_NAMES}\n"
    )
    with open(dataset_dir / "data.yaml", 'w') as f:
        f.write(yaml_content)

    if verbose:
        print(f"\n✓ Equipment dataset prepared: {processed} images ({skipped} skipped, {neg_removed} negatives removed)")
        print(f"  Riser bboxes replaced with 5'x2' attachment box: {riser_bbox_replaced}")
        for split, count in split_counts.items():
            print(f"  {split}: {count} images")
        print(f"  Class distribution in crops: {class_counts}")
        print(f"  data.yaml: {dataset_dir / 'data.yaml'}")
        print(f"  Classes: {EQUIPMENT_CLASS_NAMES}")


def prepare_attachment_detection_dataset(photos_dir: Path, labels_dir: Path, dataset_dir: Path, verbose: bool = False, workers: int = 1, max_neg_ratio: float = 0.2) -> None:
    """
    Prepare attachment detection dataset (comm, down_guy) for YOLO training.

    Crops each image to the pole bounding box, then takes the upper 70% of that
    crop (where attachments typically appear). Expands the bbox width so the crop
    has 2:5 aspect ratio (no padding—pure crop from source). Only attachments fully contained
    within this region are included. Negative examples (labeled images with no 
    attachments in the crop) are included with empty label files. Unlabeled photos
    (no equipment or attachment markers) are skipped.

    Args:
        photos_dir: Path to directory with photos (*.jpg)
        labels_dir: Path to directory with *_location.txt label files
        dataset_dir: Output directory for prepared dataset

    Note:
        Delete the existing dataset directory before re-running if you change
        the preparation logic (e.g. crop strategy).
    """
    import shutil
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    photos_dir = Path(photos_dir)
    labels_dir = Path(labels_dir)
    dataset_dir = Path(dataset_dir)
    manifest = load_split_manifest()

    if check_dataset_complete(dataset_dir, manifest=manifest, domain="pole") if manifest else check_dataset_complete(dataset_dir):
        if verbose:
            print(f"✓ Attachment detection dataset already prepared at {dataset_dir}")
            for split in ['train', 'val', 'test']:
                n_img = len(list((dataset_dir / "images" / split).glob("*.jpg")))
                n_lbl = len(list((dataset_dir / "labels" / split).glob("*.txt")))
                print(f"  {split}: {n_img} images, {n_lbl} labels")
        return

    # Collect photos that have pole bbox (required for cropping)
    from PIL import Image
    photo_files = []
    attachment_cache = {}
    photos_skipped_no_pole = 0
    photos_skipped_unlabeled = 0
    photos_with_attachments = 0

    for photo_path in sorted(photos_dir.glob("*.jpg")):
        label_path = labels_dir / f"{photo_path.stem}_location.txt"
        if not label_path.exists():
            continue
        if not is_photo_labeled(label_path):
            photos_skipped_unlabeled += 1
            continue
        try:
            with Image.open(photo_path) as im:
                img_w, img_h = im.size
        except Exception:
            continue
        pole_bbox = load_pole_bbox_from_location_file(label_path, img_w, img_h)
        if pole_bbox is None:
            photos_skipped_no_pole += 1
            continue
        attachments = parse_attachments_from_label_file(label_path)
        attachment_cache[photo_path.stem] = attachments
        if attachments:
            photos_with_attachments += 1
        photo_files.append(photo_path)

    if verbose:
        print(f"Found {len(photo_files)} pole photos with pole bbox")
        print(f"  Skipped {photos_skipped_no_pole} (no pole bbox)")
        print(f"  Skipped {photos_skipped_unlabeled} (unlabeled - no equipment/attachment markers)")
        print(f"  {photos_with_attachments} with attachments (comm/down_guy)")

    # Create splits
    for split in ['train', 'val', 'test']:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    if manifest:
        pole_split_map = get_pole_split_map(manifest)
        split_map = {f: pole_split_map.get(f.stem, 'train') for f in photo_files}
        if verbose:
            train_count = sum(1 for s in split_map.values() if s == 'train')
            val_count = sum(1 for s in split_map.values() if s == 'val')
            test_count = sum(1 for s in split_map.values() if s == 'test')
            print(f"Split (manifest): {train_count} train / {val_count} val / {test_count} test")
    else:
        train_files, temp_files = train_test_split(photo_files, test_size=0.2, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        split_map = {f: 'train' for f in train_files}
        split_map.update({f: 'val' for f in val_files})
        split_map.update({f: 'test' for f in test_files})
        if verbose:
            print(f"Split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")
    if verbose:
        print(f"Strategy: Crop to pole bbox, upper 70%, expand width for 2:5 ratio, include negatives")

    processed = 0
    skipped = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    class_counts = {name: 0 for name in ATTACHMENT_CLASS_NAMES}

    def _process_one_att(photo_path):
        label_path = labels_dir / f"{photo_path.stem}_location.txt"
        img = cv2.imread(str(photo_path))
        if img is None:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0}, {n: 0 for n in ATTACHMENT_CLASS_NAMES})
        img_h, img_w = img.shape[:2]
        pole_bbox = load_pole_bbox_from_location_file(label_path, img_w, img_h)
        if pole_bbox is None:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0}, {n: 0 for n in ATTACHMENT_CLASS_NAMES})
        crop_result = _compute_pole_upper70_2x5_crop(img, pole_bbox, img_w, img_h)
        if crop_result is None:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0}, {n: 0 for n in ATTACHMENT_CLASS_NAMES})
        crop, x1_new, y1, x2_new, crop_y2, crop_w_actual, crop_h_actual = crop_result
        attachments = attachment_cache.get(photo_path.stem, [])
        lines = []
        _class = {n: 0 for n in ATTACHMENT_CLASS_NAMES}
        for att in attachments:
            l_px = att['left'] / 100.0 * img_w
            r_px = att['right'] / 100.0 * img_w
            t_px = att['top'] / 100.0 * img_h
            b_px = att['bottom'] / 100.0 * img_h
            if l_px >= x1_new and r_px <= x2_new and t_px >= y1 and b_px <= crop_y2:
                left_crop = (l_px - x1_new) / crop_w_actual * 100.0
                right_crop = (r_px - x1_new) / crop_w_actual * 100.0
                top_crop = (t_px - y1) / crop_h_actual * 100.0
                bottom_crop = (b_px - y1) / crop_h_actual * 100.0
                cx, cy, w, h = equipment_bbox_to_yolo(left_crop, right_crop, top_crop, bottom_crop)
                lines.append(f"{att['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                _class[att['class_name']] += 1
        split = split_map[photo_path]
        img_dst = dataset_dir / "images" / split / photo_path.name
        lbl_dst = dataset_dir / "labels" / split / f"{photo_path.stem}.txt"
        _split = {'train': 0, 'val': 0, 'test': 0}
        _split[split] = 1
        if not lines:
            cv2.imwrite(str(img_dst), crop)
            lbl_dst.write_text("")
            return (1, 0, _split, _class)
        cv2.imwrite(str(img_dst), crop)
        lbl_dst.write_text('\n'.join(lines) + '\n')
        return (1, 0, _split, _class)

    results = _parallel_map(photo_files, _process_one_att, workers, desc="Preparing attachment dataset", verbose=verbose)
    for _processed, _skipped, _split, _class in results:
        processed += _processed
        skipped += _skipped
        for k in split_counts:
            split_counts[k] += _split.get(k, 0)
        for k in class_counts:
            class_counts[k] += _class.get(k, 0)

    # Downsample negative examples (empty label files) per split
    import random as _random
    _random.seed(42)
    neg_removed = 0
    for split in ['train', 'val', 'test']:
        lbl_dir = dataset_dir / "labels" / split
        img_dir = dataset_dir / "images" / split
        all_labels = list(lbl_dir.glob("*.txt"))
        positives = [p for p in all_labels if p.stat().st_size > 0]
        negatives = [p for p in all_labels if p.stat().st_size == 0]
        max_neg = int(len(positives) * max_neg_ratio)
        if len(negatives) > max_neg:
            _random.shuffle(negatives)
            to_remove = negatives[max_neg:]
            for lbl_path in to_remove:
                img_path = img_dir / f"{lbl_path.stem}.jpg"
                lbl_path.unlink(missing_ok=True)
                img_path.unlink(missing_ok=True)
            neg_removed += len(to_remove)
            split_counts[split] -= len(to_remove)
            if verbose:
                print(f"  {split}: kept {max_neg}/{len(negatives)} negatives ({len(to_remove)} removed)")

    yaml_content = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(ATTACHMENT_CLASS_NAMES)}\n"
        f"names: {ATTACHMENT_CLASS_NAMES}\n"
    )
    (dataset_dir / "data.yaml").write_text(yaml_content)

    if verbose:
        print(f"\n✓ Attachment dataset prepared: {processed} images ({skipped} skipped, {neg_removed} negatives removed)")
        for split, count in split_counts.items():
            print(f"  {split}: {count} images")
        print(f"  Class distribution in crops: {class_counts}")
        print(f"  data.yaml: {dataset_dir / 'data.yaml'}")
        print(f"  Classes: {ATTACHMENT_CLASS_NAMES}")


def prepare_attachment_keypoint_dataset(
    photos_dir: Path,
    labels_dir: Path,
    att_type: str,
    dataset_dir: Path,
    verbose: bool = False,
    workers: int = 1,
) -> None:
    """
    Prepare keypoint detection dataset for attachments (comm, down_guy).
    Uses attachment bbox from location files; keypoint = center of bbox.
    """
    from .config import (
        COMM_NUM_KEYPOINTS,
        COMM_KEYPOINT_NAMES,
        DOWN_GUY_NUM_KEYPOINTS,
        DOWN_GUY_KEYPOINT_NAMES,
    )
    from collections import Counter
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    _single_att = {'num_keypoints': 1, 'keypoint_names': ['attachment']}
    config_map = {
        'comm': {'num_keypoints': COMM_NUM_KEYPOINTS, 'keypoint_names': COMM_KEYPOINT_NAMES},
        'down_guy': {'num_keypoints': DOWN_GUY_NUM_KEYPOINTS, 'keypoint_names': DOWN_GUY_KEYPOINT_NAMES},
        'primary': _single_att,
        'secondary': _single_att,
        'neutral': _single_att,
        'guy': _single_att,
    }
    cfg = config_map.get(att_type)
    if not cfg:
        raise ValueError(f"Unknown attachment type: {att_type}")

    photos_dir = Path(photos_dir)
    labels_dir = Path(labels_dir)
    dataset_dir = Path(dataset_dir)
    num_kp = cfg['num_keypoints']

    if check_dataset_complete(dataset_dir):
        if verbose:
            print(f"✓ {att_type.upper()} keypoint dataset already prepared at {dataset_dir}")
            for split in ['train', 'val', 'test']:
                n = len(list((dataset_dir / "images" / split).glob("*.jpg")))
                print(f"  {split}: {n} crops")
        return

    from PIL import Image
    instances = []
    for photo_path in sorted(photos_dir.glob("*.jpg")):
        label_path = labels_dir / f"{photo_path.stem}_location.txt"
        if not label_path.exists():
            continue
        try:
            with Image.open(photo_path) as im:
                img_w, img_h = im.size
        except Exception:
            continue
        pole_bbox = load_pole_bbox_from_location_file(label_path, img_w, img_h)
        if pole_bbox is None:
            continue
        x1, y1, x2, y2 = pole_bbox
        crop_h_full = y2 - y1
        crop_h = int(crop_h_full * 0.7)
        target_width = int(crop_h * (2 / 5))
        center_x = (x1 + x2) / 2
        x1_new = max(0, int(center_x - target_width / 2))
        x2_new = min(img_w, int(center_x + target_width / 2))
        crop_y2 = y1 + crop_h
        for att in parse_attachments_from_label_file(label_path):
            if att['class_name'] != att_type:
                continue
            l_px = att['left'] / 100.0 * img_w
            r_px = att['right'] / 100.0 * img_w
            t_px = att['top'] / 100.0 * img_h
            b_px = att['bottom'] / 100.0 * img_h
            if l_px >= x1_new and r_px <= x2_new and t_px >= y1 and b_px <= crop_y2:
                instances.append((photo_path, label_path, att, img_w, img_h))

    if not instances:
        if verbose:
            print(f"No {att_type} instances found in 70% crop region")
        return

    if verbose:
        photos = len(set(p for p, _, _, _, _ in instances))
        print(f"{att_type.upper()} — {len(instances)} instances across {photos} photos ({num_kp} keypoint)")

    for split in ['train', 'val', 'test']:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    unique_photos = sorted(set(p for p, _, _, _, _ in instances))
    manifest = load_split_manifest()
    if manifest:
        pole_split_map = get_pole_split_map(manifest)
        split_map = {p: pole_split_map.get(p.stem, 'train') for p in unique_photos}
    else:
        train_photos, temp_photos = train_test_split(unique_photos, test_size=0.2, random_state=42)
        val_photos, test_photos = train_test_split(temp_photos, test_size=0.5, random_state=42)
        split_map = {p: s for ps, s in [(train_photos, 'train'), (val_photos, 'val'), (test_photos, 'test')] for p in ps}

    processed = 0
    skipped = 0
    split_counts = Counter()
    # Assign instance index per photo for stem naming
    photo_instance_idx = Counter()
    instances_with_idx = []
    for photo_path, label_path, att, img_w, img_h in instances:
        photo_instance_idx[photo_path] += 1
        instances_with_idx.append((photo_path, label_path, att, img_w, img_h, photo_instance_idx[photo_path]))

    from .config import ATTACHMENT_BBOX_HEIGHT_FEET, DOWN_GUY_BBOX_HEIGHT_FEET
    height_feet_map = {
        'comm': ATTACHMENT_BBOX_HEIGHT_FEET,
        'down_guy': DOWN_GUY_BBOX_HEIGHT_FEET,
        'primary': ATTACHMENT_BBOX_HEIGHT_FEET,
        'secondary': ATTACHMENT_BBOX_HEIGHT_FEET,
        'neutral': ATTACHMENT_BBOX_HEIGHT_FEET,
        'guy': ATTACHMENT_BBOX_HEIGHT_FEET,
    }
    height_inches = height_feet_map[att_type] * 12.0  # feet to inches

    def _process_one_att_kp(item):
        photo_path, label_path, att, img_w, img_h, idx = item
        img = cv2.imread(str(photo_path))
        if img is None:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0})
        x1 = max(0, int(att['left'] / 100 * img_w))
        x2 = min(img_w, int(att['right'] / 100 * img_w))
        y1 = max(0, int(att['top'] / 100 * img_h))
        y2 = min(img_h, int(att['bottom'] / 100 * img_h))
        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < 10 or crop_h < 10:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0})
        pad_w, pad_h = int(crop_w * 0.15), int(crop_h * 0.15)
        x1 = max(0, x1 - pad_w)
        x2 = min(img_w, x2 + pad_w)
        y1 = max(0, y1 - pad_h)
        y2 = min(img_h, y2 + pad_h)
        crop_w, crop_h = x2 - x1, y2 - y1
        crop = img[y1:y2, x1:x2]
        orig_cx = (att['left'] + att['right']) / 200 * img_w - x1
        orig_cy = (att['top'] + att['bottom']) / 200 * img_h - y1
        kp_x = min(max(orig_cx / crop_w, 0.0), 0.999999)
        kp_y = min(max(orig_cy / crop_h, 0.0), 0.999999)
        bbox_h_px = (att['bottom'] - att['top']) / 100.0 * img_h
        ppi = bbox_h_px / height_inches if height_inches > 0 and bbox_h_px > 0 else None
        ppi_comment = f"# PPI={ppi:.6f}\n" if ppi and ppi > 0 else ""
        split = split_map[photo_path]
        stem = f"{photo_path.stem}_{att_type}{idx}"
        img_dst = dataset_dir / "images" / split / f"{stem}.jpg"
        lbl_dst = dataset_dir / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(img_dst), crop)
        label_line = "0 0.5 0.5 1.0 1.0"
        for _ in range(num_kp):
            label_line += f" {kp_x:.6f} {kp_y:.6f} 2"
        lbl_dst.write_text(ppi_comment + label_line + "\n")
        _split = {'train': 0, 'val': 0, 'test': 0}
        _split[split] = 1
        return (1, 0, _split)

    results = _parallel_map(instances_with_idx, _process_one_att_kp, workers, desc=f"Cropping {att_type}", verbose=verbose)
    for _processed, _skipped, _split in results:
        processed += _processed
        skipped += _skipped
        for k in split_counts:
            split_counts[k] += _split.get(k, 0)

    yaml_content = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names: ['{att_type}']\n"
        f"kpt_shape: [{num_kp}, 3]\n"
    )
    (dataset_dir / "data.yaml").write_text(yaml_content)
    if verbose:
        print(f"✓ {processed} crops ({skipped} skipped)")
        for split in ['train', 'val', 'test']:
            print(f"  {split}: {split_counts[split]} crops")
        print(f"  Keypoints: {cfg['keypoint_names']}")


def prepare_keypoint_detection_dataset(photos_dir: Path, labels_dir: Path, eq_type: str, dataset_dir: Path, verbose: bool = False, workers: int = 1) -> None:
    """
    Prepare keypoint detection dataset for HRNet training.

    Args:
        photos_dir: Path to directory with photos (*.jpg)
        labels_dir: Path to directory with *_location.txt label files
        eq_type: Equipment type ('riser', 'transformer', or 'street_light')
        dataset_dir: Output directory for prepared dataset
    """
    import cv2
    from collections import Counter
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    from PIL import Image

    photos_dir = Path(photos_dir)
    labels_dir = Path(labels_dir)
    dataset_dir = Path(dataset_dir)

    # Configuration for equipment types
    config_map = {
        'riser': {
            'num_keypoints': RISER_NUM_KEYPOINTS,
            'keypoint_names': RISER_KEYPOINT_NAMES,
        },
        'transformer': {
            'num_keypoints': TRANSFORMER_NUM_KEYPOINTS,
            'keypoint_names': TRANSFORMER_KEYPOINT_NAMES,
        },
        'street_light': {
            'num_keypoints': STREET_LIGHT_NUM_KEYPOINTS,
            'keypoint_names': STREET_LIGHT_KEYPOINT_NAMES,
        },
        'secondary_drip_loop': {
            'num_keypoints': SECONDARY_DRIP_LOOP_NUM_KEYPOINTS,
            'keypoint_names': SECONDARY_DRIP_LOOP_KEYPOINT_NAMES,
        },
    }

    cfg = config_map.get(eq_type)
    if not cfg:
        raise ValueError(f"Unknown equipment type: {eq_type}")

    num_kp = cfg['num_keypoints']

    if check_dataset_complete(dataset_dir):
        if verbose:
            print(f"✓ {eq_type.upper()} keypoint dataset already prepared at {dataset_dir}")
            for split in ['train', 'val', 'test']:
                n = len(list((dataset_dir / "images" / split).glob("*.jpg")))
                print(f"  {split}: {n} crops")
        return

    # Collect all equipment instances
    instances = []
    for photo_path in sorted(photos_dir.glob("*.jpg")):
        label_path = labels_dir / f"{photo_path.stem}_location.txt"
        for eq in parse_equipment_with_keypoints(label_path):
            if eq['class_name'] == eq_type:
                instances.append((photo_path, label_path, eq))

    if not instances:
        if verbose:
            print(f"No {eq_type} instances found")
        return

    if verbose:
        photos = len(set(p for p, _, _ in instances))
        print(f"{eq_type.upper()} — {len(instances)} instances across {photos} photos ({num_kp} keypoint(s))")

    # Create directory structure
    for split in ['train', 'val', 'test']:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Split by photo; use manifest for consistency
    unique_photos = sorted(set(p for p, _, _ in instances))
    ppi_cache = {p.stem: load_ppi_from_label(labels_dir / f"{p.stem}_location.txt") for p in unique_photos}
    manifest = load_split_manifest()
    if manifest:
        pole_split_map = get_pole_split_map(manifest)
        split_map = {p: pole_split_map.get(p.stem, 'train') for p in unique_photos}
    else:
        train_photos, temp_photos = train_test_split(unique_photos, test_size=0.2, random_state=42)
        val_photos, test_photos = train_test_split(temp_photos, test_size=0.5, random_state=42)
        split_map = {p: s for ps, s in [(train_photos, 'train'), (val_photos, 'val'), (test_photos, 'test')] for p in ps}

    if verbose:
        train_count = sum(1 for s in split_map.values() if s == 'train')
        val_count = sum(1 for s in split_map.values() if s == 'val')
        test_count = sum(1 for s in split_map.values() if s == 'test')
        print(f"Split: {train_count} train / {val_count} val / {test_count} test photos")

    # Generate crops
    processed = 0
    skipped = 0
    split_counts = Counter()
    photo_instance_idx = Counter()
    instances_with_idx = []
    for photo_path, label_path, eq in instances:
        photo_instance_idx[photo_path] += 1
        instances_with_idx.append((photo_path, label_path, eq, photo_instance_idx[photo_path]))

    def _process_one_eq_kp(item):
        photo_path, label_path, eq, idx = item
        img = cv2.imread(str(photo_path))
        if img is None:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0})
        h, w = img.shape[:2]
        bbox = eq['bbox']
        if eq_type == 'riser' and eq['kp0'] is not None:
            ppi = ppi_cache.get(photo_path.stem)
            if ppi is not None:
                bbox = riser_attachment_bbox(eq['kp0'], ppi, w, h)
        x1 = max(0, int(bbox['left'] / 100 * w))
        x2 = min(w, int(bbox['right'] / 100 * w))
        y1 = max(0, int(bbox['top'] / 100 * h))
        y2 = min(h, int(bbox['bottom'] / 100 * h))
        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < 10 or crop_h < 10:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0})
        crop = img[y1:y2, x1:x2]
        kp_sources = [eq.get('kp0')]
        for i in range(1, num_kp):
            kp_sources.append(eq.get(f'kp{i}'))
        kp_data = []
        for kp in kp_sources:
            if kp is not None:
                kp_x = min(max((kp[0] / 100 * w - x1) / crop_w, 0.0), 0.999999)
                kp_y = min(max((kp[1] / 100 * h - y1) / crop_h, 0.0), 0.999999)
                kp_data.append((kp_x, kp_y, 2))
            else:
                kp_data.append((0.0, 0.0, 0))
        split = split_map[photo_path]
        stem = f"{photo_path.stem}_{eq_type}{idx}"
        img_dst = dataset_dir / "images" / split / f"{stem}.jpg"
        lbl_dst = dataset_dir / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(img_dst), crop)
        label_line = "0 0.5 0.5 1.0 1.0"
        for kx, ky, kv in kp_data:
            label_line += f" {kx:.6f} {ky:.6f} {kv}"
        ppi = ppi_cache.get(photo_path.stem)
        ppi_comment = f"# PPI={ppi}\n" if ppi else ""
        with open(lbl_dst, 'w') as f:
            f.write(ppi_comment + label_line + '\n')
        _split = {'train': 0, 'val': 0, 'test': 0}
        _split[split] = 1
        return (1, 0, _split)

    results = _parallel_map(instances_with_idx, _process_one_eq_kp, workers, desc=f"Cropping {eq_type}", verbose=verbose)
    for _processed, _skipped, _split in results:
        processed += _processed
        skipped += _skipped
        for k in split_counts:
            split_counts[k] += _split.get(k, 0)

    # Write data.yaml
    yaml_content = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names: ['{eq_type}']\n"
        f"kpt_shape: [{num_kp}, 3]\n"
    )
    with open(dataset_dir / "data.yaml", 'w') as f:
        f.write(yaml_content)

    if verbose:
        print(f"✓ {processed} crops ({skipped} skipped)")
        for split in ['train', 'val', 'test']:
            print(f"  {split}: {split_counts[split]} crops")
        print(f"  Keypoints: {cfg['keypoint_names']}")


def prepare_calibration_datasets(
    pole_photos_dir: Path, pole_labels_dir: Path,
    midspan_photos_dir: Path, midspan_labels_dir: Path,
    datasets_dir: Path = DATASETS_DIR,
    verbose: bool = False,
    workers: int = 1,
) -> None:
    """
    Prepare all calibration datasets (pole, ruler, ruler marking, pole top detection).

    Handles:
    - Train/val/test splitting (80/10/10)
    - YOLO format label generation
    - Filtering by keypoint visibility
    - Dataset completion checking
    """
    # Ensure all paths are Path objects
    pole_photos_dir = Path(pole_photos_dir)
    pole_labels_dir = Path(pole_labels_dir)
    midspan_photos_dir = Path(midspan_photos_dir)
    midspan_labels_dir = Path(midspan_labels_dir)
    datasets_dir = Path(datasets_dir)

    # Create dataset directories
    datasets = {
        'pole_detection': datasets_dir / 'pole_detection',
        'ruler_detection': datasets_dir / 'ruler_detection',
        'ruler_marking_detection': datasets_dir / 'ruler_marking_detection',
        'pole_top_detection': datasets_dir / 'pole_top_detection',
    }

    for dataset_dir in datasets.values():
        for split in ['train', 'val', 'test']:
            (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Collect images with labels
    pole_files = [p for p in pole_photos_dir.glob('*.jpg')
                  if (pole_labels_dir / f'{p.stem}_location.txt').exists()]
    midspan_files = [p for p in midspan_photos_dir.glob('*.jpg')
                     if (midspan_labels_dir / f'{p.stem}_location.txt').exists()]

    photo_files = pole_files + midspan_files
    if not photo_files:
        raise RuntimeError(f'No images found (pole: {len(pole_files)}, midspan: {len(midspan_files)})')

    # Use master split manifest; pole and midspan have separate splits
    manifest = load_split_manifest()
    if manifest:
        pole_map = get_pole_split_map(manifest)
        midspan_map = get_midspan_split_map(manifest)
        split_map = {}
        for p in pole_files:
            split_map[p] = pole_map.get(p.stem, 'train')
        for p in midspan_files:
            split_map[p] = midspan_map.get(p.stem, 'train')
    else:
        train_files, temp_files = train_test_split(photo_files, test_size=0.2, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        split_map = {f: 'train' for f in train_files}
        split_map.update({f: 'val' for f in val_files})
        split_map.update({f: 'test' for f in test_files})

    completeness = {k: check_dataset_complete(v) for k, v in datasets.items()}
    if verbose:
        print(f'Dataset completion: {sum(completeness.values())}/{len(completeness)}')
    if all(completeness.values()):
        if verbose:
            print('✓ All datasets complete, skipping preparation')
        return

    # Process images
    processed = {k: 0 for k in datasets}
    skipped_total = 0

    def _process_one(photo_path):
        out = {'processed': processed.copy(), 'skipped': 0}
        out['processed'] = {k: 0 for k in datasets}
        if photo_path.resolve().is_relative_to(midspan_photos_dir.resolve()):
            label_path = midspan_labels_dir / f'{photo_path.stem}_location.txt'
        else:
            label_path = pole_labels_dir / f'{photo_path.stem}_location.txt'

        img = cv2.imread(str(photo_path))
        if img is None:
            return (out['processed'], 1)  # skipped

        h, w = img.shape[:2]
        pole_bbox, ruler_bbox, keypoints, ppi = parse_label_file(label_path)
        subdir = split_map[photo_path]

        # Pole detection
        if pole_bbox and not completeness['pole_detection']:
            pole_top_kp = keypoints.get('pole_top')
            added_pole = False
            if pole_top_kp:
                x1, x2, y1, y2 = [int(v/100*w) if i < 2 else int(v/100*h)
                                   for i, v in enumerate(pole_bbox)]
                if x1 < x2 and y1 < y2:
                    img_path = datasets['pole_detection'] / 'images' / subdir / photo_path.name
                    lbl_path = datasets['pole_detection'] / 'labels' / subdir / f'{photo_path.stem}.txt'
                    if not img_path.exists() or not lbl_path.exists():
                        shutil.copy(photo_path, img_path)
                        cx, cy = (pole_bbox[0] + pole_bbox[1]) / 200, (pole_bbox[2] + pole_bbox[3]) / 200
                        bw, bh = (pole_bbox[1] - pole_bbox[0]) / 100, (pole_bbox[3] - pole_bbox[2]) / 100
                        with open(lbl_path, 'w') as f:
                            f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
                        out['processed']['pole_detection'] = 1
                        added_pole = True
            if not added_pole:
                out['skipped'] += 1

        # Ruler and ruler marking detection
        if ruler_bbox and (not completeness['ruler_detection'] or not completeness['ruler_marking_detection']):
            x1, x2, y1, y2 = [int(v/100*w) if i < 2 else int(v/100*h)
                               for i, v in enumerate(ruler_bbox)]
            if x1 >= x2 or y1 >= y2:
                out['skipped'] += 1
            else:
                visible_kps = sum(1 for kp_name in KEYPOINT_NAMES
                                if kp_name in keypoints and
                                x1 <= keypoints[kp_name][0]/100*w <= x2 and
                                y1 <= keypoints[kp_name][1]/100*h <= y2)
                if visible_kps < 5:
                    out['skipped'] += 1
                else:
                    if not completeness['ruler_detection']:
                        img_path = datasets['ruler_detection'] / 'images' / subdir / photo_path.name
                        lbl_path = datasets['ruler_detection'] / 'labels' / subdir / f'{photo_path.stem}.txt'
                        if not img_path.exists() or not lbl_path.exists():
                            shutil.copy(photo_path, img_path)
                            cx, cy = (ruler_bbox[0] + ruler_bbox[1]) / 200, (ruler_bbox[2] + ruler_bbox[3]) / 200
                            bw, bh = (ruler_bbox[1] - ruler_bbox[0]) / 100, (ruler_bbox[3] - ruler_bbox[2]) / 100
                            with open(lbl_path, 'w') as f:
                                f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
                            out['processed']['ruler_detection'] = 1

                    if not completeness['ruler_marking_detection']:
                        crop = img[y1:y2, x1:x2]
                        crop_h, crop_w = crop.shape[:2]
                        if crop_h >= 10 and crop_w >= 10:
                            img_path = datasets['ruler_marking_detection'] / 'images' / subdir / photo_path.name
                            lbl_path = datasets['ruler_marking_detection'] / 'labels' / subdir / f'{photo_path.stem}.txt'
                            if not img_path.exists() or not lbl_path.exists():
                                cv2.imwrite(str(img_path), crop)
                                ppi_comment = f'# PPI={ppi:.6f}\n' if ppi and ppi > 0 else ''
                                label_content = ppi_comment + '0 0.5 0.5 1.0 1.0'
                                for kp_name in KEYPOINT_NAMES:
                                    if kp_name in keypoints:
                                        kx_px = keypoints[kp_name][0] / 100 * w
                                        ky_px = keypoints[kp_name][1] / 100 * h
                                        if x1 <= kx_px <= x2 and y1 <= ky_px <= y2:
                                            kx_norm = min(max((kx_px - x1) / crop_w, 0), 0.999999)
                                            ky_norm = min(max((ky_px - y1) / crop_h, 0), 0.999999)
                                            label_content += f' {kx_norm:.6f} {ky_norm:.6f} 2'
                                        else:
                                            label_content += ' 0.0 0.0 0'
                                    else:
                                        label_content += ' 0.0 0.0 0'
                                label_content += '\n'
                                with open(lbl_path, 'w') as f:
                                    f.write(label_content)
                                out['processed']['ruler_marking_detection'] = 1

        # Pole top detection
        if pole_bbox and not completeness['pole_top_detection']:
            pole_top_kp = keypoints.get('pole_top')
            x1, x2, y1, y2 = [int(v/100*w) if i < 2 else int(v/100*h)
                               for i, v in enumerate(pole_bbox)]
            if x1 < x2 and y1 < y2 and pole_top_kp:
                crop = img[y1:y2, x1:x2]
                crop_h, crop_w = crop.shape[:2]
                if crop_h >= 10 and crop_w >= 10:
                    kx_px = pole_top_kp[0] / 100 * w
                    ky_px = pole_top_kp[1] / 100 * h
                    kx_norm = min(max((kx_px - x1) / crop_w, 0), 0.999999)
                    ky_norm = min(max((ky_px - y1) / crop_h, 0), 0.999999)
                    img_path = datasets['pole_top_detection'] / 'images' / subdir / photo_path.name
                    lbl_path = datasets['pole_top_detection'] / 'labels' / subdir / f'{photo_path.stem}.txt'
                    if not img_path.exists() or not lbl_path.exists():
                        cv2.imwrite(str(img_path), crop)
                        ppi_comment = f'# PPI={ppi:.6f}\n' if ppi and ppi > 0 else ''
                        with open(lbl_path, 'w') as f:
                            f.write(f'{ppi_comment}0 0.5 0.5 1.0 1.0 {kx_norm:.6f} {ky_norm:.6f} 2\n')
                        out['processed']['pole_top_detection'] = 1

        return (out['processed'], out['skipped'])

    results = _parallel_map(photo_files, _process_one, workers, desc='Preparing datasets', verbose=verbose)
    for proc, skipped in results:
        for k in processed:
            processed[k] += proc.get(k, 0)
        skipped_total += skipped

    if verbose:
        for dataset_name, count in processed.items():
            if count > 0:
                print(f'  {dataset_name}: {count} images')
        if skipped_total > 0:
            print(f'  Skipped: {skipped_total} images')