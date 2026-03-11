"""
Evaluation utilities for model performance assessment.
"""

import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict
import cv2
import torch

# Import data loading utilities
from .data_utils import (
    load_yolo_label, load_pole_top_from_location_file,
    load_ruler_marking_keypoints_from_location_file, load_pole_top_ppi,
    load_pole_bbox_from_location_file, load_ruler_bbox_from_location_file
)
from .inference_utils import infer_keypoints_on_crop, infer_pole_top_on_crop
from .inference_utils import load_trained_keypoint_model, load_pole_top_model
from .config import (
    POLE_DETECTION_CONFIG, RULER_DETECTION_CONFIG,
    INFERENCE_POLE_CONF_THRESHOLD,
    INFERENCE_RULER_CONF_THRESHOLD,
    INFERENCE_MAX_DETECTIONS,
    POLE_LABELS_DIR,
    MIDSPAN_LABELS_DIR,
    RESULTS_CALIBRATION_DIR,
    RESULTS_ATTACHMENT_DIR,
    RESULTS_EQUIPMENT_DIR,
    EVALUATION_DATASETS_CONFIG,
    COLOR_GT,
    COLOR_PRED,
    INFERENCE_POLE_WEIGHTS,
    INFERENCE_RULER_WEIGHTS,
    INFERENCE_RULER_MARKING_WEIGHTS,
    INFERENCE_POLE_TOP_WEIGHTS,
    INFERENCE_USE_TTA,
    DATASETS_DIR,
    EVALUATION_YOLO_BATCH_SIZE,
)


def yolo_to_bbox(yolo_format, img_width, img_height):
    """Convert YOLO format to (x1, y1, x2, y2)."""
    if len(yolo_format) == 5:
        class_id, x_center, y_center, width, height = yolo_format
    elif len(yolo_format) == 4:
        x_center, y_center, width, height = yolo_format
    else:
        raise ValueError(f"Invalid YOLO format: expected 4 or 5 values, got {len(yolo_format)}")
    
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return [x1, y1, x2, y2]


def is_point_in_bbox(point_x, point_y, bbox):
    """Check if a point (x, y) is inside a bounding box [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = bbox
    return x1 <= point_x <= x2 and y1 <= point_y <= y2


def bbox_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_map(pred_boxes_list, gt_boxes_list, iou_threshold=0.5):
    """Calculate Average Precision (AP) at a given IoU threshold."""
    all_preds = []
    for img_idx, preds in enumerate(pred_boxes_list):
        for pred in preds:
            all_preds.append({
                'box': pred['box'],
                'conf': pred['conf'],
                'img_idx': img_idx
            })
    
    all_preds.sort(key=lambda x: x['conf'], reverse=True)
    gt_matched = [False] * len(gt_boxes_list)
    
    tp, fp = [], []
    for pred in all_preds:
        img_idx = pred['img_idx']
        pred_box = pred['box']
        
        if img_idx < len(gt_boxes_list) and gt_boxes_list[img_idx] is not None:
            gt_box = gt_boxes_list[img_idx]
            iou = bbox_iou(pred_box, gt_box)
            
            if iou >= iou_threshold and not gt_matched[img_idx]:
                tp.append(1)
                fp.append(0)
                gt_matched[img_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        else:
            tp.append(0)
            fp.append(1)
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    num_gt = sum(1 for gt in gt_boxes_list if gt is not None)
    if num_gt == 0:
        return 0.0
    
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


def calculate_map_range(pred_boxes_list, gt_boxes_list, iou_start=0.5, iou_end=0.95, iou_step=0.05):
    """Calculate mAP averaged over IoU thresholds."""
    iou_thresholds = np.arange(iou_start, iou_end + iou_step, iou_step)
    aps = [calculate_map(pred_boxes_list, gt_boxes_list, iou_threshold=t) for t in iou_thresholds]
    return np.mean(aps) if aps else 0.0


def calculate_map_multi(pred_boxes_list, gt_boxes_list, iou_threshold=0.5):
    """
    AP for multiple instances per image.
    pred_boxes_list[i] = [{'box': [x1,y1,x2,y2], 'conf': c}, ...]
    gt_boxes_list[i] = [box1, box2, ...] where box = [x1,y1,x2,y2], or None/[] for no GT.
    """
    all_preds = []
    for img_idx, preds in enumerate(pred_boxes_list):
        for p in preds:
            all_preds.append({'box': p['box'], 'conf': p['conf'], 'img_idx': img_idx})
    all_preds.sort(key=lambda x: x['conf'], reverse=True)

    gt_per_img = []
    for g in gt_boxes_list:
        if g is None or (isinstance(g, (list, tuple)) and len(g) == 0):
            gt_per_img.append([])
        elif isinstance(g, (list, tuple)) and len(g) > 0 and isinstance(g[0], (list, tuple)):
            gt_per_img.append([[float(x) for x in b] for b in g])
        else:
            gt_per_img.append([[float(x) for x in g]])
    gt_matched = [[False] * len(boxes) for boxes in gt_per_img]

    tp, fp = [], []
    for pred in all_preds:
        img_idx = pred['img_idx']
        pred_box = pred['box']
        if img_idx >= len(gt_per_img):
            tp.append(0)
            fp.append(1)
            continue
        gts = gt_per_img[img_idx]
        best_iou, best_j = -1.0, -1
        for j, gt_box in enumerate(gts):
            if gt_matched[img_idx][j]:
                continue
            iou = bbox_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j >= 0:
            tp.append(1)
            fp.append(0)
            gt_matched[img_idx][best_j] = True
        else:
            tp.append(0)
            fp.append(1)

    num_gt = sum(len(b) for b in gt_per_img)
    if num_gt == 0:
        return 0.0
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p = np.max(precisions[recalls >= t]) if np.sum(recalls >= t) > 0 else 0.0
        ap += p / 11.0
    return ap


def calculate_pck_percentages(accuracies_dict):
    """Calculate PCK percentages from accuracy dictionary.
    
    PCK uses vertical distance (Y-axis error only), converted to inches via PPI.
    
    Args:
        accuracies_dict: Dictionary with keys 'within_3_inch', 'within_2_inch', 
                        'within_1_inch', 'within_0_5_inch', 'total'
    
    Returns:
        Tuple of (pck_3inch, pck_2inch, pck_1inch, pck_0_5inch) percentages
    """
    total = accuracies_dict.get('total', 0)
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    pck_3inch = (accuracies_dict.get('within_3_inch', 0) / total) * 100
    pck_2inch = (accuracies_dict.get('within_2_inch', 0) / total) * 100
    pck_1inch = (accuracies_dict.get('within_1_inch', 0) / total) * 100
    pck_0_5inch = (accuracies_dict.get('within_0_5_inch', 0) / total) * 100
    
    return pck_3inch, pck_2inch, pck_1inch, pck_0_5inch


def calculate_precision_recall_f1(pred_boxes_list, gt_boxes_list, iou_threshold=0.5):
    """Calculate Precision, Recall, and F1-Score"""
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    
    for img_idx in range(len(gt_boxes_list)):
        gt_box = gt_boxes_list[img_idx]
        has_gt = gt_box is not None
        
        # Get predictions for this image
        preds = pred_boxes_list[img_idx] if img_idx < len(pred_boxes_list) else []
        has_pred = len(preds) > 0
        
        if has_gt and has_pred:
            # Calculate IoU with best prediction
            best_iou = max([bbox_iou(pred['box'], gt_box) for pred in preds])
            if best_iou >= iou_threshold:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif has_pred and not has_gt:
            fp += 1
        elif has_gt and not has_pred:
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision * 100, recall * 100, f1 * 100


def initialize_evaluation_metrics() -> Dict[str, Any]:
    """Initialize metrics dictionary for evaluation."""
    return {
        'pole_detection': {
            'iou_scores': [], 'detected_count': 0, 'gt_count': 0,
            'pred_boxes_list': [], 'gt_boxes_list': [],
            'valid_detections': 0, 'total_with_kp': 0
        },
        'ruler_detection': {
            'iou_scores': [], 'detected_count': 0, 'gt_count': 0,
            'pred_boxes_list': [], 'gt_boxes_list': [],
            'valid_detections': 0, 'total_with_kp': 0
        },
        'ruler_marking_keypoints': {
            'errors_by_height': defaultdict(list), 'errors_pixels': [],
            'accuracies': {  # per-instance (all markings within threshold)
                'within_3_inch': 0, 'within_2_inch': 0,
                'within_1_inch': 0, 'within_0_5_inch': 0, 'total': 0
            },
            'kp_accuracies': {  # per-keypoint (each marking independently)
                'within_3_inch': 0, 'within_2_inch': 0,
                'within_1_inch': 0, 'within_0_5_inch': 0, 'total': 0
            },
        },
        'pole_top_keypoint': {
            'errors_pixels': [],
            'accuracies': {
                'within_3_inch': 0, 'within_2_inch': 0,
                'within_1_inch': 0, 'within_0_5_inch': 0, 'total': 0
            }
        },
        'calibration_success': {
            'per_image_success': [], 'total_images': 0
        }
    }


def load_evaluation_ground_truth(
    img_path: Path,
    pole_labels_dir: Optional[Path],
    ruler_labels_dir: Optional[Path],
    ruler_marking_labels_dir: Optional[Path],
    pole_top_labels_dir: Optional[Path],
    location_files_dir: Path,
    img_width: int,
    img_height: int
) -> Dict[str, Any]:
    """
    Load all ground truth data for evaluation.
    
    Prioritizes location files (most accurate, global coordinates) over YOLO labels.
    Location files contain: pole bbox, ruler bbox, ruler marking keypoints, and pole top.
    
    Args:
        img_path: Path to image file
        pole_labels_dir: Directory with pole detection labels (None if not available)
        ruler_labels_dir: Directory with ruler detection labels (None if not available)
        ruler_marking_labels_dir: Directory with ruler marking labels (None if not available)
        pole_top_labels_dir: Directory with pole top labels (None if not available)
        location_files_dir: Directory with location files (global coordinates) - used as fallback
        img_width: Image width
        img_height: Image height
    
    Returns:
        Dictionary with ground truth data
    """
    base_name = img_path.stem
    # Create label paths only if directories are provided
    pole_label_path = pole_labels_dir / f'{base_name}.txt' if pole_labels_dir is not None else None
    ruler_label_path = ruler_labels_dir / f'{base_name}.txt' if ruler_labels_dir is not None else None
    ruler_marking_label_path = ruler_marking_labels_dir / f'{base_name}.txt' if ruler_marking_labels_dir is not None else None
    pole_top_label_path = pole_top_labels_dir / f'{base_name}.txt' if pole_top_labels_dir is not None else None
    
    # Location file path (contains all ground truth in global coordinates)
    # Prioritize the provided location_files_dir (correct dataset-specific directory)
    location_file_path = None
    
    # First, try the provided location_files_dir (dataset-specific)
    if location_files_dir:
        provided_location_path = location_files_dir / f'{base_name}_location.txt'
        if provided_location_path.exists():
            location_file_path = provided_location_path
    
    # Fallback to config paths if provided directory doesn't have the file
    # This is intentional: ruler test set contains both midspan and pole images
    # - Midspan images (*_Midspan_*): Use midspan GT
    # - Pole images (*_Main): Use pole GT (fallback)
    if location_file_path is None:
        location_paths = [
            POLE_LABELS_DIR / f"{base_name}_location.txt",
            MIDSPAN_LABELS_DIR / f"{base_name}_location.txt",
        ]

        # Try to find location file in either directory
        for loc_path in location_paths:
            if loc_path.exists():
                location_file_path = loc_path
                break
    
    # Initialize ground truth variables
    gt_pole_bbox = None
    gt_ruler_bbox = None
    gt_pole_top_global = None
    gt_keypoints_global = None
    
    # Load all ground truth from location file first (most accurate, global coordinates)
    if location_file_path and location_file_path.exists():
        # Load pole bbox from location file
        gt_pole_bbox = load_pole_bbox_from_location_file(location_file_path, img_width, img_height)
        
        # Load ruler bbox from location file
        gt_ruler_bbox = load_ruler_bbox_from_location_file(location_file_path, img_width, img_height)
        
        # Load ground truth pole top from location file (global coordinates)
        gt_pole_top_dict = load_pole_top_from_location_file(location_file_path, img_width, img_height)
        if gt_pole_top_dict:
            # Match inference.py: pole top coordinates are not clamped
            gt_pole_top_global = {
                'x': gt_pole_top_dict['x'],
                'y': gt_pole_top_dict['y']
            }
        
        # Load ground truth ruler marking keypoints from location file (global coordinates)
        gt_keypoints_dict = load_ruler_marking_keypoints_from_location_file(
            location_file_path, img_width, img_height
        )
        if gt_keypoints_dict:
            # Convert dictionary format and clamp coordinates (matching inference.py)
            gt_keypoints_global = {}
            for height, coords in sorted(gt_keypoints_dict.items()):
                # Clamp coordinates to ensure they're within image bounds (matching inference.py)
                x_coord = max(0, min(img_width - 1, int(round(coords['x']))))
                y_coord = max(0, min(img_height - 1, int(round(coords['y']))))
                gt_keypoints_global[height] = {
                    'x': x_coord,
                    'y': y_coord
                }
    
    # Fall back to YOLO labels if location file doesn't exist or didn't have the data
    # Load ground truth bounding boxes from YOLO labels (fallback)
    if gt_pole_bbox is None:
        gt_pole = load_yolo_label(pole_label_path) if pole_labels_dir is not None else None
        gt_pole_bbox = yolo_to_bbox(gt_pole, img_width, img_height) if gt_pole else None
    else:
        gt_pole = None  # Not needed if loaded from location file
    
    if gt_ruler_bbox is None:
        gt_ruler = load_yolo_label(ruler_label_path) if ruler_labels_dir is not None else None
        gt_ruler_bbox = yolo_to_bbox(gt_ruler, img_width, img_height) if gt_ruler else None
    else:
        gt_ruler = None  # Not needed if loaded from location file
    
    return {
        'gt_pole': gt_pole,
        'gt_ruler': gt_ruler,
        'gt_pole_bbox': gt_pole_bbox,
        'gt_ruler_bbox': gt_ruler_bbox,
        'gt_pole_top_global': gt_pole_top_global,
        'gt_keypoints_global': gt_keypoints_global,
        'pole_label_path': pole_label_path,
        'ruler_label_path': ruler_label_path,
        'ruler_marking_label_path': ruler_marking_label_path,
        'pole_top_label_path': pole_top_label_path,
        'location_file_path': location_file_path
    }


def run_evaluation_inference(
    img_bgr: np.ndarray,
    img_rgb: np.ndarray,
    pole_detector: Any,
    ruler_detector: Any,
    keypoint_model: Optional[Any],
    pole_top_model: Optional[Any],
    device: torch.device,
    use_tta: bool = True,
    img_path: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Run inference on a single image for evaluation.
    
    Args:
        img_bgr: BGR image array
        img_rgb: RGB image array
        pole_detector: YOLO pole detector model
        ruler_detector: YOLO ruler detector model
        keypoint_model: Keypoint detection model
        pole_top_model: Pole top detection model
        device: torch device
        use_tta: Use test-time augmentation
        img_path: Optional path to image file (Path or str). Used to determine if pole detection should be used.
                  If image is in pole detection dataset directory, pole and pole top detection will be performed.
                  Otherwise, pole and pole top detection will be skipped.
    
    Returns:
        Dictionary with predictions
    """
    img_height, img_width = img_rgb.shape[:2]
    
    # Determine if pole detection should be used based on dataset directory
    is_pole_detection_dataset = False
    if img_path is not None:
        img_path_obj = Path(img_path) if not isinstance(img_path, Path) else img_path
        pole_detection_dataset_dir = DATASETS_DIR / 'pole_detection' / 'images'
        
        # Convert to absolute paths for comparison
        try:
            img_path_abs = img_path_obj.resolve()
            pole_dir_abs = pole_detection_dataset_dir.resolve()
            
            # Check if image path starts with pole detection dataset directory
            # This handles both absolute and relative paths
            img_path_str = str(img_path_abs)
            pole_dir_str = str(pole_dir_abs)
            is_pole_detection_dataset = img_path_str.startswith(pole_dir_str)
        except (OSError, ValueError):
            # Fallback: check if path string contains the dataset directory
            img_path_str = str(img_path_obj)
            pole_dir_str = str(pole_detection_dataset_dir)
            is_pole_detection_dataset = pole_dir_str in img_path_str
    
    # Step 1: Detect pole on full image (only if image is from pole detection dataset)
    pole_res = None
    pred_pole_bbox = None
    if is_pole_detection_dataset and pole_detector is not None:
        pole_res = pole_detector(img_bgr, conf=INFERENCE_POLE_CONF_THRESHOLD, max_det=INFERENCE_MAX_DETECTIONS, verbose=False, imgsz=POLE_DETECTION_CONFIG['imgsz'])[0]
        if pole_res.boxes and len(pole_res.boxes) > 0:
            px1, py1, px2, py2 = pole_res.boxes.xyxy[0].cpu().numpy().astype(int)
            # Validate bounding box and clamp to image boundaries
            px1 = max(0, min(px1, img_width - 1))
            py1 = max(0, min(py1, img_height - 1))
            px2 = max(px1 + 1, min(px2, img_width))
            py2 = max(py1 + 1, min(py2, img_height))
            # Store validated bbox
            pred_pole_bbox = (px1, py1, px2, py2)
    
    # Step 2: Detect ruler on full image
    ruler_res = ruler_detector(img_bgr, conf=INFERENCE_RULER_CONF_THRESHOLD, max_det=INFERENCE_MAX_DETECTIONS, verbose=False, imgsz=RULER_DETECTION_CONFIG['imgsz'])[0]
    pred_ruler_bbox_full = None
    if ruler_res.boxes and len(ruler_res.boxes) > 0:
        rx1_full, ry1_full, rx2_full, ry2_full = ruler_res.boxes.xyxy[0].cpu().numpy().astype(int)
        # Validate bounding box and clamp to image boundaries
        rx1_full = max(0, min(rx1_full, img_width - 1))
        ry1_full = max(0, min(ry1_full, img_height - 1))
        rx2_full = max(rx1_full + 1, min(rx2_full, img_width))
        ry2_full = max(ry1_full + 1, min(ry2_full, img_height))
        # Store validated bbox
        pred_ruler_bbox_full = (rx1_full, ry1_full, rx2_full, ry2_full)
    
    # Step 3: Detect keypoints on ruler crop (if ruler detected)
    keypoints_pred_global = None
    if pred_ruler_bbox_full and keypoint_model is not None:
        rx1_full, ry1_full, rx2_full, ry2_full = pred_ruler_bbox_full
        keypoints_pred_crop = []
        if rx2_full > rx1_full and ry2_full > ry1_full:
            ruler_crop = img_rgb[ry1_full:ry2_full, rx1_full:rx2_full]
            if ruler_crop.size > 0:
                keypoints_pred_crop, _ = infer_keypoints_on_crop(keypoint_model, ruler_crop, device, use_tta=use_tta)
        # Convert to global coordinates
        keypoints_pred_global = []
        for kp in keypoints_pred_crop:
            keypoints_pred_global.append({
                'name': kp['name'],
                'x': kp['x'] + rx1_full,
                'y': kp['y'] + ry1_full,
                'conf': kp['conf']
            })
    
    # Step 4: Detect pole top on pole crop (if pole detected and image is from pole detection dataset)
    pred_pole_top_global = None
    pred_pole_top = None
    if is_pole_detection_dataset and pred_pole_bbox and pole_top_model is not None:
        px1, py1, px2, py2 = pred_pole_bbox
        # Bbox already validated in Step 1, but double-check
        # Only proceed if crop is valid and non-empty
        if px2 > px1 and py2 > py1:
            pole_crop_bgr = img_bgr[py1:py2, px1:px2]
            if pole_crop_bgr.size > 0:
                pole_crop_rgb = cv2.cvtColor(pole_crop_bgr, cv2.COLOR_BGR2RGB)
                pred_pole_top = infer_pole_top_on_crop(pole_top_model, pole_crop_rgb, device, use_tta=use_tta)
        if pred_pole_top and pred_pole_top['conf'] >= 0.01:
            # Transform pole top from pole crop coordinates to global coordinates
            pole_top_x_global = pred_pole_top['x'] + px1
            pole_top_y_global = pred_pole_top['y'] + py1
            
            # Validate that pole top is within pole bounding box and image bounds
            pole_top_x_global = max(px1, min(px2, pole_top_x_global))
            pole_top_y_global = max(py1, min(py2, pole_top_y_global))
            pole_top_x_global = max(0, min(img_width - 1, pole_top_x_global))
            pole_top_y_global = max(0, min(img_height - 1, pole_top_y_global))
            
            pred_pole_top_global = {
                'x': pole_top_x_global,
                'y': pole_top_y_global,
                'conf': pred_pole_top['conf']
            }
    
    return {
        'pred_pole_bbox': pred_pole_bbox,
        'pred_ruler_bbox_full': pred_ruler_bbox_full,
        'keypoints_pred_global': keypoints_pred_global,
        'pred_pole_top_global': pred_pole_top_global,
        'pole_res': pole_res,
        'ruler_res': ruler_res
    }


def evaluate_single_image(
    predictions: Dict[str, Any],
    ground_truth: Dict[str, Any],
    metrics: Dict[str, Any],
    counters: Dict[str, int]
) -> List[float]:
    """
    Evaluate predictions against ground truth for a single image and update metrics.
    
    Args:
        predictions: Dictionary with predictions from run_evaluation_inference()
        ground_truth: Dictionary with ground truth from load_evaluation_ground_truth()
        metrics: Metrics dictionary to update
        counters: Counters dictionary to update
    
    Returns:
        List of ruler marking errors for this image (for calibration success tracking)
    """
    current_image_ruler_errors = []
    
    gt_pole = ground_truth['gt_pole']
    gt_ruler = ground_truth['gt_ruler']
    gt_pole_bbox = ground_truth['gt_pole_bbox']
    gt_ruler_bbox = ground_truth['gt_ruler_bbox']
    gt_pole_top_global = ground_truth['gt_pole_top_global']
    gt_keypoints_global = ground_truth['gt_keypoints_global']
    
    pred_pole_bbox = predictions['pred_pole_bbox']
    pred_ruler_bbox_full = predictions['pred_ruler_bbox_full']
    keypoints_pred_global = predictions['keypoints_pred_global']
    pred_pole_top_global = predictions['pred_pole_top_global']
    pole_res = predictions['pole_res']
    ruler_res = predictions['ruler_res']
    
    # Evaluate pole detection - only if GT exists (check bbox, not YOLO label)
    if gt_pole_bbox is not None:
        metrics['pole_detection']['gt_count'] += 1
        metrics['pole_detection']['gt_boxes_list'].append(gt_pole_bbox)
        
        pred_pole_boxes = []
        if pred_pole_bbox:
            px1, py1, px2, py2 = pred_pole_bbox
            pred_pole = [px1, py1, px2, py2]
            counters['images_with_pole_detected'] += 1
            metrics['pole_detection']['detected_count'] += 1
            # Get confidence from pole_res if boxes exist
            conf = 0.0
            if pole_res and pole_res.boxes and len(pole_res.boxes) > 0:
                conf = float(pole_res.boxes.conf[0].cpu().numpy())
            pred_pole_boxes.append({'box': pred_pole, 'conf': conf})
            metrics['pole_detection']['iou_scores'].append(bbox_iou(pred_pole, gt_pole_bbox))
        
        metrics['pole_detection']['pred_boxes_list'].append(pred_pole_boxes)
    
    # Evaluate pole top
    if pred_pole_top_global:
        counters['images_with_pole_top_detected'] += 1

    if gt_pole_top_global:
        # Load PPI first — skip PCK entirely if unavailable (can't convert px to inches)
        pole_top_label_path = ground_truth['pole_top_label_path']
        location_file_path = ground_truth.get('location_file_path')
        ppi_crop = load_pole_top_ppi(pole_top_label_path) if pole_top_label_path is not None else None
        if not ppi_crop and location_file_path and location_file_path.exists():
            ppi_crop = load_pole_top_ppi(location_file_path)
        if ppi_crop and ppi_crop > 0:
            # Count every GT keypoint — missed predictions are failures
            metrics['pole_top_keypoint']['accuracies']['total'] += 1
            if pred_pole_top_global:
                error_px = abs(pred_pole_top_global['y'] - gt_pole_top_global['y'])
                metrics['pole_top_keypoint']['errors_pixels'].append(error_px)
                error_inch = error_px / ppi_crop
                threshold_map = {3.0: 'within_3_inch', 2.0: 'within_2_inch', 1.0: 'within_1_inch', 0.5: 'within_0_5_inch'}
                for threshold in [3.0, 2.0, 1.0, 0.5]:
                    if error_inch <= threshold:
                        metrics['pole_top_keypoint']['accuracies'][threshold_map[threshold]] += 1
    
    # Evaluate ruler detection - only if GT exists (check bbox, not YOLO label)
    if gt_ruler_bbox is not None:
        metrics['ruler_detection']['gt_count'] += 1
        metrics['ruler_detection']['gt_boxes_list'].append(gt_ruler_bbox)
        
        pred_ruler_boxes = []
        if pred_ruler_bbox_full:
            pred_ruler_full = list(pred_ruler_bbox_full)
            # Get confidence from ruler_res if boxes exist
            conf = 0.0
            if ruler_res and ruler_res.boxes and len(ruler_res.boxes) > 0:
                conf = float(ruler_res.boxes.conf[0].cpu().numpy())
            pred_ruler_boxes = [{'box': pred_ruler_full, 'conf': conf}]
            metrics['ruler_detection']['iou_scores'].append(bbox_iou(pred_ruler_full, gt_ruler_bbox))
            counters['images_with_ruler_detected'] += 1
            metrics['ruler_detection']['detected_count'] += 1
        
        metrics['ruler_detection']['pred_boxes_list'].append(pred_ruler_boxes)
    
    # Evaluate ruler keypoints
    if gt_keypoints_global:
        if keypoints_pred_global:
            counters['images_with_ruler_keypoints_detected'] += 1

        ruler_marking_label_path = ground_truth['ruler_marking_label_path']
        ruler_marking_ppi = None
        if ruler_marking_label_path is not None and ruler_marking_label_path.exists():
            with open(ruler_marking_label_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('# PPI='):
                        try:
                            ruler_marking_ppi = float(line.split('=')[1])
                            break
                        except (ValueError, IndexError):
                            pass

        # Build prediction lookup by height for O(1) matching
        pred_by_height = {float(kp['name']): kp for kp in keypoints_pred_global} if keypoints_pred_global else {}

        # Collect per-marking pixel errors (for mean/median stats)
        for height, kp_gt in gt_keypoints_global.items():
            kp_pred = pred_by_height.get(height)
            if kp_pred is not None:
                error_px = abs(kp_pred['y'] - kp_gt['y'])
                metrics['ruler_marking_keypoints']['errors_pixels'].append(error_px)
                metrics['ruler_marking_keypoints']['errors_by_height'][height].append(error_px)

        if not (ruler_marking_ppi and ruler_marking_ppi > 0):
            pass  # Skip PCK when PPI unavailable (can't convert px to inches)
        else:
            threshold_map = {3.0: 'within_3_inch', 2.0: 'within_2_inch', 1.0: 'within_1_inch', 0.5: 'within_0_5_inch'}
            kp_errors_in = []

            # Per-keypoint PCK: each GT marking evaluated independently (bar chart)
            for height, kp_gt in gt_keypoints_global.items():
                metrics['ruler_marking_keypoints']['kp_accuracies']['total'] += 1
                kp_pred = pred_by_height.get(height)
                if kp_pred is not None:
                    err_in = abs(kp_pred['y'] - kp_gt['y']) / ruler_marking_ppi
                    kp_errors_in.append(err_in)
                    for threshold in [3.0, 2.0, 1.0, 0.5]:
                        if err_in <= threshold:
                            metrics['ruler_marking_keypoints']['kp_accuracies'][threshold_map[threshold]] += 1

            # Per-instance PCK: ALL markings within threshold = successful calibration (pie chart)
            metrics['ruler_marking_keypoints']['accuracies']['total'] += 1
            if len(kp_errors_in) == len(gt_keypoints_global):
                for threshold in [3.0, 2.0, 1.0, 0.5]:
                    if all(e <= threshold for e in kp_errors_in):
                        metrics['ruler_marking_keypoints']['accuracies'][threshold_map[threshold]] += 1
                if all(e <= 1.0 for e in kp_errors_in):
                    current_image_ruler_errors.append(max(kp_errors_in))
    
    # Validate BB detections
    pole_top_label_path = ground_truth['pole_top_label_path']
    ruler_marking_label_path = ground_truth['ruler_marking_label_path']
    
    # For Pole: Check if pole top keypoint is inside pole bounding box
    if pole_top_label_path is not None and pole_top_label_path.exists() and gt_pole_top_global:
        metrics['pole_detection']['total_with_kp'] += 1
        if pred_pole_bbox:
            pole_top_x_px = gt_pole_top_global['x']
            pole_top_y_px = gt_pole_top_global['y']
            if is_point_in_bbox(pole_top_x_px, pole_top_y_px, pred_pole_bbox):
                metrics['pole_detection']['valid_detections'] += 1
    
    # For Ruler: Check if all GT ruler markings are inside predicted bbox
    if ruler_marking_label_path is not None and ruler_marking_label_path.exists() and gt_keypoints_global:
        metrics['ruler_detection']['total_with_kp'] += 1
        if pred_ruler_bbox_full:
            # Check if all GT markings are inside predicted bbox
            all_inside = True
            for height, kp_gt in gt_keypoints_global.items():
                if not is_point_in_bbox(kp_gt['x'], kp_gt['y'], pred_ruler_bbox_full):
                    all_inside = False
                    break
            if all_inside:
                metrics['ruler_detection']['valid_detections'] += 1
    
    return current_image_ruler_errors


def aggregate_metrics(
    aggregated_metrics: Dict[str, Any],
    metrics: Dict[str, Any]
) -> None:
    """
    Aggregate metrics from a dataset into aggregated metrics.
    
    Args:
        aggregated_metrics: Aggregated metrics dictionary to update
        metrics: Metrics dictionary from a single dataset
    """
    for key in ['pole_detection', 'ruler_detection']:
        aggregated_metrics[key]['iou_scores'].extend(metrics[key]['iou_scores'])
        aggregated_metrics[key]['detected_count'] += metrics[key]['detected_count']
        aggregated_metrics[key]['gt_count'] += metrics[key]['gt_count']
        aggregated_metrics[key]['pred_boxes_list'].extend(metrics[key]['pred_boxes_list'])
        aggregated_metrics[key]['gt_boxes_list'].extend(metrics[key]['gt_boxes_list'])
        aggregated_metrics[key]['valid_detections'] += metrics[key]['valid_detections']
        aggregated_metrics[key]['total_with_kp'] += metrics[key]['total_with_kp']
    
    aggregated_metrics['ruler_marking_keypoints']['errors_pixels'].extend(
        metrics['ruler_marking_keypoints']['errors_pixels']
    )
    for height, errors in metrics['ruler_marking_keypoints']['errors_by_height'].items():
        aggregated_metrics['ruler_marking_keypoints']['errors_by_height'][height].extend(errors)
    
    for key in ['within_3_inch', 'within_2_inch', 'within_1_inch', 'within_0_5_inch', 'total']:
        aggregated_metrics['ruler_marking_keypoints']['accuracies'][key] += \
            metrics['ruler_marking_keypoints']['accuracies'][key]
        aggregated_metrics['ruler_marking_keypoints']['kp_accuracies'][key] += \
            metrics['ruler_marking_keypoints']['kp_accuracies'][key]
    
    aggregated_metrics['pole_top_keypoint']['errors_pixels'].extend(
        metrics['pole_top_keypoint']['errors_pixels']
    )
    for key in ['within_3_inch', 'within_2_inch', 'within_1_inch', 'within_0_5_inch', 'total']:
        aggregated_metrics['pole_top_keypoint']['accuracies'][key] += \
            metrics['pole_top_keypoint']['accuracies'][key]
    
    aggregated_metrics['calibration_success']['per_image_success'].extend(
        metrics['calibration_success']['per_image_success']
    )
    aggregated_metrics['calibration_success']['total_images'] += \
        metrics['calibration_success']['total_images']


def _infer_keypoints_from_detections(
    img_bgr: np.ndarray,
    img_rgb: np.ndarray,
    pole_res: Any,
    ruler_res: Any,
    keypoint_model: Optional[Any],
    pole_top_model: Optional[Any],
    device: torch.device,
    use_tta: bool = True,
    img_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run keypoint inference given pre-computed YOLO pole/ruler results.

    Mirrors run_evaluation_inference but skips YOLO detection (uses pre-computed results).
    """
    img_height, img_width = img_rgb.shape[:2]

    # Determine pole dataset membership (same logic as run_evaluation_inference)
    is_pole_detection_dataset = False
    if img_path is not None:
        img_path_obj = Path(img_path) if not isinstance(img_path, Path) else img_path
        pole_detection_dataset_dir = DATASETS_DIR / 'pole_detection' / 'images'
        try:
            is_pole_detection_dataset = str(img_path_obj.resolve()).startswith(
                str(pole_detection_dataset_dir.resolve())
            )
        except (OSError, ValueError):
            is_pole_detection_dataset = str(pole_detection_dataset_dir) in str(img_path_obj)

    # Extract pole bbox from pre-computed YOLO result
    pred_pole_bbox = None
    if is_pole_detection_dataset and pole_res is not None and pole_res.boxes and len(pole_res.boxes) > 0:
        px1, py1, px2, py2 = pole_res.boxes.xyxy[0].cpu().numpy().astype(int)
        px1 = max(0, min(px1, img_width - 1))
        py1 = max(0, min(py1, img_height - 1))
        px2 = max(px1 + 1, min(px2, img_width))
        py2 = max(py1 + 1, min(py2, img_height))
        pred_pole_bbox = (px1, py1, px2, py2)

    # Extract ruler bbox from pre-computed YOLO result
    pred_ruler_bbox_full = None
    if ruler_res is not None and ruler_res.boxes and len(ruler_res.boxes) > 0:
        rx1, ry1, rx2, ry2 = ruler_res.boxes.xyxy[0].cpu().numpy().astype(int)
        rx1 = max(0, min(rx1, img_width - 1))
        ry1 = max(0, min(ry1, img_height - 1))
        rx2 = max(rx1 + 1, min(rx2, img_width))
        ry2 = max(ry1 + 1, min(ry2, img_height))
        pred_ruler_bbox_full = (rx1, ry1, rx2, ry2)

    # Keypoints on ruler crop
    keypoints_pred_global = None
    if pred_ruler_bbox_full and keypoint_model is not None:
        rx1, ry1, rx2, ry2 = pred_ruler_bbox_full
        keypoints_pred_crop = []
        if rx2 > rx1 and ry2 > ry1:
            ruler_crop = img_rgb[ry1:ry2, rx1:rx2]
            if ruler_crop.size > 0:
                keypoints_pred_crop, _ = infer_keypoints_on_crop(keypoint_model, ruler_crop, device, use_tta=use_tta)
        keypoints_pred_global = [
            {'name': kp['name'], 'x': kp['x'] + rx1, 'y': kp['y'] + ry1, 'conf': kp['conf']}
            for kp in keypoints_pred_crop
        ]

    # Pole top on pole crop
    pred_pole_top_global = None
    if is_pole_detection_dataset and pred_pole_bbox and pole_top_model is not None:
        px1, py1, px2, py2 = pred_pole_bbox
        if px2 > px1 and py2 > py1:
            pole_crop_bgr = img_bgr[py1:py2, px1:px2]
            if pole_crop_bgr.size > 0:
                pole_crop_rgb = cv2.cvtColor(pole_crop_bgr, cv2.COLOR_BGR2RGB)
                pred_pole_top = infer_pole_top_on_crop(pole_top_model, pole_crop_rgb, device, use_tta=use_tta)
                if pred_pole_top and pred_pole_top['conf'] >= 0.01:
                    pole_top_x = max(0, min(img_width - 1, max(px1, min(px2, pred_pole_top['x'] + px1))))
                    pole_top_y = max(0, min(img_height - 1, max(py1, min(py2, pred_pole_top['y'] + py1))))
                    pred_pole_top_global = {'x': pole_top_x, 'y': pole_top_y, 'conf': pred_pole_top['conf']}

    return {
        'pred_pole_bbox': pred_pole_bbox,
        'pred_ruler_bbox_full': pred_ruler_bbox_full,
        'keypoints_pred_global': keypoints_pred_global,
        'pred_pole_top_global': pred_pole_top_global,
        'pole_res': pole_res,
        'ruler_res': ruler_res,
    }


def run_dataset_evaluation(dataset_config: Dict[str, Path], pole_detector, ruler_detector,
                          keypoint_model, pole_top_model, device, use_tta: bool,
                          batch_size: int = EVALUATION_YOLO_BATCH_SIZE) -> Dict[str, Any]:
    """
    Run evaluation on a dataset and return results with failure tracking.

    Optimized: concurrent image+GT loading, batched YOLO pole/ruler detection,
    then sequential per-image keypoint inference.

    Args:
        dataset_config: Configuration dict with images_dir, pole_labels_dir, etc.
        pole_detector, ruler_detector, keypoint_model, pole_top_model: Loaded models
        device: torch device
        use_tta: Whether to use test-time augmentation
        batch_size: Number of images per YOLO inference batch

    Returns:
        Dictionary with metrics, images_processed, and failure lists
    """
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor

    test_images = sorted(dataset_config['images_dir'].glob('*.jpg'))
    metrics = initialize_evaluation_metrics()
    counters, ppis = defaultdict(int), []
    failed_pole_detection, failed_pole_top = [], []
    failed_ruler_detection, failed_ruler_marking = [], []

    if not test_images:
        return {
            'metrics': metrics, 'images_processed': 0, 'ppis': ppis,
            'failed_pole_detection': [], 'failed_pole_top': [],
            'failed_ruler_detection': [], 'failed_ruler_marking': [],
        }

    # Determine once whether this dataset uses pole detection
    try:
        pole_dir_abs = str((DATASETS_DIR / 'pole_detection' / 'images').resolve())
        is_pole_dataset = str(test_images[0].resolve()).startswith(pole_dir_abs)
    except (OSError, ValueError):
        is_pole_dataset = str(DATASETS_DIR / 'pole_detection' / 'images') in str(test_images[0])

    # Step 1: Load GT only (images loaded on-demand to avoid memory spike)
    def _load_gt_only(img_path):
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                return None
            h, w = img_bgr.shape[:2]
            gt = load_evaluation_ground_truth(
                img_path, dataset_config.get('pole_labels_dir'),
                dataset_config.get('ruler_labels_dir'),
                dataset_config.get('ruler_marking_labels_dir'),
                dataset_config.get('pole_top_labels_dir'),
                dataset_config.get('location_files_dir'), w, h,
            )
            return h, w, gt
        except Exception:
            return None

    print("  Loading ground truth...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        gt_data = list(tqdm(pool.map(_load_gt_only, test_images),
                            total=len(test_images), desc="  Loading GT", leave=False))

    valid = [(p, d) for p, d in zip(test_images, gt_data) if d is not None]
    if not valid:
        return {
            'metrics': metrics, 'images_processed': 0, 'ppis': ppis,
            'failed_pole_detection': [], 'failed_pole_top': [],
            'failed_ruler_detection': [], 'failed_ruler_marking': [],
        }

    valid_paths = [p for p, _ in valid]
    valid_gt_data = [d for _, d in valid]

    # Step 2: Batched YOLO inference (load images just-in-time for batch)
    def _batch_yolo_jit(detector, img_paths, conf, imgsz, desc):
        """YOLO batching with just-in-time image loading to reduce memory."""
        results_all = []
        for i in tqdm(range(0, len(img_paths), batch_size), desc=desc, leave=False):
            batch_paths = img_paths[i:i + batch_size]
            batch = []
            for p in batch_paths:
                img_bgr = cv2.imread(str(p))
                if img_bgr is not None:
                    batch.append(img_bgr)
                else:
                    batch.append(None)
            # Filter out None entries and run YOLO
            valid_batch = [img for img in batch if img is not None]
            if valid_batch:
                res = detector(valid_batch, conf=conf, max_det=INFERENCE_MAX_DETECTIONS,
                               verbose=False, imgsz=imgsz)
                # Reconstruct with None placeholders for failed loads
                j = 0
                for img in batch:
                    if img is not None:
                        r = res[j]
                        r.orig_img = None  # Drop stored image; only .boxes needed downstream
                        results_all.append(r)
                        j += 1
                    else:
                        results_all.append(None)
                del res  # Free the batch result list
            else:
                results_all.extend([None] * len(batch_paths))
            # Explicitly free batch memory
            del batch, valid_batch
        return results_all

    if is_pole_dataset:
        print("  Running pole detection (batched, memory-optimized)...")
        pole_results_all = _batch_yolo_jit(
            pole_detector, valid_paths, INFERENCE_POLE_CONF_THRESHOLD,
            POLE_DETECTION_CONFIG['imgsz'], "  Pole detection",
        )
    else:
        pole_results_all = [None] * len(valid)

    print("  Running ruler detection (batched, memory-optimized)...")
    ruler_results_all = _batch_yolo_jit(
        ruler_detector, valid_paths, INFERENCE_RULER_CONF_THRESHOLD,
        RULER_DETECTION_CONFIG['imgsz'], "  Ruler detection",
    )

    # Step 3: Keypoint inference + evaluation (sequential, load images on-demand)
    for idx, (img_path, gt_data_item) in enumerate(tqdm(valid, desc="  Keypoints & eval")):
        h, w, gt = gt_data_item

        # Load image just-in-time for this single image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pole_res = pole_results_all[idx]
        ruler_res = ruler_results_all[idx]

        pred = _infer_keypoints_from_detections(
            img_bgr, img_rgb, pole_res, ruler_res,
            keypoint_model, pole_top_model, device, use_tta, img_path,
        )

        evaluate_single_image(pred, gt, metrics, counters)

        # Track pole detection failures (pole dataset)
        if gt['gt_pole_top_global'] and pred['pred_pole_bbox']:
            if not is_point_in_bbox(gt['gt_pole_top_global']['x'], gt['gt_pole_top_global']['y'], pred['pred_pole_bbox']):
                failed_pole_detection.append(img_path.name)

        # Track pole top detection failures (pole dataset)
        _pole_top_lbl = gt['pole_top_label_path']
        if _pole_top_lbl is not None and _pole_top_lbl.exists() and gt['gt_pole_top_global'] and pred['pred_pole_top_global']:
            error_px = abs(pred['pred_pole_top_global']['y'] - gt['gt_pole_top_global']['y'])
            ppi = load_pole_top_ppi(_pole_top_lbl)
            if ppi and error_px / ppi > 3.0:
                failed_pole_top.append(img_path.name)

        # Track ruler detection failures (ruler dataset): GT markings not in pred bbox
        if gt['gt_ruler_bbox'] and gt['gt_keypoints_global'] and pred['pred_ruler_bbox_full']:
            all_inside = all(
                is_point_in_bbox(kp['x'], kp['y'], pred['pred_ruler_bbox_full'])
                for kp in gt['gt_keypoints_global'].values()
            )
            if not all_inside:
                failed_ruler_detection.append(img_path.name)

        # Track ruler marking failures (ruler dataset): keypoint error > 3"
        _rm_lbl = gt['ruler_marking_label_path']
        if _rm_lbl is not None and _rm_lbl.exists() and gt['gt_keypoints_global'] and pred.get('keypoints_pred_global'):
            ruler_ppi = None
            with open(_rm_lbl, 'r') as f:
                    for line in f:
                        if line.strip().startswith('# PPI='):
                            try:
                                ruler_ppi = float(line.split('=')[1])
                                break
                            except (ValueError, IndexError):
                                pass
            if ruler_ppi and ruler_ppi > 0:
                for kp_pred in pred['keypoints_pred_global']:
                    height = float(kp_pred['name'])
                    if height in gt['gt_keypoints_global']:
                        kp_gt = gt['gt_keypoints_global'][height]
                        error_px = abs(kp_pred['y'] - kp_gt['y'])
                        if error_px / ruler_ppi > 3.0:
                            failed_ruler_marking.append(img_path.name)
                            break

        # Track PPI
        _pole_top_lbl = gt['pole_top_label_path']
        _rm_lbl = gt['ruler_marking_label_path']
        if _pole_top_lbl is not None and _pole_top_lbl.exists():
            ppi = load_pole_top_ppi(_pole_top_lbl)
            if ppi:
                ppis.append(ppi)
        elif _rm_lbl is not None and _rm_lbl.exists():
            with open(_rm_lbl, 'r') as f:
                for line in f:
                    if line.strip().startswith('# PPI='):
                        try:
                            ppis.append(float(line.split('=')[1]))
                            break
                        except (ValueError, IndexError):
                            pass

        counters['total_images_processed'] += 1

    return {
        'metrics': metrics,
        'images_processed': counters['total_images_processed'],
        'ppis': ppis,
        'failed_pole_detection': failed_pole_detection,
        'failed_pole_top': failed_pole_top,
        'failed_ruler_detection': failed_ruler_detection,
        'failed_ruler_marking': failed_ruler_marking,
    }


def load_calibration_evaluation_models(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Load all models needed for calibration pipeline evaluation.

    Returns:
        Dict with pole_detector, ruler_detector, keypoint_model, pole_top_model
    """
    from ultralytics import YOLO

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pole_detector = YOLO(str(INFERENCE_POLE_WEIGHTS))
    ruler_detector = YOLO(str(INFERENCE_RULER_WEIGHTS))
    keypoint_model = load_trained_keypoint_model(str(INFERENCE_RULER_MARKING_WEIGHTS), device=device)
    pole_top_model = load_pole_top_model(str(INFERENCE_POLE_TOP_WEIGHTS), device=device)

    return {
        'pole_detector': pole_detector,
        'ruler_detector': ruler_detector,
        'keypoint_model': keypoint_model,
        'pole_top_model': pole_top_model,
        'device': device,
    }


def run_full_evaluation(
    dataset_name: str,
    dataset_config: Optional[Dict[str, Path]] = None,
    models: Optional[Dict[str, Any]] = None,
    results_dir: Optional[Path] = None,
    use_tta: bool = True,
) -> Dict[str, Any]:
    """Run full evaluation for a calibration dataset: inference, metrics, save JSON.

    Args:
        dataset_name: 'pole_detection' or 'ruler_detection'
        dataset_config: Dataset config dict; default from EVALUATION_DATASETS_CONFIG
        models: Dict from load_calibration_evaluation_models(); loads if None
        results_dir: Where to save JSON; default RESULTS_CALIBRATION_DIR
        use_tta: Use test-time augmentation for keypoint models

    Returns:
        Results dict (also saved to JSON)
    """
    if dataset_config is None:
        dataset_config = EVALUATION_DATASETS_CONFIG[dataset_name]
    if results_dir is None:
        results_dir = RESULTS_CALIBRATION_DIR
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = load_calibration_evaluation_models()
    device = models['device']

    result = run_dataset_evaluation(
        dataset_config,
        models['pole_detector'],
        models['ruler_detector'],
        models['keypoint_model'],
        models['pole_top_model'],
        device,
        use_tta,
    )

    metrics = result['metrics']
    ppis = result['ppis']
    avg_ppi = float(np.mean(ppis)) if ppis else None
    errors_pixels = metrics['pole_top_keypoint']['errors_pixels']
    errors_inch = [e / avg_ppi for e in errors_pixels] if avg_ppi and avg_ppi > 0 else []

    if dataset_name == 'pole_detection':
        det_metrics = metrics['pole_detection']
        kp_metrics = metrics['pole_top_keypoint']
        # Pole top is a single keypoint — per-keypoint == per-instance
        pck_3, pck_2, pck_1, pck_05 = calculate_pck_percentages(kp_metrics['accuracies'])

        iou_mean = float(np.mean(det_metrics['iou_scores'])) if det_metrics['iou_scores'] else 0.0
        map_50 = calculate_map(det_metrics['pred_boxes_list'], det_metrics['gt_boxes_list'], iou_threshold=0.5)
        map_range = calculate_map_range(det_metrics['pred_boxes_list'], det_metrics['gt_boxes_list'])
        valid_rate = (det_metrics['valid_detections'] / det_metrics['total_with_kp'] * 100) if det_metrics['total_with_kp'] > 0 else 0.0

        results = {
            'dataset': dataset_name,
            'evaluation_date': datetime.now().isoformat(),
            'images_processed': result['images_processed'],
            'pole_detection': {
                'detected_count': det_metrics['detected_count'],
                'gt_count': det_metrics['gt_count'],
                'mean_iou': iou_mean,
                'map_0_5': map_50,
                'map_0_5_to_0_95': map_range,
                'valid_rate_percent': valid_rate,
                'failed_images': result['failed_pole_detection'],
            },
            'pole_top_detection': {
                'total_keypoints': kp_metrics['accuracies']['total'],
                'pck_3_inch': pck_3, 'pck_2_inch': pck_2, 'pck_1_inch': pck_1, 'pck_0_5_inch': pck_05,
                'instance_pck_3_inch': pck_3, 'instance_pck_2_inch': pck_2,
                'instance_pck_1_inch': pck_1, 'instance_pck_0_5_inch': pck_05,
                'mean_error_pixels': float(np.mean(errors_pixels)) if errors_pixels else 0.0,
                'mean_error_inches': float(np.mean(errors_inch)) if errors_inch else None,
                'median_error_inches': float(np.median(errors_inch)) if errors_inch else None,
                'std_error_inches': float(np.std(errors_inch)) if errors_inch else None,
                'avg_ppi': avg_ppi,
                'failed_images': result['failed_pole_top'],
            },
        }
    else:
        det_metrics = metrics['ruler_detection']
        rm_kp = metrics['ruler_marking_keypoints']
        # Per-keypoint PCK (bar chart): each marking evaluated independently
        pck_3, pck_2, pck_1, pck_05 = calculate_pck_percentages(rm_kp['kp_accuracies'])
        # Per-instance PCK (pie chart): ALL markings within threshold
        inst_pck_3, inst_pck_2, inst_pck_1, inst_pck_05 = calculate_pck_percentages(rm_kp['accuracies'])

        iou_mean = float(np.mean(det_metrics['iou_scores'])) if det_metrics['iou_scores'] else 0.0
        map_50 = calculate_map(det_metrics['pred_boxes_list'], det_metrics['gt_boxes_list'], iou_threshold=0.5)
        map_range = calculate_map_range(det_metrics['pred_boxes_list'], det_metrics['gt_boxes_list'])
        valid_rate = (det_metrics['valid_detections'] / det_metrics['total_with_kp'] * 100) if det_metrics['total_with_kp'] > 0 else 0.0

        rm_errors = rm_kp['errors_pixels']
        rm_errors_inch = [e / avg_ppi for e in rm_errors] if avg_ppi and avg_ppi > 0 else []

        results = {
            'dataset': dataset_name,
            'evaluation_date': datetime.now().isoformat(),
            'images_processed': result['images_processed'],
            'ruler_detection': {
                'detected_count': det_metrics['detected_count'],
                'gt_count': det_metrics['gt_count'],
                'mean_iou': iou_mean,
                'map_0_5': map_50,
                'map_0_5_to_0_95': map_range,
                'valid_rate_percent': valid_rate,
                'failed_images': result.get('failed_ruler_detection', result['failed_pole_detection']),
            },
            'ruler_marking_detection': {
                'total_keypoints': rm_kp['kp_accuracies'].get('total', 0),
                'total_instances': rm_kp['accuracies'].get('total', 0),
                'pck_3_inch': pck_3, 'pck_2_inch': pck_2, 'pck_1_inch': pck_1, 'pck_0_5_inch': pck_05,
                'instance_pck_3_inch': inst_pck_3, 'instance_pck_2_inch': inst_pck_2,
                'instance_pck_1_inch': inst_pck_1, 'instance_pck_0_5_inch': inst_pck_05,
                'mean_error_pixels': float(np.mean(rm_errors)) if rm_errors else 0.0,
                'mean_error_inches': float(np.mean(rm_errors_inch)) if rm_errors_inch else None,
                'median_error_inches': float(np.median(rm_errors_inch)) if rm_errors_inch else None,
                'std_error_inches': float(np.std(rm_errors_inch)) if rm_errors_inch else None,
                'avg_ppi': avg_ppi,
                'failed_images': result.get('failed_ruler_marking', result['failed_pole_top']),
            },
        }

    # Save as pole_detection.json / ruler_detection.json for consistency
    results_path = results_dir / f'{dataset_name}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {results_path}")

    return results


def generate_evaluation_plots(results_dir: Path = None) -> None:
    """
    Generate evaluation charts from JSON results using centralized chart style.
    Creates pole_detection.png and ruler_detection.png in results_dir.
    """
    if results_dir is None:
        results_dir = RESULTS_CALIBRATION_DIR
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    from .evaluation_charts import generate_all_calibration_charts
    generate_all_calibration_charts(results_dir)


def get_evaluation_dataset_config(case_type: str) -> Dict[str, Path]:
    """Get dataset config for a given failure case type."""
    return EVALUATION_DATASETS_CONFIG[
        'pole_detection' if case_type in ('pole_detection', 'pole_top_detection') else 'ruler_detection'
    ]


def visualize_failed_case(case_type: str = 'pole_detection', dataset_config: Optional[Dict[str, Path]] = None,
                         results_dir: Optional[Path] = None) -> None:
    """
    Visualize a random failed case from evaluation results.

    Args:
        case_type: 'pole_detection', 'pole_top_detection', 'ruler_detection', or 'ruler_marking_detection'
        dataset_config: Configuration dict; default from EVALUATION_DATASETS_CONFIG
        results_dir: Directory containing evaluation JSON files; default RESULTS_CALIBRATION_DIR
    """
    import json
    import random
    import matplotlib.pyplot as plt

    if dataset_config is None:
        dataset_config = get_evaluation_dataset_config(case_type)
    if results_dir is None:
        results_dir = RESULTS_CALIBRATION_DIR
    results_dir = Path(results_dir)

    # Map case types to result files and keys
    case_type_mapping = {
        'pole_detection': ('pole_detection.json', 'pole_detection'),
        'pole_top_detection': ('pole_detection.json', 'pole_top_detection'),
        'ruler_detection': ('ruler_detection.json', 'ruler_detection'),
        'ruler_marking_detection': ('ruler_detection.json', 'ruler_marking_detection'),
    }

    if case_type not in case_type_mapping:
        print(f"Unknown case type: {case_type}")
        return

    results_file, results_key = case_type_mapping[case_type]
    results_path = results_dir / results_file
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    failed_images = results.get(results_key, {}).get('failed_images', [])
    if not failed_images:
        print(f"No failed {case_type} cases found")
        return

    # Select random failed case
    random_failed_name = random.choice(failed_images)
    img_path = dataset_config['images_dir'] / random_failed_name

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Could not load image: {img_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Load GT
    gt = load_evaluation_ground_truth(
        img_path, dataset_config.get('pole_labels_dir'),
        dataset_config.get('ruler_labels_dir'),
        dataset_config.get('ruler_marking_labels_dir'),
        dataset_config.get('pole_top_labels_dir'),
        dataset_config.get('location_files_dir'), w, h
    )

    # Display results
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img_rgb)
    ax.set_title(f'Failed {case_type.replace("_", " ").title()}: {random_failed_name}', fontsize=14, fontweight='bold')

    # Draw GT bbox (centralized colors from config)
    gt_pole_bbox = gt.get('gt_pole_bbox')
    if gt_pole_bbox:
        x1, y1, x2, y2 = gt_pole_bbox
        c_gt = np.array(COLOR_GT) / 255.0
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=c_gt, facecolor='none')
        ax.add_patch(rect)
        ax.text(int(x1), int(y1) - 5, 'GT Pole BB', color=c_gt, fontsize=10, fontweight='bold')

    # Draw GT pole top
    gt_pole_top = gt.get('gt_pole_top_global')
    if gt_pole_top:
        x, y = int(gt_pole_top['x']), int(gt_pole_top['y'])
        c_pred = np.array(COLOR_PRED) / 255.0
        ax.plot(x, y, 'o', color=c_pred, markersize=8, label='GT Pole Top')
        ax.text(x + 10, y, "GT Pole Top", color=c_pred, fontsize=9)

    ax.legend(loc='upper right')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    print(f"✓ Displayed failed case: {random_failed_name}")
