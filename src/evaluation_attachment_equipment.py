"""
Evaluation for attachment (comm, down_guy, primary, secondary, neutral, guy) and equipment
(street_light, transformer, riser, secondary_drip_loop).
Produces JSON per detection class and uses centralized chart generation.
All evaluation runs on the end-to-end inference pipeline (pole -> crop -> detect -> keypoints).

Detection-only evaluation (attachment_detection): runs YOLO val, threshold sweep, updates config,
produces attachment_detection.json and attachment_detection.png with 2x2 chart.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Tuple
from tqdm import tqdm

from .config import (
    PROJECT_ROOT,
    EQUIPMENT_CLASS_NAMES,
    ATTACHMENT_CLASS_NAMES,
    ATTACHMENT_EVALUATION_CONFIG,
    EQUIPMENT_EVALUATION_CONFIG,
    RESULTS_ATTACHMENT_DIR,
    RESULTS_EQUIPMENT_DIR,
    ATTACHMENT_DETECTION_CONFIG,
    EQUIPMENT_DETECTION_CONFIG,
    INFERENCE_ATTACHMENT_CONF_THRESHOLD,
    INFERENCE_EQUIPMENT_CONF_THRESHOLD,
    INFERENCE_EQUIPMENT_CONF_PER_CLASS,
    INFERENCE_SECONDARY_DRIP_LOOP_MAX_DET,
    INFERENCE_POLE_CONF_THRESHOLD,
    INFERENCE_USE_TTA,
    INFERENCE_ATTACHMENT_WEIGHTS,
    INFERENCE_EQUIPMENT_WEIGHTS,
    ATTACHMENT_KEYPOINT_CONFIGS,
    EQUIPMENT_KEYPOINT_CONFIGS,
    EQUIPMENT_E2E_IMAGES_DIR,
    ATTACHMENT_E2E_IMAGES_DIR,
    POLE_LABELS_DIR,
    E2E_USE_TEST_SPLIT_ONLY,
)
from .config import EQUIPMENT_DATASET_DIR, ATTACHMENT_DATASET_DIR
from .threshold_utils import (
    YOLO_DETECTION_MODELS,
    run_threshold_sweep,
    run_val_on_split,
    update_attachment_config,
    update_equipment_config,
)
from .evaluation_utils import (
    bbox_iou,
    calculate_map,
    calculate_map_range,
    calculate_map_multi,
    calculate_pck_percentages,
    is_point_in_bbox,
)
from .visualization import parse_yolo_label, parse_keypoint_label
from .data_utils import (
    parse_equipment_with_keypoints,
    parse_attachments_with_keypoints,
    load_ppi_from_label,
    get_e2e_test_images,
)


def _load_attachment_models(device):
    """Load attachment detector and keypoint models."""
    from ultralytics import YOLO
    from .inference import load_attachment_keypoint_detector

    detector = YOLO(str(INFERENCE_ATTACHMENT_WEIGHTS))
    kp_models = {}
    for name in ATTACHMENT_KEYPOINT_CONFIGS:
        try:
            kp_models[name] = load_attachment_keypoint_detector(name, device)
        except (FileNotFoundError, ValueError):
            pass
    return detector, kp_models


def _load_equipment_models(device):
    """Load equipment detector and keypoint models."""
    from ultralytics import YOLO
    from .inference import load_keypoint_detector

    detector = YOLO(str(INFERENCE_EQUIPMENT_WEIGHTS))
    kp_models = {}
    for name in EQUIPMENT_KEYPOINT_CONFIGS:
        try:
            kp_models[name] = load_keypoint_detector(name, device)
        except (FileNotFoundError, ValueError):
            pass
    return detector, kp_models


def _load_pole_detector():
    """Load pole detector for E2E pipeline."""
    from ultralytics import YOLO
    from .config import INFERENCE_POLE_WEIGHTS
    return YOLO(str(INFERENCE_POLE_WEIGHTS))


def _extract_equipment_crop(img, pole_bbox: Tuple[int, int, int, int]):
    """Extract upper 70% 2:5 crop from pole bbox. Returns (crop_bgr, (x1,y1,x2,y2))."""
    x1, y1, x2, y2 = pole_bbox
    crop_h_full = y2 - y1
    crop_h = int(crop_h_full * 0.7)
    if crop_h < 10 or (x2 - x1) < 10:
        return None, None
    target_width = int(crop_h * (2 / 5))
    center_x = (x1 + x2) / 2
    x1_new = max(0, int(center_x - target_width / 2))
    x2_new = min(img.shape[1], int(center_x + target_width / 2))
    if x2_new - x1_new < 10:
        return None, None
    crop = img[y1 : y1 + crop_h, x1_new:x2_new]
    return crop, (x1_new, y1, x2_new, y1 + crop_h)


def _detect_and_kp_on_crop(
    crop_bgr,
    img_rgb,
    crop_x1: int,
    crop_y1: int,
    detector,
    conf_thresh: float,
    imgsz: int,
    class_names: List[str],
    kp_models: Dict,
    device,
) -> List[Dict]:
    """Run detector + keypoint models on a pre-extracted crop. Returns preds in full-image coords."""
    import torch

    results = detector(crop_bgr, conf=conf_thresh, max_det=20, verbose=False, imgsz=imgsz)[0]
    preds = []
    if results.boxes is None or len(results.boxes) == 0:
        return preds
    for i in range(len(results.boxes)):
        bbox = results.boxes.xyxy[i].cpu().numpy()
        conf = float(results.boxes.conf[i].cpu().numpy())
        cls_id = int(results.boxes.cls[i].cpu().numpy())
        cls_name = class_names[cls_id] if cls_id < len(class_names) else 'unknown'
        ex1, ey1, ex2, ey2 = map(int, bbox)
        x1_full = crop_x1 + ex1
        y1_full = crop_y1 + ey1
        x2_full = crop_x1 + ex2
        y2_full = crop_y1 + ey2
        det = {
            'cls_id': cls_id,
            'cls_name': cls_name,
            'bbox': (x1_full, y1_full, x2_full, y2_full),
            'conf': conf,
            'keypoints': [],
        }
        det_crop = img_rgb[y1_full:y2_full, x1_full:x2_full]
        if det_crop.shape[0] >= 10 and det_crop.shape[1] >= 10 and cls_name in kp_models:
            kp_cfg = kp_models[cls_name]
            model = kp_cfg['model']
            kp_names = kp_cfg['kp_names']
            tensor = kp_cfg['preprocess'](det_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                heatmaps = torch.sigmoid(model(tensor))[0].detach().cpu().numpy()
            det_h, det_w = det_crop.shape[:2]
            for idx, hm in enumerate(heatmaps):
                y_int, x_int = np.unravel_index(np.argmax(hm), hm.shape)
                conf_kp = float(hm[y_int, x_int])
                y_sub, x_sub = float(y_int), float(x_int)
                x_px = x1_full + x_sub / max(hm.shape[1] - 1, 1) * (det_w - 1) if det_w > 1 else x1_full
                y_px = y1_full + y_sub / max(hm.shape[0] - 1, 1) * (det_h - 1) if det_h > 1 else y1_full
                det['keypoints'].append({'name': kp_names[idx], 'x': x_px, 'y': y_px, 'conf': conf_kp})
        preds.append(det)

    # Apply max detection limits per class
    # Limit secondary_drip_loop to max 1 (keep highest conf)
    sdl_preds = [d for d in preds if d["cls_name"] == "secondary_drip_loop"]
    if len(sdl_preds) > INFERENCE_SECONDARY_DRIP_LOOP_MAX_DET:
        sdl_preds.sort(key=lambda d: d["conf"], reverse=True)
        keep = {id(d) for d in sdl_preds[:INFERENCE_SECONDARY_DRIP_LOOP_MAX_DET]}
        preds = [d for d in preds if d["cls_name"] != "secondary_drip_loop" or id(d) in keep]

    return preds


def _run_e2e_single_image(
    img_path: Path,
    pole_detector,
    detector,
    kp_models: Dict,
    device,
    conf_thresh: float,
    imgsz: int,
    class_names: List[str],
    pole_conf: float = INFERENCE_POLE_CONF_THRESHOLD,
) -> Tuple[List[Dict], Optional[float], int, int]:
    """
    Run E2E pipeline on one image. Returns list of preds per instance:
    [{cls_id, cls_name, bbox, conf, keypoints: [{name, x, y, conf}]}, ...]
    Also returns PPI and image dimensions (h, w) from location file if available.
    """
    import cv2

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return [], None, 0, 0
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]

    lbl_path = POLE_LABELS_DIR / f"{img_path.stem}_location.txt"
    ppi = load_ppi_from_label(lbl_path) if lbl_path.exists() else None

    pole_res = pole_detector(img_bgr, conf=pole_conf, max_det=1, verbose=False, imgsz=960)[0]
    if pole_res.boxes is None or len(pole_res.boxes) == 0:
        return [], ppi, h_img, w_img
    px1, py1, px2, py2 = map(int, pole_res.boxes.xyxy[0].cpu().numpy())

    crop, crop_bounds = _extract_equipment_crop(img_bgr, (px1, py1, px2, py2))
    if crop is None:
        return [], ppi, h_img, w_img
    crop_x1, crop_y1 = crop_bounds[0], crop_bounds[1]

    preds = _detect_and_kp_on_crop(
        crop, img_rgb, crop_x1, crop_y1,
        detector, conf_thresh, imgsz, class_names, kp_models, device,
    )
    return preds, ppi, h_img, w_img


def run_e2e_annotation_single_image(
    img_path: Path,
    pole_detector,
    equip_detector,
    attach_detector,
    equip_kp_models: Dict,
    attach_kp_models: Dict,
    device,
    pole_conf: float = INFERENCE_POLE_CONF_THRESHOLD,
) -> Tuple[List[Dict], List[Dict], Optional[float]]:
    """
    Run combined equipment + attachment pipeline on one image.
    Optimized: loads image once, runs pole detection once, extracts crop once,
    then runs both equipment and attachment detectors on the same crop.

    Returns:
        (equip_preds, attach_preds, ppi)
    """
    import cv2
    import torch

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return [], [], None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    lbl_path = POLE_LABELS_DIR / f"{img_path.stem}_location.txt"
    ppi = load_ppi_from_label(lbl_path) if lbl_path.exists() else None

    pole_res = pole_detector(img_bgr, conf=pole_conf, max_det=1, verbose=False, imgsz=960)[0]
    if pole_res.boxes is None or len(pole_res.boxes) == 0:
        return [], [], ppi
    px1, py1, px2, py2 = map(int, pole_res.boxes.xyxy[0].cpu().numpy())
    pole_bbox = (px1, py1, px2, py2)

    crop_bgr, crop_bounds = _extract_equipment_crop(img_bgr, pole_bbox)
    if crop_bgr is None:
        return [], [], ppi
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds

    # Equipment detection on shared crop
    # Use min(per_class) so low-threshold classes (riser, street_light) are not filtered at detection time
    base_conf = min(INFERENCE_EQUIPMENT_CONF_PER_CLASS.values()) if INFERENCE_EQUIPMENT_CONF_PER_CLASS else INFERENCE_EQUIPMENT_CONF_THRESHOLD
    equip_results = equip_detector(
        crop_bgr,
        conf=base_conf,
        max_det=20,
        verbose=False,
        imgsz=EQUIPMENT_DETECTION_CONFIG['imgsz'],
    )[0]

    equip_preds = []
    if equip_results.boxes is not None and len(equip_results.boxes) > 0:
        for i in range(len(equip_results.boxes)):
            bbox = equip_results.boxes.xyxy[i].cpu().numpy()
            conf = float(equip_results.boxes.conf[i].cpu().numpy())
            cls_id = int(equip_results.boxes.cls[i].cpu().numpy())
            cls_name = EQUIPMENT_CLASS_NAMES[cls_id] if cls_id < len(EQUIPMENT_CLASS_NAMES) else 'unknown'
            cls_thresh = INFERENCE_EQUIPMENT_CONF_PER_CLASS.get(cls_name, INFERENCE_EQUIPMENT_CONF_THRESHOLD)
            if conf < cls_thresh:
                continue  # Filter by per-class threshold
            ex1, ey1, ex2, ey2 = map(int, bbox)
            x1_full = crop_x1 + ex1
            y1_full = crop_y1 + ey1
            x2_full = crop_x1 + ex2
            y2_full = crop_y1 + ey2
            det = {'cls_id': cls_id, 'cls_name': cls_name, 'bbox': (x1_full, y1_full, x2_full, y2_full), 'conf': conf, 'keypoints': []}
            eq_crop = img_rgb[y1_full:y2_full, x1_full:x2_full]
            if eq_crop.shape[0] >= 10 and eq_crop.shape[1] >= 10 and cls_name in equip_kp_models:
                kp_cfg = equip_kp_models[cls_name]
                model = kp_cfg['model']
                kp_names = kp_cfg['kp_names']
                tensor = kp_cfg['preprocess'](eq_crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    heatmaps = torch.sigmoid(model(tensor))[0].detach().cpu().numpy()
                eq_h, eq_w = eq_crop.shape[:2]
                for idx, hm in enumerate(heatmaps):
                    y_int, x_int = np.unravel_index(np.argmax(hm), hm.shape)
                    conf_kp = float(hm[y_int, x_int])
                    y_sub, x_sub = float(y_int), float(x_int)
                    x_px = x1_full + x_sub / max(hm.shape[1] - 1, 1) * (eq_w - 1) if eq_w > 1 else x1_full
                    y_px = y1_full + y_sub / max(hm.shape[0] - 1, 1) * (eq_h - 1) if eq_h > 1 else y1_full
                    det['keypoints'].append({'name': kp_names[idx], 'x': x_px, 'y': y_px, 'conf': conf_kp})
            equip_preds.append(det)

    # Limit secondary_drip_loop to max 1 (keep highest conf)
    sdl_preds = [d for d in equip_preds if d["cls_name"] == "secondary_drip_loop"]
    if len(sdl_preds) > INFERENCE_SECONDARY_DRIP_LOOP_MAX_DET:
        sdl_preds.sort(key=lambda d: d["conf"], reverse=True)
        keep = {id(d) for d in sdl_preds[:INFERENCE_SECONDARY_DRIP_LOOP_MAX_DET]}
        equip_preds = [d for d in equip_preds if d["cls_name"] != "secondary_drip_loop" or id(d) in keep]

    # Attachment detection on same shared crop
    attach_results = attach_detector(
        crop_bgr,
        conf=INFERENCE_ATTACHMENT_CONF_THRESHOLD,
        max_det=20,
        verbose=False,
        imgsz=ATTACHMENT_DETECTION_CONFIG['imgsz'],
    )[0]

    attach_preds = []
    if attach_results.boxes is not None and len(attach_results.boxes) > 0:
        for i in range(len(attach_results.boxes)):
            bbox = attach_results.boxes.xyxy[i].cpu().numpy()
            conf = float(attach_results.boxes.conf[i].cpu().numpy())
            cls_id = int(attach_results.boxes.cls[i].cpu().numpy())
            cls_name = ATTACHMENT_CLASS_NAMES[cls_id] if cls_id < len(ATTACHMENT_CLASS_NAMES) else 'unknown'
            ex1, ey1, ex2, ey2 = map(int, bbox)
            x1_full, y1_full = crop_x1 + ex1, crop_y1 + ey1
            x2_full, y2_full = crop_x1 + ex2, crop_y1 + ey2
            det = {'cls_id': cls_id, 'cls_name': cls_name, 'bbox': (x1_full, y1_full, x2_full, y2_full), 'conf': conf, 'keypoints': []}
            attach_crop_rgb = img_rgb[y1_full:y2_full, x1_full:x2_full]
            if attach_crop_rgb.shape[0] >= 10 and attach_crop_rgb.shape[1] >= 10 and cls_name in attach_kp_models:
                kp_cfg = attach_kp_models[cls_name]
                model = kp_cfg['model']
                kp_names = kp_cfg['kp_names']
                tensor = kp_cfg['preprocess'](attach_crop_rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    heatmaps = torch.sigmoid(model(tensor))[0].detach().cpu().numpy()
                eq_h, eq_w = attach_crop_rgb.shape[:2]
                for idx, hm in enumerate(heatmaps):
                    y_int, x_int = np.unravel_index(np.argmax(hm), hm.shape)
                    conf_kp = float(hm[y_int, x_int])
                    y_sub, x_sub = float(y_int), float(x_int)
                    x_px = x1_full + x_sub / max(hm.shape[1] - 1, 1) * (eq_w - 1) if eq_w > 1 else x1_full
                    y_px = y1_full + y_sub / max(hm.shape[0] - 1, 1) * (eq_h - 1) if eq_h > 1 else y1_full
                    det['keypoints'].append({'name': kp_names[idx], 'x': x_px, 'y': y_px, 'conf': conf_kp})
            attach_preds.append(det)

    return equip_preds, attach_preds, ppi


def run_detection_class_evaluation(
    eval_config: Dict,
    detector,
    domain: str,
    device,
) -> Dict[str, Any]:
    """
    Run bbox evaluation for a single class (e.g. comm, riser).
    Filters GT and predictions by class_id.
    """
    import cv2

    images_dir = Path(eval_config['images_dir'])
    labels_dir = Path(eval_config['labels_dir'])
    class_id = eval_config['class_id']
    class_name = eval_config['class_name']

    config = ATTACHMENT_DETECTION_CONFIG if domain == 'attachment' else EQUIPMENT_DETECTION_CONFIG
    conf_thresh = INFERENCE_ATTACHMENT_CONF_THRESHOLD if domain == 'attachment' else INFERENCE_EQUIPMENT_CONF_THRESHOLD

    test_images = sorted(images_dir.glob('*.jpg'))
    pred_boxes_list, gt_boxes_list = [], []
    iou_scores = []
    valid_detections, total_with_kp = 0, 0

    for img_path in tqdm(test_images, desc=f"Evaluating {class_name}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_path = labels_dir / f'{img_path.stem}.txt'
        gt_boxes = parse_yolo_label(lbl_path, w, h)
        gt_class = [b for b in gt_boxes if b['class_id'] == class_id]
        if not gt_class:
            pred_boxes_list.append([])
            gt_boxes_list.append(None)
            continue

        gt_bbox = gt_class[0]['bbox']

        results = detector(img, conf=conf_thresh, max_det=20, verbose=False, imgsz=config['imgsz'])[0]
        pred_boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            for i in range(len(results.boxes)):
                cls_id = int(results.boxes.cls[i].cpu().numpy())
                if cls_id != class_id:
                    continue
                bbox = results.boxes.xyxy[i].cpu().numpy()
                conf = float(results.boxes.conf[i].cpu().numpy())
                pred_boxes.append({'box': list(map(int, bbox)), 'conf': conf})

        pred_boxes_list.append(pred_boxes)
        gt_boxes_list.append(gt_bbox)

        if gt_bbox and pred_boxes:
            best_iou = max(bbox_iou(p['box'], gt_bbox) for p in pred_boxes)
            iou_scores.append(best_iou)
            total_with_kp += 1
            cx, cy = (gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2
            if any(is_point_in_bbox(cx, cy, p['box']) for p in pred_boxes):
                valid_detections += 1

    map_50 = calculate_map(pred_boxes_list, gt_boxes_list, iou_threshold=0.5)
    map_range = calculate_map_range(pred_boxes_list, gt_boxes_list)
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    valid_rate = (valid_detections / total_with_kp * 100) if total_with_kp > 0 else 0.0
    gt_count = sum(1 for g in gt_boxes_list if g is not None)
    detected_count = sum(1 for p, g in zip(pred_boxes_list, gt_boxes_list) if g is not None and p)

    return {
        'detected_count': detected_count,
        'gt_count': gt_count,
        'mean_iou': mean_iou,
        'map_0_5': map_50,
        'map_0_5_to_0_95': map_range,
        'valid_rate_percent': valid_rate,
    }


def run_keypoint_class_evaluation(
    eval_config: Dict,
    kp_model_dict: Optional[Dict],
    device,
    class_name: str,
) -> Dict[str, Any]:
    """
    Run keypoint evaluation on keypoint dataset test set (crops).
    """
    import torch
    import cv2

    empty_result = {
        'pck_3_inch': 0, 'pck_2_inch': 0, 'pck_1_inch': 0, 'pck_0_5_inch': 0,
        'mean_error_inches': None, 'median_error_inches': None, 'std_error_inches': None,
        'total_keypoints': 0,
    }

    kp_dataset = Path(eval_config.get('keypoint_dataset', ''))
    if not kp_dataset.exists():
        return empty_result

    test_dir = kp_dataset / 'images' / 'test'
    test_lbl = kp_dataset / 'labels' / 'test'
    if not test_dir.exists() or not test_lbl.exists():
        return empty_result

    if not kp_model_dict:
        return empty_result

    model = kp_model_dict['model']
    preprocess = kp_model_dict['preprocess']
    num_kp = kp_model_dict['num_kp']

    test_images = sorted(test_dir.glob('*.jpg'))
    errors_px = []
    accuracies = {'within_3_inch': 0, 'within_2_inch': 0, 'within_1_inch': 0, 'within_0_5_inch': 0, 'total': 0}

    PPI_FALLBACK = 50.0

    for img_path in tqdm(test_images, desc=f"Keypoint {class_name}"):
        lbl_path = test_lbl / f'{img_path.stem}.txt'
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        gt_kps = parse_keypoint_label(lbl_path, w, h)
        if len(gt_kps) < num_kp:
            continue

        ppi = PPI_FALLBACK
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                if line.strip().startswith('# PPI='):
                    try:
                        ppi = float(line.split('=')[1].strip())
                    except (ValueError, IndexError):
                        pass
                    break

        tensor = preprocess(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        heatmaps = torch.sigmoid(logits)[0].cpu().numpy()

        for idx in range(min(num_kp, len(heatmaps), len(gt_kps))):
            gt = gt_kps[idx]
            hm = heatmaps[idx]
            y_int, x_int = np.unravel_index(np.argmax(hm), hm.shape)
            y_sub, x_sub = float(y_int), float(x_int)
            pred_x = x_sub / max(hm.shape[1] - 1, 1) * (w - 1) if w > 1 else x_sub
            pred_y = y_sub / max(hm.shape[0] - 1, 1) * (h - 1) if h > 1 else y_sub
            err_px = abs(pred_y - gt['y'])  # Vertical distance (PCK uses vertical error)
            errors_px.append(err_px)
            err_in = err_px / ppi if ppi > 0 else err_px
            accuracies['total'] += 1
            for thresh, key in [(3.0, 'within_3_inch'), (2.0, 'within_2_inch'), (1.0, 'within_1_inch'), (0.5, 'within_0_5_inch')]:
                if err_in <= thresh:
                    accuracies[key] += 1

    pck_3, pck_2, pck_1, pck_05 = calculate_pck_percentages(accuracies)
    errors_inch = [e / PPI_FALLBACK for e in errors_px] if errors_px else []
    return {
        'pck_3_inch': pck_3, 'pck_2_inch': pck_2, 'pck_1_inch': pck_1, 'pck_0_5_inch': pck_05,
        'mean_error_inches': float(np.mean(errors_inch)) if errors_inch else None,
        'median_error_inches': float(np.median(errors_inch)) if errors_inch else None,
        'std_error_inches': float(np.std(errors_inch)) if errors_inch else None,
        'mean_error_pixels': float(np.mean(errors_px)) if errors_px else None,
        'total_keypoints': accuracies['total'],
    }


def _equipment_gt_normalizer(inst: Dict, num_kp: int) -> Dict:
    """Normalize equipment GT instance to common format.

    Returns {'bbox_pct': (left, top, right, bottom), 'kps_pct': [(x, y), ...]}.
    Coordinates are percentages (0-100).
    """
    bbox = inst['bbox']
    kps = [inst.get(f'kp{i}') for i in range(num_kp)]
    return {
        'bbox_pct': (bbox['left'], bbox['top'], bbox['right'], bbox['bottom']),
        'kps_pct': [kp for kp in kps if kp is not None],
    }


def _attachment_gt_normalizer(inst: Dict) -> Optional[Dict]:
    """Normalize attachment GT instance to common format.

    Returns None if the instance has no center keypoint (skip it entirely).
    Returns {'bbox_pct': (left, top, right, bottom), 'kps_pct': [(x, y)]}.
    Coordinates are percentages (0-100).
    """
    if not inst.get('center'):
        return None
    return {
        'bbox_pct': (inst['left'], inst['top'], inst['right'], inst['bottom']),
        'kps_pct': [inst['center']],
    }


def _run_all_inference(
    images: List[Path],
    pole_detector,
    detector,
    kp_models: Dict,
    device,
    conf_thresh: float,
    imgsz: int,
    class_names: List[str],
    gt_parser,
    desc: str,
) -> List[Dict]:
    """Single-pass E2E inference for all images. Returns cached results list."""
    results = []
    for img_path in tqdm(images, desc=desc):
        preds, ppi, h, w = _run_e2e_single_image(
            img_path, pole_detector, detector, kp_models, device,
            conf_thresh, imgsz, class_names,
        )
        lbl_path = POLE_LABELS_DIR / f"{img_path.stem}_location.txt"
        gt = gt_parser(lbl_path) if lbl_path.exists() else []
        results.append({'preds': preds, 'ppi': ppi, 'h': h, 'w': w, 'gt': gt})
    return results


def _compute_class_metrics(
    cached_results: List[Dict],
    class_name: str,
    gt_normalizer: Callable,
) -> Dict[str, Any]:
    """Compute detection + keypoint metrics for one class from cached E2E inference.

    gt_normalizer: callable(inst) → {'bbox_pct': ..., 'kps_pct': ...} or None
    """
    PPI_FALLBACK = 7.0

    n = len(cached_results)
    pred_boxes_list = [[] for _ in range(n)]
    gt_boxes_list = [[] for _ in range(n)]
    iou_scores = []
    _thresh_keys = [(3.0, 'within_3_inch'), (2.0, 'within_2_inch'), (1.0, 'within_1_inch'), (0.5, 'within_0_5_inch')]
    kp_acc = {'within_3_inch': 0, 'within_2_inch': 0, 'within_1_inch': 0, 'within_0_5_inch': 0, 'total': 0}
    instance_acc = {'within_3_inch': 0, 'within_2_inch': 0, 'within_1_inch': 0, 'within_0_5_inch': 0, 'total': 0}
    errors_px = []
    valid_bb_count = 0  # GT instances where any pred bbox encloses ALL GT keypoints

    for img_idx, r in enumerate(cached_results):
        h, w = r['h'], r['w']
        if h == 0 or w == 0:
            continue
        ppi_val = r['ppi'] if r['ppi'] and r['ppi'] > 0 else PPI_FALLBACK

        gt_instances_raw = [inst for inst in r['gt'] if inst['class_name'] == class_name]
        gt_instances = []
        gt_normalized = []
        for inst in gt_instances_raw:
            norm = gt_normalizer(inst)
            if norm is not None:
                gt_instances.append(inst)
                gt_normalized.append(norm)

        gt_boxes_list[img_idx] = [
            [n['bbox_pct'][0]/100*w, n['bbox_pct'][1]/100*h,
             n['bbox_pct'][2]/100*w, n['bbox_pct'][3]/100*h]
            for n in gt_normalized
        ] if gt_normalized else []

        preds_class = [p for p in r['preds'] if p['cls_name'] == class_name]
        pred_boxes_list[img_idx] = [{'box': list(p['bbox']), 'conf': p['conf']} for p in preds_class]

        for norm in gt_normalized:
            if not norm['kps_pct']:
                continue  # no keypoints annotated for this instance

            gt_bbox_px = [
                norm['bbox_pct'][0]/100*w, norm['bbox_pct'][1]/100*h,
                norm['bbox_pct'][2]/100*w, norm['bbox_pct'][3]/100*h,
            ]
            kps_px = [(kp[0]/100*w, kp[1]/100*h) for kp in norm['kps_pct']]

            best_pred, best_iou = None, -1
            for p in preds_class:
                iou = bbox_iou(p['bbox'], gt_bbox_px)
                if iou > best_iou:
                    best_iou, best_pred = iou, p

            matched = best_pred is not None and best_iou >= 0.5
            if matched:
                iou_scores.append(best_iou)

            # Valid BB: any predicted bbox encloses ALL GT keypoints
            if any(all(is_point_in_bbox(kx, ky, p['bbox']) for kx, ky in kps_px) for p in preds_class):
                valid_bb_count += 1

            # PCK: missed detection = all keypoints of this instance fail
            instance_acc['total'] += 1
            kp_acc['total'] += len(kps_px)
            if matched and best_pred.get('keypoints'):
                kp_errors_in = []
                all_evaluated = True
                for ki, (gt_x, gt_y) in enumerate(kps_px):
                    if ki >= len(best_pred['keypoints']):
                        all_evaluated = False
                        break
                    pred_kp = best_pred['keypoints'][ki]
                    err_px = abs(pred_kp['y'] - gt_y)  # Vertical distance (PCK uses vertical error)
                    errors_px.append(err_px)
                    err_in = err_px / ppi_val if ppi_val > 0 else err_px
                    kp_errors_in.append(err_in)
                    # Per-keypoint: each keypoint evaluated independently
                    for thresh, key in _thresh_keys:
                        if err_in <= thresh:
                            kp_acc[key] += 1
                if all_evaluated and kp_errors_in:
                    # Per-instance: ALL keypoints must be within threshold
                    for thresh, key in _thresh_keys:
                        if all(e <= thresh for e in kp_errors_in):
                            instance_acc[key] += 1

    map_50 = calculate_map_multi(pred_boxes_list, gt_boxes_list, iou_threshold=0.5)
    map_range = float(np.mean([
        calculate_map_multi(pred_boxes_list, gt_boxes_list, iou_threshold=t)
        for t in np.arange(0.5, 1.0, 0.05)
    ])) if any(gt_boxes_list) else 0.0
    gt_count = sum(len(g) for g in gt_boxes_list)
    # Pad with zeros for missed GT instances so mean_iou reflects overall recall, not just matched pairs
    mean_iou = float(np.mean(iou_scores + [0.0] * (gt_count - len(iou_scores)))) if gt_count > 0 else 0.0
    # GT instances with at least one prediction matched at IoU >= 0.5
    detected_count = sum(
        1 for i, g_list in enumerate(gt_boxes_list)
        for g in g_list
        if any(bbox_iou(p['box'], g) >= 0.5 for p in pred_boxes_list[i])
    )
    pred_count = sum(len(p) for p in pred_boxes_list)
    valid_rate = 100.0 * valid_bb_count / max(1, gt_count)
    # Precision, Recall, F1 at IoU >= 0.5
    precision = detected_count / max(1, pred_count)
    recall = detected_count / max(1, gt_count)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    # Per-keypoint PCK (bar chart): each individual keypoint evaluated independently
    pck_3, pck_2, pck_1, pck_05 = calculate_pck_percentages(kp_acc)
    # Per-instance PCK (pie chart): ALL keypoints must be within threshold
    inst_pck_3, inst_pck_2, inst_pck_1, inst_pck_05 = calculate_pck_percentages(instance_acc)
    errors_inch = [e / PPI_FALLBACK for e in errors_px] if errors_px else []

    return {
        'detection': {
            'detected_count': detected_count,
            'pred_count': pred_count,
            'gt_count': int(gt_count),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_iou': mean_iou,
            'map_0_5': map_50,
            'map_0_5_to_0_95': map_range,
            'valid_rate_percent': valid_rate,
        },
        'keypoint': {
            'pck_3_inch': pck_3, 'pck_2_inch': pck_2, 'pck_1_inch': pck_1, 'pck_0_5_inch': pck_05,
            'instance_pck_3_inch': inst_pck_3, 'instance_pck_2_inch': inst_pck_2,
            'instance_pck_1_inch': inst_pck_1, 'instance_pck_0_5_inch': inst_pck_05,
            'mean_error_inches': float(np.mean(errors_inch)) if errors_inch else None,
            'median_error_inches': float(np.median(errors_inch)) if errors_inch else None,
            'std_error_inches': float(np.std(errors_inch)) if errors_inch else None,
            'total_keypoints': kp_acc['total'],
            'total_instances': instance_acc['total'],
        },
    }


def _run_all_inference_combined(
    images: List[Path],
    pole_detector,
    equip_detector,
    attach_detector,
    equip_kp_models: Dict,
    attach_kp_models: Dict,
    device,
) -> Tuple[List[Dict], List[Dict]]:
    """Single pole-detection pass yielding cached results for both equipment and attachment."""
    import cv2

    equip_cached, attach_cached = [], []
    for img_path in tqdm(images, desc="E2E combined inference"):
        img_bgr = cv2.imread(str(img_path))
        empty_equip = {'preds': [], 'ppi': None, 'h': 0, 'w': 0, 'gt': []}
        empty_attach = {'preds': [], 'ppi': None, 'h': 0, 'w': 0, 'gt': []}
        if img_bgr is None:
            equip_cached.append(empty_equip)
            attach_cached.append(empty_attach)
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = img_rgb.shape[:2]

        lbl_path = POLE_LABELS_DIR / f"{img_path.stem}_location.txt"
        ppi = load_ppi_from_label(lbl_path) if lbl_path.exists() else None
        equip_gt = parse_equipment_with_keypoints(lbl_path) if lbl_path.exists() else []
        attach_gt = parse_attachments_with_keypoints(lbl_path) if lbl_path.exists() else []

        base = {'ppi': ppi, 'h': h_img, 'w': w_img}

        pole_res = pole_detector(img_bgr, conf=INFERENCE_POLE_CONF_THRESHOLD, max_det=1, verbose=False, imgsz=960)[0]
        if pole_res.boxes is None or len(pole_res.boxes) == 0:
            equip_cached.append({**base, 'preds': [], 'gt': equip_gt})
            attach_cached.append({**base, 'preds': [], 'gt': attach_gt})
            continue

        px1, py1, px2, py2 = map(int, pole_res.boxes.xyxy[0].cpu().numpy())
        crop, crop_bounds = _extract_equipment_crop(img_bgr, (px1, py1, px2, py2))
        if crop is None:
            equip_cached.append({**base, 'preds': [], 'gt': equip_gt})
            attach_cached.append({**base, 'preds': [], 'gt': attach_gt})
            continue

        crop_x1, crop_y1 = crop_bounds[0], crop_bounds[1]

        equip_preds = _detect_and_kp_on_crop(
            crop, img_rgb, crop_x1, crop_y1,
            equip_detector, INFERENCE_EQUIPMENT_CONF_THRESHOLD,
            EQUIPMENT_DETECTION_CONFIG['imgsz'], EQUIPMENT_CLASS_NAMES, equip_kp_models, device,
        )
        attach_preds = _detect_and_kp_on_crop(
            crop, img_rgb, crop_x1, crop_y1,
            attach_detector, INFERENCE_ATTACHMENT_CONF_THRESHOLD,
            ATTACHMENT_DETECTION_CONFIG['imgsz'], ATTACHMENT_CLASS_NAMES, attach_kp_models, device,
        )

        equip_cached.append({**base, 'preds': equip_preds, 'gt': equip_gt})
        attach_cached.append({**base, 'preds': attach_preds, 'gt': attach_gt})

    return equip_cached, attach_cached


def run_combined_evaluation(
    equip_results_dir=None,
    attach_results_dir=None,
    device=None,
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Run equipment + attachment E2E evaluation in a single pole-detection pass.
    Also runs attachment detection-only evaluation (threshold sweep, config update).
    Returns (equip_results, attach_results) — same dicts as the individual functions.
    """
    import torch

    # Run detection-only evals first (sweep, config update, *_detection.json)
    run_attachment_detection_evaluation(attach_results_dir or RESULTS_ATTACHMENT_DIR)
    run_equipment_detection_evaluation(equip_results_dir or RESULTS_EQUIPMENT_DIR)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if equip_results_dir is None:
        equip_results_dir = RESULTS_EQUIPMENT_DIR
    if attach_results_dir is None:
        attach_results_dir = RESULTS_ATTACHMENT_DIR
    equip_results_dir = Path(equip_results_dir)
    attach_results_dir = Path(attach_results_dir)
    equip_results_dir.mkdir(parents=True, exist_ok=True)
    attach_results_dir.mkdir(parents=True, exist_ok=True)

    pole_detector = _load_pole_detector()
    equip_detector, equip_kp_models = _load_equipment_models(device)
    attach_detector, attach_kp_models = _load_attachment_models(device)

    # Both domains share the same image set (same source dir + same test stems)
    images = get_e2e_test_images('equipment')

    equip_cached, attach_cached = _run_all_inference_combined(
        images, pole_detector, equip_detector, attach_detector,
        equip_kp_models, attach_kp_models, device,
    )

    def gt_normalizer_equip(inst):
        num_kp = EQUIPMENT_KEYPOINT_CONFIGS.get(inst['class_name'], (None, 2, None))[1]
        return _equipment_gt_normalizer(inst, num_kp)

    equip_all = {}
    for name, cfg in EQUIPMENT_EVALUATION_CONFIG.items():
        res = _compute_class_metrics(equip_cached, cfg['class_name'], gt_normalizer_equip)
        out = {
            'dataset': name,
            'evaluation_date': datetime.now().isoformat(),
            'pipeline': 'end_to_end',
            'evaluation_split': 'test' if E2E_USE_TEST_SPLIT_ONLY else 'all',
            'images_dir': str(EQUIPMENT_E2E_IMAGES_DIR),
            'images_evaluated': len(images),
            'gt_instance_count': res['detection']['gt_count'],
            'detection': res['detection'],
            'keypoint': res['keypoint'],
        }
        path = equip_results_dir / f'{name}.json'
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"✓ {name} saved to {path}")
        equip_all[name] = out

    attach_all = {}
    for name, cfg in ATTACHMENT_EVALUATION_CONFIG.items():
        res = _compute_class_metrics(attach_cached, cfg['class_name'], _attachment_gt_normalizer)
        out = {
            'dataset': name,
            'evaluation_date': datetime.now().isoformat(),
            'pipeline': 'end_to_end',
            'evaluation_split': 'test' if E2E_USE_TEST_SPLIT_ONLY else 'all',
            'images_dir': str(ATTACHMENT_E2E_IMAGES_DIR),
            'images_evaluated': len(images),
            'gt_instance_count': res['detection']['gt_count'],
            'detection': res['detection'],
            'keypoint': res['keypoint'],
        }
        path = attach_results_dir / f'{name}.json'
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"✓ {name} saved to {path}")
        attach_all[name] = out

    return equip_all, attach_all


def _run_domain_evaluation(domain: str, results_dir, device) -> Dict[str, Dict]:
    """Shared evaluation driver for equipment and attachment domains."""
    import torch

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if domain == 'equipment':
        if results_dir is None:
            results_dir = RESULTS_EQUIPMENT_DIR
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        pole_detector = _load_pole_detector()
        detector, kp_models = _load_equipment_models(device)
        images = get_e2e_test_images('equipment')
        cached = _run_all_inference(
            images, pole_detector, detector, kp_models, device,
            conf_thresh=INFERENCE_EQUIPMENT_CONF_THRESHOLD,
            imgsz=EQUIPMENT_DETECTION_CONFIG['imgsz'],
            class_names=EQUIPMENT_CLASS_NAMES,
            gt_parser=parse_equipment_with_keypoints,
            desc="E2E equipment inference",
        )
        eval_config = EQUIPMENT_EVALUATION_CONFIG
        images_dir_str = str(EQUIPMENT_E2E_IMAGES_DIR)

        def gt_normalizer(inst):
            num_kp = EQUIPMENT_KEYPOINT_CONFIGS.get(inst['class_name'], (None, 2, None))[1]
            return _equipment_gt_normalizer(inst, num_kp)

    else:  # attachment
        if results_dir is None:
            results_dir = RESULTS_ATTACHMENT_DIR
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        pole_detector = _load_pole_detector()
        detector, kp_models = _load_attachment_models(device)
        images = get_e2e_test_images('attachment')
        cached = _run_all_inference(
            images, pole_detector, detector, kp_models, device,
            conf_thresh=INFERENCE_ATTACHMENT_CONF_THRESHOLD,
            imgsz=ATTACHMENT_DETECTION_CONFIG['imgsz'],
            class_names=ATTACHMENT_CLASS_NAMES,
            gt_parser=parse_attachments_with_keypoints,
            desc="E2E attachment inference",
        )
        eval_config = ATTACHMENT_EVALUATION_CONFIG
        images_dir_str = str(ATTACHMENT_E2E_IMAGES_DIR)
        gt_normalizer = _attachment_gt_normalizer

    all_results = {}
    for name, cfg in eval_config.items():
        res = _compute_class_metrics(cached, cfg['class_name'], gt_normalizer)
        out = {
            'dataset': name,
            'evaluation_date': datetime.now().isoformat(),
            'pipeline': 'end_to_end',
            'evaluation_split': 'test' if E2E_USE_TEST_SPLIT_ONLY else 'all',
            'images_dir': images_dir_str,
            'images_evaluated': len(images),
            'gt_instance_count': res['detection']['gt_count'],
            'detection': res['detection'],
            'keypoint': res['keypoint'],
        }
        path = results_dir / f'{name}.json'
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"✓ {name} saved to {path}")
        all_results[name] = out
    return all_results


def run_attachment_detection_evaluation(results_dir=None) -> Optional[Dict[str, Any]]:
    """
    Run detection-only evaluation for attachment_detection (all 6 classes).
    Runs threshold sweep, updates config, gets confusion matrix at optimal threshold.
    Saves attachment_detection.json. Returns result dict or None on error.
    """
    results_dir = results_dir or RESULTS_ATTACHMENT_DIR
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    wpath, dpath = YOLO_DETECTION_MODELS["attachment_detection"]
    weights = PROJECT_ROOT / wpath
    data_yaml = PROJECT_ROOT / dpath
    if not weights.exists():
        print(f"  ✗ Weights not found: {weights}, skipping attachment_detection evaluation")
        return None

    print("\n--- Attachment detection (threshold sweep on val, chart on test) ---")
    sweep_result = run_threshold_sweep(
        "attachment_detection",
        weights,
        data_yaml,
        split="val",
        verbose=False,
        return_curves=False,
    )
    if "error" in sweep_result:
        print(f"  ERROR: {sweep_result['error']}")
        return None

    optimal_overall = sweep_result["optimal_overall"]["confidence"]
    optimal_per_class = {k: v["confidence"] for k, v in sweep_result["optimal_per_class"].items()}
    update_attachment_config(optimal_overall, optimal_per_class)

    names_ordered = list(sweep_result["optimal_per_class"].keys())
    test_result = run_val_on_split(
        weights,
        data_yaml,
        split="test",
        conf=optimal_overall,
        optimal_per_class=sweep_result["optimal_per_class"],
        names_ordered=names_ordered,
        verbose=False,
    )
    if "error" in test_result:
        print(f"  ERROR (test eval): {test_result['error']}")
        return None

    out = {
        "dataset": "attachment_detection",
        "evaluation_date": datetime.now().isoformat(),
        "pipeline": "detection_only",
        "evaluation_split": "test",
        "data_yaml": str(data_yaml),
        "optimal_threshold": optimal_overall,
        "optimal_per_class": test_result["optimal_per_class"],
        "detection": {
            "map_0_5": test_result.get("map_0_5"),
            "mean_iou": None,
            "f1": test_result["optimal_overall"]["f1"],
        },
        "f1_curve": test_result.get("f1_curve"),
        "p_curve": test_result.get("p_curve"),
        "r_curve": test_result.get("r_curve"),
        "prec_values": test_result.get("prec_values"),
        "px": test_result.get("px"),
        "names": test_result.get("names"),
        "names_ordered": test_result.get("names_ordered"),
        "confusion_matrix": test_result.get("confusion_matrix"),
    }

    path = results_dir / "attachment_detection.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ attachment_detection saved to {path}")
    return out


def run_attachment_evaluation(results_dir=None, device=None) -> Dict[str, Dict]:
    """Run detection-only + end-to-end evaluation for comm and down_guy, save JSONs."""
    # 1. Detection-only evaluation (threshold sweep, attachment_detection.json)
    run_attachment_detection_evaluation(results_dir)
    # 2. E2E evaluation (comm_detection.json, down_guy_detection.json)
    return _run_domain_evaluation('attachment', results_dir, device)


def run_equipment_detection_evaluation(results_dir=None) -> Optional[Dict[str, Any]]:
    """
    Run detection-only evaluation for equipment_detection (riser, transformer, street_light, secondary_drip_loop).
    Runs threshold sweep, updates config, gets confusion matrix at optimal threshold.
    Saves equipment_detection.json. Returns result dict or None on error.
    """
    results_dir = results_dir or RESULTS_EQUIPMENT_DIR
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    wpath, dpath = YOLO_DETECTION_MODELS["equipment_detection"]
    weights = PROJECT_ROOT / wpath
    data_yaml = PROJECT_ROOT / dpath
    if not weights.exists():
        print(f"  ✗ Weights not found: {weights}, skipping equipment_detection evaluation")
        return None

    print("\n--- Equipment detection (threshold sweep on val, chart on test) ---")
    sweep_result = run_threshold_sweep(
        "equipment_detection",
        weights,
        data_yaml,
        split="val",
        verbose=False,
        return_curves=False,
    )
    if "error" in sweep_result:
        print(f"  ERROR: {sweep_result['error']}")
        return None

    optimal_overall = sweep_result["optimal_overall"]["confidence"]
    optimal_per_class = {k: v["confidence"] for k, v in sweep_result["optimal_per_class"].items()}
    update_equipment_config(optimal_overall, optimal_per_class)

    names_ordered = list(sweep_result["optimal_per_class"].keys())
    test_result = run_val_on_split(
        weights,
        data_yaml,
        split="test",
        conf=optimal_overall,
        optimal_per_class=sweep_result["optimal_per_class"],
        names_ordered=names_ordered,
        verbose=False,
    )
    if "error" in test_result:
        print(f"  ERROR (test eval): {test_result['error']}")
        return None

    out = {
        "dataset": "equipment_detection",
        "evaluation_date": datetime.now().isoformat(),
        "pipeline": "detection_only",
        "evaluation_split": "test",
        "data_yaml": str(data_yaml),
        "optimal_threshold": optimal_overall,
        "optimal_per_class": test_result["optimal_per_class"],
        "detection": {
            "map_0_5": test_result.get("map_0_5"),
            "mean_iou": None,
            "f1": test_result["optimal_overall"]["f1"],
        },
        "f1_curve": test_result.get("f1_curve"),
        "p_curve": test_result.get("p_curve"),
        "r_curve": test_result.get("r_curve"),
        "prec_values": test_result.get("prec_values"),
        "px": test_result.get("px"),
        "names": test_result.get("names"),
        "names_ordered": test_result.get("names_ordered"),
        "confusion_matrix": test_result.get("confusion_matrix"),
    }

    path = results_dir / "equipment_detection.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ equipment_detection saved to {path}")
    return out


def run_equipment_evaluation(results_dir=None, device=None) -> Dict[str, Dict]:
    """Run detection-only + end-to-end evaluation for street_light, transformer, riser, etc.; save JSONs."""
    run_equipment_detection_evaluation(results_dir)
    return _run_domain_evaluation('equipment', results_dir, device)
