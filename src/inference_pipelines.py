"""
Four inference pipelines:

- calibration_pipeline: pole → ruler → ruler keypoints → pole top
  Use for pole/midspan photos with ruler (height calibration).

- equipment_pipeline: pole → 70% 2:5 crop → equipment (riser, transformer, street_light) → keypoints
  Use for full pole photos; detects equipment and keypoints in pole region.

- attachment_pipeline: pole → 70% 2:5 crop → attachments (comm, down_guy) → keypoints
  Use for full pole photos; detects attachments and optional keypoints in pole region.

- annotation_pipeline: pole → 70% 2:5 crop → equipment + attachment (combined, optimized)
  Use for production; runs equipment and attachment on shared crop (load + pole once).

Usage:
    from src.inference_pipelines import calibration_pipeline, equipment_pipeline, attachment_pipeline, annotation_pipeline
    calibration_pipeline.run(image_path, models, ...)
    equipment_pipeline.run(images_dir, pole_detector, equip_detector, kp_models, ...)
    attachment_pipeline.run(images_dir, pole_detector, attachment_detector, ...)
    equip_preds, attach_preds, ppi = annotation_pipeline.run_single(img_path, pole_detector, ...)
"""

from pathlib import Path
from typing import Dict, Any, Optional

from .config import (
    INFERENCE_POLE_CONF_THRESHOLD,
    INFERENCE_ATTACHMENT_CONF_THRESHOLD,
    INFERENCE_ATTACHMENT_CONF_PER_CLASS,
    INFERENCE_ATTACHMENT_MIN_BBOX_AREA_FRAC,
    EQUIPMENT_COLORS,
    ATTACHMENT_COLORS,
    DEFAULT_UNKNOWN_COLOR,
    VIZ_FIG_DEFAULTS,
    POLE_LABELS_DIR,
    OBJECT_COLORS,
)
from .data_utils import parse_equipment_from_label_file, parse_attachments_with_keypoints
from .visualization import show_gt_vs_pred

# -----------------------------------------------------------------------------
# Calibration Pipeline
# -----------------------------------------------------------------------------


def run_calibration_pipeline(
    image_path: Path,
    models: Dict[str, Any],
    use_tta: bool = True,
    show_visualization: bool = True,
) -> Dict[str, Any]:
    """
    Run calibration pipeline: pole → ruler → ruler keypoints → pole top.
    For pole/midspan photos (ruler visible).
    """
    from .inference import run_end_to_end_inference_simple
    return run_end_to_end_inference_simple(
        image_path, models,
        use_tta=use_tta,
        show_visualization=show_visualization
    )


def run_calibration_batch(
    images_dir: Path,
    output_dir: Path,
    models: Dict[str, Any],
    use_tta: bool = True,
    save_annotated: bool = True,
    save_labels: bool = True,
) -> list:
    """Run calibration pipeline on a batch of images."""
    from .inference import run_batch_inference
    return run_batch_inference(
        images_dir, output_dir, models,
        use_tta=use_tta,
        save_annotated=save_annotated,
        save_labels=save_labels
    )


# -----------------------------------------------------------------------------
# Equipment Pipeline
# -----------------------------------------------------------------------------


def run_equipment_pipeline(
    images_dir: Path,
    pole_detector,
    equip_detector,
    kp_models_dict: Dict,
    equipment_names: Dict,
    equipment_colors: Dict,
    keypoint_colors: Dict,
    device,
    pole_conf: float = INFERENCE_POLE_CONF_THRESHOLD,
) -> None:
    """
    Run equipment pipeline on full images: pole → 70% 2:5 crop → equipment → keypoints.
    Picks a random image from images_dir, runs inference, visualizes on original image.
    """
    from .visualization import run_end_to_end_inference
    run_end_to_end_inference(
        pole_detector, equip_detector, kp_models_dict,
        images_dir, equipment_names, equipment_colors, keypoint_colors,
        device, pole_conf=pole_conf
    )


# -----------------------------------------------------------------------------
# Attachment Pipeline
# -----------------------------------------------------------------------------


def run_attachment_pipeline(
    images_dir: Path,
    pole_detector,
    attachment_detector,
    attachment_names: Dict,
    attachment_colors: Dict,
    config: Dict,
    device,
    pole_conf: float = INFERENCE_POLE_CONF_THRESHOLD,
    kp_models_dict: Optional[Dict] = None,
    keypoint_colors: Optional[Dict] = None,
) -> None:
    """
    Run attachment pipeline on full images: pole → 70% 2:5 crop → attachment detection → keypoints.
    If kp_models_dict is provided, runs keypoint detection on each comm/down_guy and overlays them.
    """
    import random
    import cv2
    import numpy as np
    import torch

    test_images = sorted(images_dir.glob('*.jpg'))
    if not test_images:
        raise RuntimeError(f"No images found in {images_dir}")

    random_image = random.choice(test_images)
    print(f"Selected image: {random_image.name}")

    img_bgr = cv2.imread(str(random_image))
    if img_bgr is None:
        raise RuntimeError(f"Failed to load {random_image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]
    print(f"Image dimensions: {w_img}x{h_img}\n")

    # 1. Pole detection
    print("Step 1: Pole detection...")
    pole_res = pole_detector(img_bgr, conf=pole_conf, max_det=1, verbose=False, imgsz=960)[0]
    if pole_res.boxes is None or len(pole_res.boxes) == 0:
        print("  ✗ No pole detected.")
        return
    px1, py1, px2, py2 = map(int, pole_res.boxes.xyxy[0].cpu().numpy())
    pole_bbox = (px1, py1, px2, py2)
    print(f"  ✓ Pole bbox: ({px1}, {py1}) -> ({px2}, {py2})")

    # 2. Extract 70% + 2:5 crop
    print("\nStep 2: Extract attachment crop (upper 70%, 2:5 ratio)...")
    crop, crop_bounds = _extract_crop_from_pole_bbox(img_bgr, pole_bbox)
    if crop is None:
        print("  ✗ Crop too small")
        return
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    crop_h, crop_w = crop.shape[:2]
    print(f"  ✓ Crop: {crop_w}x{crop_h} at ({crop_x1}, {crop_y1})")

    # 3. Attachment detection on crop
    print("\nStep 3: Attachment detection on crop...")
    base_conf = min(INFERENCE_ATTACHMENT_CONF_PER_CLASS.values()) if INFERENCE_ATTACHMENT_CONF_PER_CLASS else INFERENCE_ATTACHMENT_CONF_THRESHOLD
    results = attachment_detector(crop, conf=base_conf, max_det=20, verbose=False, imgsz=config['imgsz'])[0]

    crop_area = crop_h * crop_w
    min_bbox_area = crop_area * INFERENCE_ATTACHMENT_MIN_BBOX_AREA_FRAC
    pred_detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        for i in range(len(results.boxes)):
            bbox = results.boxes.xyxy[i].cpu().numpy()
            conf = float(results.boxes.conf[i].cpu().numpy())
            cls_id = int(results.boxes.cls[i].cpu().numpy())
            cls_name = attachment_names.get(cls_id, f'class_{cls_id}')
            # Per-class confidence filter
            cls_thresh = INFERENCE_ATTACHMENT_CONF_PER_CLASS.get(cls_name, INFERENCE_ATTACHMENT_CONF_THRESHOLD)
            if conf < cls_thresh:
                continue
            ex1, ey1, ex2, ey2 = map(int, bbox)
            # Min bbox area filter
            if (ex2 - ex1) * (ey2 - ey1) < min_bbox_area:
                continue
            x1_full = crop_x1 + ex1
            y1_full = crop_y1 + ey1
            x2_full = crop_x1 + ex2
            y2_full = crop_y1 + ey2
            pred_detections.append({
                'cls_name': cls_name,
                'bbox': (x1_full, y1_full, x2_full, y2_full),
                'conf': conf,
                'keypoints': []
            })
            print(f"  ✓ {cls_name} (conf={conf:.3f})")

    # 4. Keypoint detection on each attachment (if kp_models_dict provided)
    if kp_models_dict and pred_detections:
        print("\nStep 4: Keypoint detection on attachment crops...")
        for det in pred_detections:
            attach_type = det['cls_name']
            x1, y1, x2, y2 = det['bbox']
            attach_crop = img_rgb[y1:y2, x1:x2]
            if attach_crop.shape[0] < 10 or attach_crop.shape[1] < 10:
                print(f"  ⚠ {attach_type} crop too small, skipping")
                continue
            if attach_type not in kp_models_dict:
                print(f"  ⚠ No keypoint model for {attach_type}, skipping")
                continue
            kp_cfg = kp_models_dict[attach_type]
            model = kp_cfg['model']
            kp_names = kp_cfg['kp_names']
            tensor = kp_cfg['preprocess'](attach_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                heatmaps = torch.sigmoid(logits)[0].detach().cpu().numpy()
            eq_h, eq_w = attach_crop.shape[:2]
            for idx, hm in enumerate(heatmaps):
                y_int, x_int = np.unravel_index(np.argmax(hm), hm.shape)
                y_sub, x_sub = float(y_int), float(x_int)
                conf_kp = float(hm[y_int, x_int])
                x_px = x1 + x_sub / max(hm.shape[1] - 1, 1) * (eq_w - 1) if eq_w > 1 else x1
                y_px = y1 + y_sub / max(hm.shape[0] - 1, 1) * (eq_h - 1) if eq_h > 1 else y1
                det['keypoints'].append({'name': kp_names[idx], 'x': x_px, 'y': y_px, 'conf': conf_kp})
                print(f"  ✓ {attach_type} - {kp_names[idx]}: ({x_px:.1f}, {y_px:.1f}), conf={conf_kp:.3f}")
    else:
        print("\nStep 4: (Skipping keypoints - no kp_models_dict)" if not kp_models_dict else "")

    # 5. Visualize GT vs Predicted side by side
    step_num = 5 if kp_models_dict else 4
    print(f"\nStep {step_num}: Assemble and visualize GT vs Predicted")

    # Load GT bboxes AND keypoints from Labels/*_location.txt (coords in percent 0-100)
    gt_bboxes = []
    gt_keypoints = []
    lbl_path = POLE_LABELS_DIR / f'{random_image.stem}_location.txt'
    if lbl_path.exists():
        for eq in parse_attachments_with_keypoints(lbl_path):
            cls = eq['class_name']
            x1 = int(eq['left'] / 100.0 * w_img)
            y1 = int(eq['top'] / 100.0 * h_img)
            x2 = int(eq['right'] / 100.0 * w_img)
            y2 = int(eq['bottom'] / 100.0 * h_img)
            gt_bboxes.append({'bbox': (x1, y1, x2, y2), 'label': cls})
            center = eq.get('center')
            if center is not None:
                gt_keypoints.append({
                    'x': center[0] / 100.0 * w_img,
                    'y': center[1] / 100.0 * h_img,
                    'name': 'attachment',
                    'color': attachment_colors.get(cls),
                })

    # Tag pred keypoints with parent class color
    pred_bboxes_viz = [{'bbox': d['bbox'], 'label': d['cls_name'], 'conf': d['conf']} for d in pred_detections]
    pred_keypoints_viz = [
        {'x': kp['x'], 'y': kp['y'], 'name': kp['name'], 'color': attachment_colors.get(det['cls_name'])}
        for det in pred_detections for kp in det.get('keypoints', [])
    ]

    show_gt_vs_pred(
        img_rgb,
        {'bboxes': gt_bboxes, 'keypoints': gt_keypoints},
        {'bboxes': pred_bboxes_viz, 'keypoints': pred_keypoints_viz},
        title=f'Attachment pipeline: {random_image.name}',
        figsize=(24, 16),
        line_length=max(20, w_img // 20),
        bbox_color_per_label_gt=attachment_colors,
        bbox_color_per_label_pred=attachment_colors,
        legend_items=attachment_colors,
        show_labels=True,
        kp_line_width=4,
        show_plot=True,
    )

    print("\n" + "="*60)
    print("ATTACHMENT PIPELINE SUMMARY")
    print("="*60)
    print(f"Image: {random_image.name} ({w_img}x{h_img})")
    print(f"Detections: {len(pred_detections)}")
    for det in pred_detections:
        print(f"  • {det['cls_name']} (conf={det['conf']:.3f})")
        for kp in det.get('keypoints', []):
            print(f"    {kp['name']}: ({kp['x']:.1f}, {kp['y']:.1f})")
    print("="*60)


def _extract_crop_from_pole_bbox(img, pole_bbox):
    """Extract upper 70% with 2:5 aspect ratio from pole bbox (same as training)."""
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


# -----------------------------------------------------------------------------
# Annotation Pipeline (Equipment + Attachment, optimized)
# -----------------------------------------------------------------------------


def run_annotation_pipeline_single(
    image_path: Path,
    pole_detector,
    equip_detector,
    attach_detector,
    equip_kp_models: Dict,
    attach_kp_models: Dict,
    device,
    pole_conf: float = INFERENCE_POLE_CONF_THRESHOLD,
) -> tuple:
    """
    Run annotation pipeline on one image: equipment + attachment on shared crop.
    Returns (equip_preds, attach_preds, ppi).
    """
    from .evaluation_attachment_equipment import run_e2e_annotation_single_image
    return run_e2e_annotation_single_image(
        image_path, pole_detector, equip_detector, attach_detector,
        equip_kp_models, attach_kp_models, device, pole_conf=pole_conf
    )


# Convenience namespace objects for import
calibration_pipeline = type('CalibrationPipeline', (), {'run': staticmethod(run_calibration_pipeline), 'run_batch': staticmethod(run_calibration_batch)})()
equipment_pipeline = type('EquipmentPipeline', (), {'run': staticmethod(run_equipment_pipeline)})()
attachment_pipeline = type('AttachmentPipeline', (), {'run': staticmethod(run_attachment_pipeline)})()
annotation_pipeline = type('AnnotationPipeline', (), {'run_single': staticmethod(run_annotation_pipeline_single)})()
