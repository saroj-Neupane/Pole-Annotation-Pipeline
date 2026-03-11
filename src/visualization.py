"""
Visualization utilities for dataset inspection and debugging.
"""

import random
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    DATASETS_DIR,
    BASE_DIR_POLE,
    path_relative_to_project,
    POLE_LABELS_DIR,
    POLE_DETECTION_CONFIG,
    RULER_DETECTION_CONFIG,
    EQUIPMENT_DETECTION_CONFIG,
    INFERENCE_POLE_CONF_THRESHOLD,
    INFERENCE_POLE_CONF_THRESHOLD,
    INFERENCE_RULER_CONF_THRESHOLD,
    INFERENCE_EQUIPMENT_CONF_THRESHOLD,
    INFERENCE_EQUIPMENT_CONF_PER_CLASS,
    INFERENCE_EQUIPMENT_MIN_BBOX_AREA_FRAC,
    INFERENCE_SECONDARY_DRIP_LOOP_MAX_DET,
    INFERENCE_ATTACHMENT_CONF_THRESHOLD,
    INFERENCE_ATTACHMENT_CONF_PER_CLASS,
    INFERENCE_ATTACHMENT_MIN_BBOX_AREA_FRAC,
    INFERENCE_MAX_DETECTIONS,
    EQUIPMENT_CLASSES,
    ATTACHMENT_CLASS_NAMES,
    INFERENCE_POLE_WEIGHTS,
    INFERENCE_RULER_WEIGHTS,
    COLOR_GT,
    COLOR_PRED,
    COLOR_OVERLAP,
    COLOR_POLE,
    COLOR_RULER,
    EQUIPMENT_COLORS,
    ATTACHMENT_COLORS,
    KEYPOINT_COLORS,
    OBJECT_COLORS,
    DEFAULT_UNKNOWN_COLOR,
    FALLBACK_KEYPOINT_COLORS,
    KEYPOINT_VIZ_LINE_COLOR,
    RULER_MARKING_COLOR_MAP,
    VIZ_FIG_DEFAULTS,
    VIZ_BBOX_THICKNESS,
    VIZ_DETECTION_BBOX_THICKNESS,
    VIZ_LINE_LENGTH_FRAC,
    VIZ_FONT_SCALE_DENOM,
    VIZ_FONT_THICK_DENOM,
)
from .data_utils import parse_label_file, parse_equipment_from_label_file, parse_equipment_with_keypoints
from .geometry_utils import line_perfect_overlap

# =============================================================================
# Centralized drawing primitives (object detection + keypoint detection)
# Use these consistently across inference, notebooks, and evaluation
# =============================================================================

# Re-export for convenience; config is source of truth
# COLOR_GT, COLOR_PRED, COLOR_OVERLAP, EQUIPMENT_COLORS, ATTACHMENT_COLORS, KEYPOINT_COLORS imported above


def bgr_to_rgb_color(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Convert a BGR color tuple to RGB. Use when a color was defined for OpenCV BGR order."""
    return (color[2], color[1], color[0])


def rgb_to_bgr_color(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Convert an RGB color tuple to BGR for OpenCV drawing on BGR images."""
    return (color[2], color[1], color[0])


def rgb_tuple_to_hex(color: Tuple[int, int, int]) -> str:
    """Convert RGB tuple (0-255) to matplotlib hex string (e.g. '#ff0000')."""
    return '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])


def compute_font_scale_thickness(img_width: int) -> Tuple[float, int]:
    """Compute font_scale and font_thickness from image width (consistent across viz)."""
    font_scale = max(0.5, img_width / VIZ_FONT_SCALE_DENOM)
    font_thick = max(1, int(img_width / VIZ_FONT_THICK_DENOM))
    return font_scale, font_thick


def put_text_with_border(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    line_type: int = cv2.LINE_AA,
) -> None:
    """Draw text with black outline for clarity on any background."""
    cv2.putText(img, text, org, font_face, font_scale, (0, 0, 0), thickness + 2, line_type)
    cv2.putText(img, text, org, font_face, font_scale, color, thickness, line_type)


def draw_bboxes(
    rgb: np.ndarray,
    boxes: List[Dict],
    color: Optional[Tuple[int, int, int]] = None,
    color_per_label: Optional[Dict[str, Tuple[int, int, int]]] = None,
    show_label: bool = False,
    show_conf: bool = False,
    line_width: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes on an RGB image.

    Args:
        rgb: RGB image array (H, W, 3)
        boxes: List of dicts with 'bbox': (x1,y1,x2,y2) and 'label' or 'class_name'
        color: Single color (R,G,B) for all boxes. Ignored if color_per_label set.
        color_per_label: Dict mapping label str -> (R,G,B). Uses COLOR_PRED if neither set.
        show_label: Draw text label above box
        show_conf: Include confidence in label if present
        line_width: Box border width

    Returns:
        RGB image with boxes overlaid
    """
    vis = rgb.copy()
    for box in boxes:
        bbox = box.get('bbox')
        if bbox is None or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        label = box.get('label') or box.get('class_name', '')
        conf = box.get('conf')
        if color_per_label and label:
            c = color_per_label.get(label, color or COLOR_PRED)
        else:
            c = color or COLOR_PRED
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, line_width)
        if show_label:
            text = f"{label}"
            if show_conf and conf is not None:
                text += f" ({conf:.2f})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw, y1), c, -1)
            put_text_with_border(vis, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def draw_keypoints(
    rgb: np.ndarray,
    points: List[Dict],
    color: Tuple[int, int, int],
    line_length: Optional[int] = None,
    style: str = 'line',
    show_labels: bool = False,
    line_width: int = 2,
) -> np.ndarray:
    """
    Draw keypoints on an RGB image.

    Args:
        rgb: RGB image array
        points: List of dicts with 'x', 'y', optional 'name'
        color: RGB color tuple for lines/dots
        line_length: Length of horizontal line to the right. Default: min(20, w//20)
        style: 'line' (horizontal line), 'dot' (circle), or 'both'
        show_labels: If True, draw keypoint name to the right of the line

    Returns:
        RGB image with keypoints overlaid
    """
    vis = rgb.copy()
    h, w = vis.shape[:2]
    if line_length is None:
        line_length = min(20, max(5, w // 20))
    for kp in points:
        x = kp.get('x')
        y = kp.get('y')
        if x is None or y is None:
            continue
        x_int, y_int = int(round(x)), int(round(y))
        c = kp.get('color') or color
        if style in ('line', 'both'):
            x_end = min(x_int + line_length, w - 1)
            cv2.line(vis, (x_int, y_int), (x_end, y_int), c, line_width)
        if style in ('dot', 'both'):
            cv2.circle(vis, (x_int, y_int), 4, c, -1)
        if show_labels and kp.get('name'):
            label = str(kp['name']).replace('_', ' ')
            fscale, fthick = compute_font_scale_thickness(w)
            put_text_with_border(vis, label, (x_int + line_length + 5, y_int + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, fscale, c, fthick)
    return vis


def show_gt_vs_pred(
    image_rgb: np.ndarray,
    gt_data: Dict,
    pred_data: Dict,
    title: str = 'GT vs Predicted',
    figsize: Tuple[int, int] = (16, 10),
    line_length: Optional[int] = None,
    bbox_color_gt: Tuple[int, int, int] = COLOR_GT,
    bbox_color_pred: Tuple[int, int, int] = COLOR_PRED,
    kp_color_gt: Tuple[int, int, int] = COLOR_GT,
    kp_color_pred: Tuple[int, int, int] = COLOR_PRED,
    bbox_color_per_label_gt: Optional[Dict[str, Tuple]] = None,
    bbox_color_per_label_pred: Optional[Dict[str, Tuple]] = None,
    show_plot: bool = True,
    show_labels: bool = False,
    legend_items: Optional[Dict[str, Tuple]] = None,
    kp_line_width: int = 2,
) -> None:
    """
    Side-by-side visualization: left = Ground Truth only, right = Predicted only.

    Args:
        image_rgb: Base RGB image
        gt_data: Dict with optional 'bboxes' (list of {bbox, label}) and/or 'keypoints' (list of {x,y,name})
        pred_data: Same structure as gt_data
        title: Figure title
        figsize: (width, height) for matplotlib figure
        line_length: For keypoint lines. Default: image width / 20
        ... color overrides ...
        show_plot: If True, call plt.show()
        legend_items: Dict mapping label -> RGB color for legend. Auto-derived if None.
        kp_line_width: Thickness of keypoint lines in pixels.
    """
    import matplotlib.pyplot as plt

    h, w = image_rgb.shape[:2]
    if line_length is None:
        line_length = max(10, w // 20)

    gt_bboxes = gt_data.get('bboxes', [])
    gt_keypoints = gt_data.get('keypoints', [])
    pred_bboxes = pred_data.get('bboxes', [])
    pred_keypoints = pred_data.get('keypoints', [])

    img_gt = image_rgb.copy()
    if gt_bboxes:
        img_gt = draw_bboxes(
            img_gt, gt_bboxes,
            color=bbox_color_gt,
            color_per_label=bbox_color_per_label_gt,
            show_label=show_labels,
            show_conf=False, line_width=VIZ_BBOX_THICKNESS,
        )
    if gt_keypoints:
        img_gt = draw_keypoints(img_gt, gt_keypoints, kp_color_gt, line_length=line_length, show_labels=show_labels, line_width=kp_line_width)

    img_pred = image_rgb.copy()
    if pred_bboxes:
        img_pred = draw_bboxes(
            img_pred, pred_bboxes,
            color=bbox_color_pred,
            color_per_label=bbox_color_per_label_pred,
            show_label=show_labels,
            show_conf=True, line_width=VIZ_BBOX_THICKNESS,
        )
    if pred_keypoints:
        img_pred = draw_keypoints(img_pred, pred_keypoints, kp_color_pred, line_length=line_length, show_labels=show_labels, line_width=kp_line_width)

    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=VIZ_FIG_DEFAULTS['facecolor'])
        axes[0].imshow(img_gt)
        axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        axes[1].imshow(img_pred)
        axes[1].set_title('Predicted', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

        if legend_items is not None:
            legend_colors = legend_items
        else:
            legend_colors = {}
            if bbox_color_per_label_gt:
                legend_colors.update(bbox_color_per_label_gt)
            if bbox_color_per_label_pred:
                legend_colors.update(bbox_color_per_label_pred)
        handles = make_legend_handles(legend_colors)
        add_legend_to_figure(fig, handles)

        fig.subplots_adjust(wspace=0.02, left=0.02, right=0.98, top=0.92, bottom=0.08)
        plt.show()


def inference_results_to_viz_data(results: Dict) -> Dict:
    """Convert inference results dict to viz_data format for show_gt_vs_pred."""
    bboxes = []
    keypoints = []
    if results.get('pole'):
        px1, py1, px2, py2 = results['pole']
        bboxes.append({'bbox': (px1, py1, px2, py2), 'label': 'pole'})
    if results.get('ruler'):
        rx1, ry1, rx2, ry2 = results['ruler']
        bboxes.append({'bbox': (rx1, ry1, rx2, ry2), 'label': 'ruler'})
    if results.get('keypoints'):
        keypoints.extend(results['keypoints'])
    if results.get('pole_top'):
        pt = results['pole_top']
        keypoints.append({'x': pt['x'], 'y': pt['y'], 'name': 'pole_top'})
    return {'bboxes': bboxes, 'keypoints': keypoints}


def gt_data_to_viz_data(gt_data: Dict) -> Dict:
    """Convert load_all_ground_truth output to viz_data format for show_gt_vs_pred."""
    bboxes = []
    keypoints = []
    if gt_data.get('gt_pole_bbox'):
        x1, y1, x2, y2 = gt_data['gt_pole_bbox']
        bboxes.append({'bbox': (x1, y1, x2, y2), 'label': 'pole'})
    if gt_data.get('gt_ruler_bbox'):
        x1, y1, x2, y2 = gt_data['gt_ruler_bbox']
        bboxes.append({'bbox': (x1, y1, x2, y2), 'label': 'ruler'})
    keypoints.extend(gt_data.get('gt_keypoints', []))
    if gt_data.get('gt_pole_top'):
        pt = gt_data['gt_pole_top']
        keypoints.append({'x': pt['x'], 'y': pt['y'], 'name': 'pole_top'})
    return {'bboxes': bboxes, 'keypoints': keypoints}


def print_inference_summary(
    image_path,
    gt_bbox_count: int = 0,
    gt_keypoint_count: int = 0,
    pred_bboxes: Optional[List[Dict]] = None,
    pred_keypoints: Optional[List[Dict]] = None,
    keypoint_errors_inch: Optional[List[Tuple[str, float]]] = None,
    compare_bboxes: bool = True,
) -> None:
    """
    Print standardized inference summary: image name, path (relative), confidences, counts, inches error.
    When compare_bboxes=False, bbox counts are shown as "not compared" to avoid mixing "not shown" with "not detected".
    """
    pred_bboxes = pred_bboxes or []
    pred_keypoints = pred_keypoints or []
    keypoint_errors_inch = keypoint_errors_inch or []

    path = Path(image_path)
    rel = path_relative_to_project(path)
    print(f"Image: {path.name}")
    print(f"Path: {rel}")
    gt_bbox_str = f"{gt_bbox_count} bboxes" if compare_bboxes else "bboxes not compared"
    print(f"GT: {gt_bbox_str}, {gt_keypoint_count} keypoints")
    print(f"Pred: {len(pred_bboxes)} detections, {len(pred_keypoints)} keypoints")

    for i, b in enumerate(pred_bboxes):
        conf = b.get('conf')
        label = b.get('label', b.get('class_name', 'obj'))
        c = f" ({conf:.3f})" if conf is not None else ""
        print(f"  Pred bbox {i+1}: {label}{c}")
    for i, k in enumerate(pred_keypoints):
        conf = k.get('conf')
        name = k.get('name', 'kp')
        c = f" conf={conf:.3f}" if conf is not None else ""
        err_str = ""
        for ename, einch in keypoint_errors_inch:
            if ename == name:
                err_str = f" error={einch:.3f} in"
                break
        print(f"  Pred kp {i+1}: {name}{c}{err_str}")


def get_keypoint_color(kp_name: str, keypoint_colors: Optional[Dict] = None) -> Tuple[int, int, int]:
    """Get color for a keypoint, with fallback for unknown names. Uses config KEYPOINT_COLORS by default."""
    colors = keypoint_colors if keypoint_colors is not None else KEYPOINT_COLORS
    if kp_name in colors:
        return colors[kp_name]
    kp_name_lower = kp_name.lower().replace(' ', '_')
    if kp_name_lower in colors:
        return colors[kp_name_lower]
    hash_val = hash(kp_name) % len(FALLBACK_KEYPOINT_COLORS)
    return FALLBACK_KEYPOINT_COLORS[hash_val]


def make_legend_handles(color_map: Dict[str, Tuple[int, int, int]]) -> List:
    """Create Patch legend handles from {label: rgb_tuple} dict."""
    from matplotlib.patches import Patch
    return [Patch(facecolor=rgb_tuple_to_hex(c), edgecolor='black', linewidth=1, label=n)
            for n, c in color_map.items()]


def add_legend_to_figure(fig, handles: List, ncol: Optional[int] = None) -> None:
    """Attach legend below figure with consistent style."""
    if not handles:
        return
    ncol = ncol or min(len(handles), 5)
    fig.legend(handles=handles, loc='lower center', ncol=ncol,
               frameon=True, fancybox=True, fontsize=11, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.10)


def setup_inference_environment():
    """Setup device, colors, and names for inference visualization."""
    from .config import ATTACHMENT_CLASSES
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    equipment_names = {v: k for k, v in EQUIPMENT_CLASSES.items()}
    attachment_names = {v: k for k, v in ATTACHMENT_CLASSES.items()}
    return {
        'device': device,
        'equipment_names': equipment_names,
        'attachment_names': attachment_names,
        'equipment_colors': EQUIPMENT_COLORS,
        'attachment_colors': ATTACHMENT_COLORS,
        'keypoint_colors': KEYPOINT_COLORS,
    }


def visualize_pole_and_midspan_images(
    pole_photos_dir: Path, pole_labels_dir: Path,
    midspan_photos_dir: Path, midspan_labels_dir: Path
) -> None:
    """Visualize random pole and midspan images with ruler keypoints and bboxes."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    # Find photos with labels
    pole_files = [p for p in pole_photos_dir.glob("*_1_Main.jpg")
                  if (pole_labels_dir / f"{p.stem}_location.txt").exists()]
    midspan_files = [p for p in midspan_photos_dir.glob("*_Midspan_Height_*.jpg")
                     if (midspan_labels_dir / f"{p.stem}_location.txt").exists()]

    if not pole_files or not midspan_files:
        print(f"Pole files: {len(pole_files)}, Midspan files: {len(midspan_files)}")
        return

    # Select random samples
    pole_path = np.random.choice(pole_files)
    midspan_path = np.random.choice(midspan_files)

    # Load images
    pole_img = cv2.cvtColor(cv2.imread(str(pole_path)), cv2.COLOR_BGR2RGB)
    midspan_img = cv2.cvtColor(cv2.imread(str(midspan_path)), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor=VIZ_FIG_DEFAULTS['facecolor'], dpi=VIZ_FIG_DEFAULTS['dpi'])

    color_map = RULER_MARKING_COLOR_MAP
    pole_color = np.array(COLOR_POLE) / 255.0
    ruler_color = np.array(COLOR_RULER) / 255.0

    def _draw_annotations_on_ax(ax, img, label_file, title):
        ax.imshow(img)
        ax.set_title(title, fontsize=VIZ_FIG_DEFAULTS['title_fontsize'], fontweight=VIZ_FIG_DEFAULTS['title_fontweight'], pad=10)
        ax.set_facecolor(VIZ_FIG_DEFAULTS['facecolor'])
        ax.axis('off')
        h, w = img.shape[:2]

        pole_bbox, ruler_bbox, keypoints, _ = parse_label_file(label_file)

        # Draw bounding boxes (no text labels — legend only)
        for bbox, color in [(pole_bbox, pole_color), (ruler_bbox, ruler_color)]:
            if bbox:
                left_px, right_px = int(bbox[0]/100*w), int(bbox[1]/100*w)
                top_px, bottom_px = int(bbox[2]/100*h), int(bbox[3]/100*h)
                rect = plt.Rectangle((left_px, top_px), right_px-left_px, bottom_px-top_px,
                                   linewidth=VIZ_BBOX_THICKNESS, edgecolor=color, facecolor='none', zorder=10)
                ax.add_patch(rect)

        # Draw keypoints (horizontal lines, no text labels — legend only)
        for height, (x, y) in keypoints.items():
            x_px, y_px = int(x/100*w), int(y/100*h)
            color = color_map.get(str(height), rgb_tuple_to_hex(COLOR_OVERLAP))
            x_end = min(x_px + 400, w-1)
            ax.plot([x_px, x_end], [y_px, y_px], color=color, linewidth=1, alpha=0.8, zorder=9)

    _draw_annotations_on_ax(axes[0], pole_img, pole_labels_dir / f"{pole_path.stem}_location.txt",
                            f"Pole: {pole_path.stem}")
    _draw_annotations_on_ax(axes[1], midspan_img, midspan_labels_dir / f"{midspan_path.stem}_location.txt",
                            f"Midspan: {midspan_path.stem}")

    # Build legend: pole, ruler, and each ruler marking height
    legend_items = {'pole': COLOR_POLE, 'ruler': COLOR_RULER}
    for height_str, hex_color in color_map.items():
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        legend_items[f'height {height_str}'] = (r, g, b)
    add_legend_to_figure(fig, make_legend_handles(legend_items))

    plt.show()


def parse_yolo_label(label_path: Path, img_width: int, img_height: int) -> List[Dict]:
    """
    Parse YOLO format label file (class x_center y_center width height).
    
    Args:
        label_path: Path to YOLO label file
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        List of dictionaries with 'class_id' and 'bbox' [x1, y1, x2, y2] keys
    """
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                boxes.append({
                    'class_id': class_id,
                    'bbox': [x1, y1, x2, y2]
                })
    return boxes


def parse_keypoint_label(label_path: Path, img_width: int, img_height: int) -> List[Dict]:
    """
    Parse keypoint label file (format: class x_center y_center width height x1 y1 v1 x2 y2 v2 ...).
    
    Args:
        label_path: Path to keypoint label file
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        List of dictionaries with 'x', 'y', and 'visibility' keys
    """
    keypoints = []
    if not label_path.exists():
        return keypoints
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        # Skip PPI comment line if present
        data_line = lines[0] if not lines[0].startswith('#') else lines[1] if len(lines) > 1 else ''
        
        if data_line:
            parts = data_line.strip().split()
            if len(parts) >= 5:
                # Skip bbox info (first 5 values: class x_center y_center width height)
                # Remaining are keypoints in format: x y visibility
                for i in range(5, len(parts), 3):
                    if i + 2 < len(parts):
                        x = float(parts[i]) * img_width
                        y = float(parts[i + 1]) * img_height
                        visibility = int(parts[i + 2])
                        if visibility > 0:  # Only add visible keypoints
                            keypoints.append({
                                'x': int(x),
                                'y': int(y),
                                'visibility': visibility
                            })
    return keypoints


def parse_keypoint_label_midspan(label_path: Path, img_w: int, img_h: int) -> Tuple[Dict[str, Tuple[float, float]], float]:
    """
    Parse keypoint label file from ruler_marking_detection_midspan.
    
    This is a specialized parser for midspan datasets that returns keypoints as a dict
    and includes PPI parsing.
    
    Args:
        label_path: Path to keypoint label file
        img_w: Image width in pixels
        img_h: Image height in pixels
        
    Returns:
        Tuple of (keypoints_dict, ppi)
        - keypoints_dict: Dict mapping keypoint names to (x, y) tuples in pixel coordinates
        - ppi: Pixels per inch value
    """
    keypoints = {}
    ppi = 0.0
    
    if not label_path.exists():
        return keypoints, ppi
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Parse PPI from comment line
    for line in lines:
        line = line.strip()
        if line.startswith('# PPI='):
            try:
                ppi_str = line.split('=')[1].strip()
                ppi = float(ppi_str)
            except Exception:
                pass
    
    # Parse keypoints from data line
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if len(parts) >= 5:
            # Format: class x_center y_center width height kp1_x kp1_y kp1_v ...
            # For midspan, keypoints are stored with names in a specific order
            # This is a simplified version - adjust based on your actual format
            keypoint_names = ['0.0', '2.5', '6.5', '10.5', '14.5', '16.5', '17.0']
            kp_idx = 0
            
            for i in range(5, len(parts), 3):
                if i + 2 < len(parts) and kp_idx < len(keypoint_names):
                    x = float(parts[i]) * img_w
                    y = float(parts[i + 1]) * img_h
                    visibility = int(parts[i + 2])
                    if visibility > 0:
                        keypoints[keypoint_names[kp_idx]] = (x, y)
                    kp_idx += 1
    
    return keypoints, ppi



def create_keypoint_overlay_with_overlap(
    base_img: np.ndarray,
    pred_points: List[Dict],
    gt_points: List[Dict],
    line_length: int,
    tolerance: int = 2,
) -> np.ndarray:
    """
    Create overlay with GT (green), Pred (red), Perfect Overlap (yellow).
    Handles single or multiple keypoints. Uses config colors.
    """
    overlay = base_img.copy()
    pred_list = pred_points if isinstance(pred_points, list) else [pred_points] if pred_points else []
    gt_list = gt_points if isinstance(gt_points, list) else [gt_points] if gt_points else []

    if not pred_list and not gt_list:
        return overlay

    if pred_list and gt_list:
        pred_lines = []
        gt_lines = []
        drawn_pred = set()
        drawn_gt = set()

        w_img = overlay.shape[1]
        x_end_max = w_img - 1
        for i, kp in enumerate(pred_list):
            x_int = int(round(kp['x']))
            y_int = int(round(kp['y']))
            x_end = min(x_int + line_length, x_end_max)
            pred_lines.append((i, x_int, y_int, x_end, kp.get('name', '')))

        for i, kp in enumerate(gt_list):
            x_int = int(round(kp['x']))
            y_int = int(round(kp['y']))
            x_end = min(x_int + line_length, x_end_max)
            gt_lines.append((i, x_int, y_int, x_end, kp.get('name', '')))

        for pred_idx, x1_pred, y_pred, x2_pred, name_pred in pred_lines:
            for gt_idx, x1_gt, y_gt, x2_gt, name_gt in gt_lines:
                if name_pred == name_gt and pred_idx not in drawn_pred and gt_idx not in drawn_gt:
                    overlap = line_perfect_overlap(
                        (x1_pred, y_pred, x2_pred), (x1_gt, y_gt, x2_gt), tolerance=tolerance
                    )
                    if overlap:
                        x_start, y_overlap, x_end = overlap
                        cv2.line(overlay, (x_start, y_overlap), (x_end, y_overlap), COLOR_OVERLAP, 2)
                        drawn_pred.add(pred_idx)
                        drawn_gt.add(gt_idx)
                        break

        for pred_idx, x1, y, x2, _ in pred_lines:
            if pred_idx not in drawn_pred:
                cv2.line(overlay, (x1, y), (x2, y), COLOR_PRED, 2)
        for gt_idx, x1, y, x2, _ in gt_lines:
            if gt_idx not in drawn_gt:
                cv2.line(overlay, (x1, y), (x2, y), COLOR_GT, 2)
    elif pred_list:
        overlay = draw_keypoints(overlay, pred_list, COLOR_PRED, line_length=line_length, style='line')
    elif gt_list:
        overlay = draw_keypoints(overlay, gt_list, COLOR_GT, line_length=line_length, style='line')

    return overlay


def show_keypoint_inference_3panel(
    panel1_img: np.ndarray,
    panel2_img: np.ndarray,
    panel3_img: np.ndarray,
    title1: str,
    title2: str,
    title3: str,
    figsize: Tuple[int, int] = (18, 12),
    dpi: int = 123,
    show_legend: bool = True,
    show_plot: bool = True,
) -> None:
    """
    Centralized 3-panel keypoint inference display.
    Panel 1: GT vs Pred overlay | Panel 2: Resized with keypoints | Panel 3: Heatmap.
    Uses consistent styling: bold black titles, legend with black-bordered patches.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor=VIZ_FIG_DEFAULTS['facecolor'], dpi=dpi)
    fig.patch.set_facecolor(VIZ_FIG_DEFAULTS['facecolor'])

    for ax, img, title in zip(axes, [panel1_img, panel2_img, panel3_img], [title1, title2, title3]):
        ax.imshow(img)
        ax.set_title(title, color='black', fontsize=VIZ_FIG_DEFAULTS['title_fontsize'], fontweight=VIZ_FIG_DEFAULTS['title_fontweight'], pad=8)
        ax.axis('off')
        ax.set_facecolor(VIZ_FIG_DEFAULTS['facecolor'])

    if show_legend:
        legend_elements = [
            Patch(facecolor=rgb_tuple_to_hex(COLOR_PRED), edgecolor='black', linewidth=1.5, label='Predicted'),
            Patch(facecolor=rgb_tuple_to_hex(COLOR_GT), edgecolor='black', linewidth=1.5, label='Ground Truth'),
            Patch(facecolor=rgb_tuple_to_hex(COLOR_OVERLAP), edgecolor='black', linewidth=1.5, label='Perfect Overlap'),
        ]
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=3,
            frameon=True,
            fancybox=True,
            fontsize=11,
            bbox_to_anchor=(0.5, 0.02),
        )

    plt.subplots_adjust(wspace=0.01, left=0, right=1, top=0.92, bottom=0.08)
    if show_plot:
        plt.show()


def show_detection(detection_type: str, conf: float = None) -> None:
    """
    Show single-object detection on a random validation image.

    Args:
        detection_type: 'pole' or 'ruler'
        conf: Confidence threshold (default: from config per type)
    """
    from ultralytics import YOLO
    import matplotlib.pyplot as plt

    _detection_configs = {
        'pole': {
            'weights': INFERENCE_POLE_WEIGHTS,
            'conf_default': INFERENCE_POLE_CONF_THRESHOLD,
            'dataset': 'pole_detection',
            'imgsz': POLE_DETECTION_CONFIG['imgsz'],
        },
        'ruler': {
            'weights': INFERENCE_RULER_WEIGHTS,
            'conf_default': INFERENCE_RULER_CONF_THRESHOLD,
            'dataset': 'ruler_detection',
            'imgsz': RULER_DETECTION_CONFIG['imgsz'],
        },
    }

    if detection_type not in _detection_configs:
        raise ValueError(f"Unknown detection_type '{detection_type}'. Choose from: {list(_detection_configs)}")

    cfg = _detection_configs[detection_type]
    if conf is None:
        conf = cfg['conf_default']

    weights_path = Path(cfg['weights'])
    if not weights_path.exists():
        raise FileNotFoundError(f"{detection_type.title()} detection weights not found: {weights_path}")

    print(f"Loading {detection_type} detection model...")
    print(f"  - {detection_type.title()} detector: {weights_path}")
    detector = YOLO(str(weights_path))

    val_dir = DATASETS_DIR / cfg['dataset'] / 'images' / 'val'
    val_images = sorted(val_dir.glob('*.jpg'))
    if not val_images:
        raise RuntimeError(f"No validation images found in {val_dir}")

    random_image = random.choice(val_images)
    print(f"\nSelected validation image: {random_image.name}")

    img_orig = cv2.imread(str(random_image))
    if img_orig is None:
        raise RuntimeError(f"Failed to load image: {random_image}")

    img_h, img_w = img_orig.shape[:2]
    print(f"Image dimensions: {img_w}x{img_h}")

    # Load GT label
    gt_label_path = DATASETS_DIR / cfg['dataset'] / 'labels' / 'val' / f'{random_image.stem}.txt'
    gt_boxes = parse_yolo_label(gt_label_path, img_w, img_h) if gt_label_path.exists() else []

    print(f"\nRunning {detection_type} detection (conf={conf})...")
    result = detector(img_orig, conf=conf, max_det=INFERENCE_MAX_DETECTIONS, verbose=False, imgsz=cfg['imgsz'])[0]

    detected = False
    pred_bboxes_list = []
    if result.boxes is not None and len(result.boxes) > 0:
        detected = True
        bbox = result.boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)
        det_conf = float(result.boxes.conf[0].cpu().numpy())
        pred_bboxes_list.append({'bbox': (x1, y1, x2, y2), 'label': detection_type, 'conf': det_conf})
        print(f"  ✓ {detection_type.title()} detected! Confidence: {det_conf:.3f}, BBox: ({x1}, {y1}) -> ({x2}, {y2})")
    else:
        print(f"  ✗ No {detection_type} detected in image")

    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    det_color = OBJECT_COLORS.get(detection_type, COLOR_PRED)
    gt_viz = {'bboxes': [{'bbox': tuple(b['bbox']), 'label': detection_type} for b in gt_boxes], 'keypoints': []}
    pred_viz = {'bboxes': pred_bboxes_list, 'keypoints': []}

    status_text = "Detected" if detected else "Not Detected"
    show_gt_vs_pred(
        img_rgb,
        gt_viz,
        pred_viz,
        title=f'{detection_type.title()} Detection ({status_text}): {random_image.name}',
        figsize=(24, 16),
        bbox_color_per_label_gt={detection_type: det_color},
        bbox_color_per_label_pred={detection_type: det_color},
        legend_items={detection_type: det_color},
        show_plot=True,
    )

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  GT boxes: {len(gt_boxes)}")
    print(f"  {detection_type.title()} detected: {'Yes' if detected else 'No'}")
    print(f"{'='*60}")


# Backward-compatible aliases
def show_pole_detection(conf_pole: float = None) -> None:
    """Show pole detection. Alias for show_detection('pole')."""
    show_detection('pole', conf=conf_pole)


def show_ruler_detection(conf_ruler: float = None, midspan: bool = False) -> None:
    """Show ruler detection. Alias for show_detection('ruler')."""
    show_detection('ruler', conf=conf_ruler)


def visualize_sample(dataset_name: str, config: Dict) -> Tuple[Optional[np.ndarray], Optional[str], Dict]:
    """
    Visualize one sample from a dataset with its labels.

    Args:
        dataset_name: Name of the dataset
        config: Configuration dictionary with keys:
            - 'images_dir': Path to images directory
            - 'labels_dir': Path to labels directory
            - 'type': 'yolo_bbox' or 'keypoints'
            - 'class_names': List of class names (for yolo_bbox)
            - 'keypoint_names': List of keypoint names (for keypoints)

    Returns:
        Tuple of (visualized_image, dataset_name, legend_items) or (None, None, {}) if error
    """
    images_dir = config['images_dir']
    labels_dir = config['labels_dir']

    # Find available images
    image_files = list(images_dir.glob('*.jpg'))
    if not image_files:
        print(f"No images found in {images_dir}")
        return None, None, {}

    # Select random image
    image_path = random.choice(image_files)
    label_path = labels_dir / f"{image_path.stem}.txt"

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        return None, None, {}

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    legend_items: Dict[str, Tuple[int, int, int]] = {}

    if config['type'] == 'yolo_bbox':
        boxes = parse_yolo_label(label_path, w, h)
        class_colors: Dict[str, Tuple[int, int, int]] = {}
        viz_boxes = []
        for box in boxes:
            class_id = box['class_id']
            class_name = config['class_names'][class_id] if class_id < len(config['class_names']) else f'class_{class_id}'
            color_rgb = (
                COLOR_POLE if 'pole' in class_name.lower() else
                COLOR_RULER if 'ruler' in class_name.lower() else
                OBJECT_COLORS.get(class_name, COLOR_GT)
            )
            class_colors[class_name] = color_rgb
            legend_items[class_name] = color_rgb
            viz_boxes.append({'bbox': box['bbox'], 'label': class_name})
        img_rgb = draw_bboxes(img_rgb, viz_boxes, color_per_label=class_colors, show_label=False)

    elif config['type'] == 'keypoints':
        keypoints = parse_keypoint_label(label_path, w, h)
        keypoint_names = config['keypoint_names']
        line_len = max(5, w // 20)
        kp_list = []
        for idx, kp in enumerate(keypoints):
            kp_name = keypoint_names[idx] if idx < len(keypoint_names) else str(idx)
            color_rgb = get_keypoint_color(kp_name)
            legend_items[kp_name] = color_rgb
            kp_list.append({'x': kp['x'], 'y': kp['y'], 'name': kp_name, 'color': color_rgb})
        if kp_list:
            img_rgb = draw_keypoints(img_rgb, kp_list, kp_list[0]['color'], line_length=line_len, style='line')

    return img_rgb, dataset_name, legend_items


def visualize_dataset_samples_grid(
    config_dict: Dict,
    ncols: int = 2,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Visualize one random sample per dataset in a grid using centralized styling.
    Uses visualize_sample for each config; applies VIZ_FIG_DEFAULTS for consistency.

    Args:
        config_dict: Dict mapping display name -> config (e.g. VISUALIZATION_DATASETS_CONFIG)
        ncols: Number of columns in grid
        figsize: (width, height) for figure. Default: (14, 10) for 2x2
    """
    import matplotlib.pyplot as plt

    items = list(config_dict.items())
    n = len(items)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (7 * ncols, 5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                            facecolor=VIZ_FIG_DEFAULTS['facecolor'],
                            dpi=VIZ_FIG_DEFAULTS['dpi'])
    fig.patch.set_facecolor(VIZ_FIG_DEFAULTS['facecolor'])
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.ravel()

    all_legend_items: Dict[str, Tuple[int, int, int]] = {}
    for idx, (name, config) in enumerate(items):
        ax = axes_flat[idx]
        img_vis, _, legend_items = visualize_sample(name, config)
        all_legend_items.update(legend_items)
        if img_vis is not None:
            ax.imshow(img_vis)
        ax.set_title(name, fontsize=VIZ_FIG_DEFAULTS['title_fontsize'],
                     fontweight=VIZ_FIG_DEFAULTS['title_fontweight'])
        ax.set_facecolor(VIZ_FIG_DEFAULTS['facecolor'])
        ax.axis('off')

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_facecolor(VIZ_FIG_DEFAULTS['facecolor'])
        axes_flat[idx].axis('off')

    handles = make_legend_handles(all_legend_items)
    if handles:
        add_legend_to_figure(fig, handles)
    else:
        fig.subplots_adjust(bottom=0.05)
    plt.show()


def visualize_from_location(
    photos_dir: Path,
    labels_dir: Path,
    prefixes: Tuple[str, ...],
    colors: Dict[str, Tuple[int, int, int]],
    draw_style: str = 'line',
    legend_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    exclude_prefixes: Tuple[str, ...] = (),
) -> None:
    """
    Visualize bboxes and keypoints from location files for given category prefixes.

    Args:
        photos_dir: Directory containing photo images
        labels_dir: Directory containing *_location.txt label files
        prefixes: Tuple of category prefixes to match (e.g. ('riser', 'transformer', ...) or ('comm', 'down_guy'))
        colors: Dict mapping category name -> RGB color tuple
        draw_style: 'line' draws horizontal lines for keypoints, 'dot' draws circles
        legend_colors: Optional dict for legend only (e.g. subset to avoid duplicates)
        exclude_prefixes: Names starting with these are excluded (e.g. 'secondary_drip_loop' for attachment viz)
    """
    import re
    import matplotlib.pyplot as plt

    def _excluded(name: str) -> bool:
        return any(name.startswith(ex) for ex in exclude_prefixes)

    prefix_pattern = '|'.join(re.escape(p) for p in prefixes)

    # Find labels that contain relevant data
    matching_labels = [
        p for p in sorted(labels_dir.glob("*_location.txt"))
        if any(
            (n := line.split(",")[0].strip()) and re.match(rf"({prefix_pattern})", n) and not any(n.startswith(ex) for ex in exclude_prefixes)
            for line in p.read_text().splitlines() if not line.startswith("#") and line.strip() and "," in line
        )
    ]

    if not matching_labels:
        print(f"No labels with data for prefixes {prefixes} found")
        return

    label_path = random.choice(matching_labels)
    photo_path = photos_dir / (label_path.stem.replace("_location", "") + ".jpg")
    if not photo_path.exists():
        print(f"Photo not found: {photo_path}")
        return

    # Parse keypoints and bboxes
    keypoints, bboxes = {}, {}
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        name = parts[0]
        if name.endswith("_bbox") and len(parts) >= 5 and re.match(rf"({prefix_pattern})", name) and not _excluded(name):
            bboxes[name] = tuple(float(v) for v in parts[1:5])
        elif re.match(rf"({prefix_pattern})", name) and not name.endswith("_bbox") and len(parts) >= 3 and not _excluded(name):
            keypoints[name] = (float(parts[1]), float(parts[2]))

    img = cv2.imread(str(photo_path))
    if img is None:
        print(f"Failed to load image: {photo_path}")
        return
    h, w = img.shape[:2]
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    line_len = int(w * VIZ_LINE_LENGTH_FRAC)

    for bbox_name, (left, right, top, bottom) in bboxes.items():
        m = re.match(rf"({prefix_pattern})", bbox_name)
        if m:
            color = colors.get(m.group(1), DEFAULT_UNKNOWN_COLOR)
            x1, y1 = int(left / 100 * w), int(top / 100 * h)
            x2, y2 = int(right / 100 * w), int(bottom / 100 * h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, VIZ_BBOX_THICKNESS)

    for name, (px, py) in keypoints.items():
        x, y = int(px / 100 * w), int(py / 100 * h)
        m = re.match(rf"({prefix_pattern})", name)
        color = colors.get(m.group(1) if m else prefixes[0], DEFAULT_UNKNOWN_COLOR)
        if draw_style == 'dot':
            cv2.circle(vis, (x, y), 6, color, -1)
        else:
            cv2.line(vis, (x, y), (x + line_len, y), color, VIZ_BBOX_THICKNESS)

    fig, ax = plt.subplots(figsize=(12, 16), dpi=VIZ_FIG_DEFAULTS['dpi'], facecolor=VIZ_FIG_DEFAULTS['facecolor'])
    ax.imshow(vis)
    ax.set_title(photo_path.name, fontsize=VIZ_FIG_DEFAULTS['title_fontsize'], fontweight=VIZ_FIG_DEFAULTS['title_fontweight'])
    ax.axis("off")
    add_legend_to_figure(fig, make_legend_handles(legend_colors or colors))
    plt.show()
    print(f"Keypoints: {list(keypoints.keys())}")
    print(f"Bboxes: {list(bboxes.keys())}")


# Backward-compatible aliases
def visualize_equipment_from_location(photos_dir: Path, labels_dir: Path) -> None:
    """Visualize equipment from location files. Alias for visualize_from_location."""
    visualize_from_location(
        photos_dir, labels_dir,
        prefixes=("riser", "transformer", "street_light", "secondary_drip_loop"),
        colors=EQUIPMENT_COLORS, draw_style='line',
    )


# Prefix order: open_secondary before secondary for correct regex matching (incl. backward compat with old labels)
ATTACHMENT_VIZ_PREFIXES = ("comm", "down_guy", "primary", "open_secondary", "secondary", "neutral", "power_guy", "guy")
# Legend: only 6 canonical classes (open_secondary→neutral, power_guy→guy merged)
ATTACHMENT_LEGEND_COLORS = {k: v for k, v in ATTACHMENT_COLORS.items() if k in ('comm', 'down_guy', 'primary', 'secondary', 'neutral', 'guy')}


def visualize_attachments_from_location(photos_dir: Path, labels_dir: Path) -> None:
    """Visualize attachments from location files. Excludes equipment (secondary_drip_loop)."""
    visualize_from_location(
        photos_dir, labels_dir,
        prefixes=ATTACHMENT_VIZ_PREFIXES,
        colors=ATTACHMENT_COLORS, draw_style='dot',
        legend_colors=ATTACHMENT_LEGEND_COLORS,
        exclude_prefixes=("secondary_drip_loop",),
    )


def visualize_bbox_detection_dataset(
    dataset_dir: Path,
    class_names: List[str],
    colors: Dict[str, Tuple[int, int, int]],
    single_random: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Visualize a YOLO bbox detection dataset: one sample per class or one random image.

    Args:
        dataset_dir: Path to dataset with images/ and labels/ subdirs
        class_names: List of class names
        colors: Dict mapping class name -> RGB color
        single_random: If True, show one random image. If False, show one per class.
        figsize: Override figure size. Default: auto-computed from number of classes.
    """
    import matplotlib.pyplot as plt

    dataset_dir = Path(dataset_dir)

    all_imgs = list((dataset_dir / "images").glob("*/*.jpg"))
    if not all_imgs:
        print("No images found in dataset")
        return

    if single_random:
        img_path = random.choice(all_imgs)
        split = img_path.parent.name
        lbl_path = dataset_dir / "labels" / split / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load {img_path.name}")
            return
        h, w = img.shape[:2]
        vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for box in parse_yolo_label(lbl_path, w, h):
            x1, y1, x2, y2 = box['bbox']
            name = class_names[box['class_id']] if box['class_id'] < len(class_names) else f"class_{box['class_id']}"
            color = colors.get(name, DEFAULT_UNKNOWN_COLOR)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, VIZ_BBOX_THICKNESS)

        fig, ax = plt.subplots(1, 1, figsize=figsize or (10, 12))
        ax.imshow(vis)
        ax.set_title(f"[{split}] {img_path.name}", fontsize=VIZ_FIG_DEFAULTS['title_fontsize'], fontweight=VIZ_FIG_DEFAULTS['title_fontweight'])
        ax.axis("off")
        add_legend_to_figure(fig, make_legend_handles(colors))
        plt.show()
        return

    # Build index: which images contain which class
    images_by_class = {i: [] for i in range(len(class_names))}
    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        for img_path in img_dir.glob("*.jpg"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            for line in lbl_path.read_text().strip().splitlines():
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in images_by_class:
                        images_by_class[cls_id].append((img_path, split))
                        break  # one entry per image

    samples = []
    for cls_id, name in enumerate(class_names):
        candidates = images_by_class.get(cls_id, [])
        if candidates:
            img_path, split = random.choice(candidates)
        else:
            img_path = random.choice(all_imgs)
            split = img_path.parent.name
        samples.append((img_path, split, name))

    n = len(class_names)
    default_figsize = figsize or (6 * max(1, n), 16)
    fig, axes = plt.subplots(1, n, figsize=default_figsize, dpi=VIZ_FIG_DEFAULTS['dpi'], facecolor=VIZ_FIG_DEFAULTS['facecolor'])
    axes = np.atleast_1d(axes)
    for ax, (img_path, split, cls_name) in zip(axes, samples):
        lbl_path = dataset_dir / "labels" / split / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            ax.text(0.5, 0.5, f"Failed to load\n{img_path.name}", ha="center", va="center")
            ax.axis("off")
            continue
        h, w = img.shape[:2]
        vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for box in parse_yolo_label(lbl_path, w, h):
            x1, y1, x2, y2 = box['bbox']
            name = class_names[box['class_id']] if box['class_id'] < len(class_names) else f"class_{box['class_id']}"
            color = colors.get(name, DEFAULT_UNKNOWN_COLOR)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, VIZ_BBOX_THICKNESS)

        ax.imshow(vis)
        ax.set_title(f"[{split}] {cls_name}\n{img_path.name}", fontsize=VIZ_FIG_DEFAULTS['title_fontsize'], fontweight=VIZ_FIG_DEFAULTS['title_fontweight'])
        ax.axis("off")
    add_legend_to_figure(fig, make_legend_handles(colors))
    plt.show()


# Backward-compatible aliases
def visualize_equipment_detection_dataset(dataset_dir: Path, class_names: List[str]) -> None:
    """Visualize equipment detection dataset. Alias for visualize_bbox_detection_dataset."""
    class_set = set(class_names)
    colors = {k: v for k, v in EQUIPMENT_COLORS.items() if k in class_set}
    visualize_bbox_detection_dataset(dataset_dir, class_names, colors=colors)


def visualize_attachment_detection_dataset(
    dataset_dir: Path, class_names: List[str], single_random: bool = False
) -> None:
    """Visualize attachment detection dataset. Alias for visualize_bbox_detection_dataset."""
    # Only show colors for classes that exist in the dataset (no open_secondary, power_guy, etc.)
    class_set = set(class_names)
    colors = {k: v for k, v in ATTACHMENT_COLORS.items() if k in class_set}
    visualize_bbox_detection_dataset(
        dataset_dir, class_names, colors=colors,
        single_random=single_random, figsize=(14, 16),
    )


def visualize_keypoint_detection_dataset(
    dataset_paths: Dict[str, Dict],
) -> None:
    """Visualize random sample from equipment keypoint detection datasets."""
    import matplotlib.pyplot as plt

    eq_type, config = random.choice(list(dataset_paths.items()))
    split = random.choice(['train', 'val', 'test'])
    img_dir = config['path'] / 'images' / split
    imgs = list(img_dir.glob('*.jpg'))

    if not imgs:
        print(f"No images in {img_dir}")
        return

    img_path = random.choice(imgs)
    label_path = config['path'] / 'labels' / split / f"{img_path.stem}.txt"
    vis = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = vis.shape[:2]

    label_line = next(l for l in label_path.read_text().splitlines() if not l.startswith('#'))
    parts = label_line.split()

    num_kp = len(config['kp_names'])
    kp_data = [
        {
            'name': config['kp_names'][i],
            'x': float(parts[5 + i * 3]),
            'y': float(parts[5 + i * 3 + 1]),
            'visible': int(parts[5 + i * 3 + 2])
        }
        for i in range(num_kp)
    ]

    color = OBJECT_COLORS.get(eq_type, KEYPOINT_VIZ_LINE_COLOR)
    line_len = int(w * VIZ_LINE_LENGTH_FRAC)

    legend_items: Dict[str, Tuple[int, int, int]] = {}
    for kp in kp_data:
        x, y = int(kp['x'] * w), int(kp['y'] * h)
        if kp['visible']:
            cv2.line(vis, (x, y), (x + line_len, y), color, 2)
            legend_items[kp['name']] = color

    fig = plt.figure(figsize=(10, 10), dpi=VIZ_FIG_DEFAULTS['dpi'], facecolor=VIZ_FIG_DEFAULTS['facecolor'])
    plt.imshow(vis)
    plt.title(f"[{split}] {eq_type.upper()} — {img_path.name}", fontsize=VIZ_FIG_DEFAULTS['title_fontsize'], fontweight=VIZ_FIG_DEFAULTS['title_fontweight'])
    plt.axis('off')
    add_legend_to_figure(fig, make_legend_handles(legend_items))
    plt.show()

    print(f"Equipment Type: {eq_type.upper()}\nSplit: {split}\nKeypoints ({num_kp}):")
    for kp in kp_data:
        status = "✓ visible" if kp['visible'] else "✗ not visible"
        print(f"  • {kp['name']}: ({kp['x']:.3f}, {kp['y']:.3f}) {status}")


def _yolo_device_str(device) -> Optional[str]:
    """Convert torch.device to YOLO predict device string (e.g. '0', 'cpu')."""
    if device is None:
        return None
    if hasattr(device, 'type'):
        if device.type == 'cpu':
            return 'cpu'
        if device.type == 'cuda':
            return str(device.index) if device.index is not None else '0'
    return str(device) if device else None


def run_bbox_detection_inference(
    detector,
    val_dir: Path,
    config: Dict,
    class_names: Dict,
    class_colors: Dict,
    conf_per_class: Dict[str, float],
    conf_threshold: float,
    min_bbox_area_frac: float,
    title: str = 'Detection: GT vs Predicted',
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Run bbox detection inference on a random val image and visualize GT vs Predicted.

    Args:
        detector: YOLO detector model
        val_dir: Path to validation images directory
        config: Dict with 'imgsz' key
        class_names: Dict mapping class_id -> class_name
        class_colors: Dict mapping class_name -> RGB color
        conf_per_class: Dict mapping class_name -> confidence threshold
        conf_threshold: Default confidence threshold
        min_bbox_area_frac: Minimum bbox area as fraction of image area
        title: Title for visualization
        device: torch device

    Returns:
        Tuple of (gt_boxes, pred_boxes)
    """
    val_images = sorted(val_dir.glob('*.jpg'))
    if not val_images:
        raise RuntimeError(f"No validation images found in {val_dir}")

    random_image = random.choice(val_images)
    print(f"Image: {random_image.name}")
    print(f"Path: {path_relative_to_project(random_image)}")

    img_bgr = cv2.imread(str(random_image))
    if img_bgr is None:
        raise RuntimeError(f"Failed to load image: {random_image}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]
    print(f"Image dimensions: {w_img}x{h_img}")

    # Load GT using parse_yolo_label
    gt_label_path = val_dir.parent.parent / "labels" / val_dir.name / f'{random_image.stem}.txt'
    gt_boxes = []
    if gt_label_path.exists():
        print(f"Loading GT from: {path_relative_to_project(gt_label_path)}")
        for box in parse_yolo_label(gt_label_path, w_img, h_img):
            cls_name = class_names.get(box['class_id'], f"class_{box['class_id']}")
            gt_boxes.append({'class_id': box['class_id'], 'class_name': cls_name, 'bbox': tuple(box['bbox'])})
    else:
        print(f"Warning: GT label not found at {gt_label_path}")

    # Run inference (with CPU fallback on CUDA OOM)
    base_conf = min(conf_per_class.values()) if conf_per_class else conf_threshold
    device_str = _yolo_device_str(device)
    predict_kw = dict(conf=base_conf, max_det=20, verbose=False, imgsz=config['imgsz'])
    if device_str is not None:
        predict_kw['device'] = device_str
    per_class_str = ", ".join(f"{c}≥{t:.3f}" for c, t in sorted(conf_per_class.items())) if conf_per_class else ""
    print(f"\nRunning detection (imgsz={config['imgsz']}, base_conf={base_conf}, per_class=[{per_class_str}])...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        results = detector(img_bgr, **predict_kw)[0]
    except (RuntimeError,) as e:
        err_msg = str(e).lower()
        if 'out of memory' in err_msg or 'cuda error' in err_msg or 'accelerator' in err_msg:
            print("GPU OOM, falling back to CPU...")
            predict_kw['device'] = 'cpu'
            results = detector(img_bgr, **predict_kw)[0]
        else:
            raise

    img_area = h_img * w_img
    min_bbox_area = img_area * min_bbox_area_frac
    pred_boxes = []
    if results.boxes is not None and len(results.boxes) > 0:
        for i in range(len(results.boxes)):
            bbox = results.boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)
            conf = float(results.boxes.conf[i].cpu().numpy())
            cls_id = int(results.boxes.cls[i].cpu().numpy())
            cls_name = class_names.get(cls_id, f'class_{cls_id}')
            cls_thresh = conf_per_class.get(cls_name, conf_threshold)
            if conf < cls_thresh:
                continue
            if (x2 - x1) * (y2 - y1) < min_bbox_area:
                continue
            pred_boxes.append({'class_id': cls_id, 'class_name': cls_name, 'bbox': (x1, y1, x2, y2), 'conf': conf})
            print(f"  Detected: {cls_name} (conf={conf:.3f}), BBox: ({x1},{y1})->({x2},{y2})")

    # Visualize using centralized show_gt_vs_pred
    gt_viz = {'bboxes': [{'bbox': b['bbox'], 'label': b['class_name']} for b in gt_boxes], 'keypoints': []}
    pred_viz = {'bboxes': [{'bbox': b['bbox'], 'label': b['class_name'], 'conf': b.get('conf')} for b in pred_boxes], 'keypoints': []}
    print_inference_summary(
        random_image,
        gt_bbox_count=len(gt_boxes),
        gt_keypoint_count=0,
        pred_bboxes=[{'label': b['class_name'], 'conf': b.get('conf')} for b in pred_boxes],
        pred_keypoints=[],
    )

    show_gt_vs_pred(
        img_rgb, gt_viz, pred_viz,
        title=title,
        figsize=(24, 16),
        bbox_color_per_label_gt=class_colors,
        bbox_color_per_label_pred=class_colors,
        show_plot=True,
        show_labels=True,
    )

    return gt_boxes, pred_boxes


# Backward-compatible aliases
def run_equipment_detection_inference(
    detector, val_dir, config, img_width=None, img_height=None,
    equipment_names=None, equipment_colors=None, device=None,
) -> tuple:
    """Run equipment detection inference. Alias for run_bbox_detection_inference."""
    return run_bbox_detection_inference(
        detector, val_dir, config,
        class_names=equipment_names or {},
        class_colors=equipment_colors or EQUIPMENT_COLORS,
        conf_per_class=INFERENCE_EQUIPMENT_CONF_PER_CLASS,
        conf_threshold=INFERENCE_EQUIPMENT_CONF_THRESHOLD,
        min_bbox_area_frac=INFERENCE_EQUIPMENT_MIN_BBOX_AREA_FRAC,
        title='Equipment Detection: GT vs Predicted',
        device=device,
    )


def run_attachment_detection_inference(
    detector, val_dir, config,
    attachment_names=None, attachment_colors=None, device=None,
) -> tuple:
    """Run attachment detection inference. Alias for run_bbox_detection_inference."""
    names = attachment_names or {}
    raw_colors = attachment_colors or ATTACHMENT_COLORS
    class_set = set(names.values()) if names else set(ATTACHMENT_CLASS_NAMES)
    colors = {k: v for k, v in raw_colors.items() if k in class_set}
    return run_bbox_detection_inference(
        detector, val_dir, config,
        class_names=names or dict(enumerate(ATTACHMENT_CLASS_NAMES)),
        class_colors=colors,
        conf_per_class=INFERENCE_ATTACHMENT_CONF_PER_CLASS,
        conf_threshold=INFERENCE_ATTACHMENT_CONF_THRESHOLD,
        min_bbox_area_frac=INFERENCE_ATTACHMENT_MIN_BBOX_AREA_FRAC,
        title='Attachment Detection: GT vs Predicted',
        device=device,
    )


def run_keypoint_detection_inference(
    equipment_type: str,
    model,
    config: Dict,
    preprocess_fn,
    val_dir: Path,
    num_kp: int,
    kp_names: List[str],
    keypoint_colors: Dict,
    device,
):
    """Run keypoint detection inference for equipment type."""
    import matplotlib.pyplot as plt

    val_images = sorted(val_dir.glob('*.jpg'))
    if not val_images:
        raise RuntimeError(f"No validation images found in {val_dir}")

    random_image = random.choice(val_images)
    print(f"Image: {random_image.name}")
    print(f"Path: {path_relative_to_project(random_image)}")

    img_bgr = cv2.imread(str(random_image))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"Image dimensions: {w}x{h}")

    # Load GT - try multiple possible paths
    gt_keypoints = []
    gt_ppi = 0.0

    # Try different label path patterns
    possible_paths = [
        val_dir.parent / "labels" / val_dir.name / f'{random_image.stem}.txt',
        Path(str(val_dir).replace('images', 'labels')) / f'{random_image.stem}.txt',
    ]

    gt_label_path = None
    for path in possible_paths:
        if path.exists():
            gt_label_path = path
            break

    if gt_label_path:
        try:
            with open(gt_label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line.startswith('# PPI='):
                        try:
                            gt_ppi = float(line.split('=')[1])
                        except (ValueError, IndexError):
                            pass
                    elif line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 5 + num_kp * 3:
                            for i in range(num_kp):
                                idx = 5 + i * 3
                                if idx + 2 < len(parts):
                                    try:
                                        kp_x = float(parts[idx]) * w
                                        kp_y = float(parts[idx + 1]) * h
                                        vis = float(parts[idx + 2])
                                        if vis > 0:  # Only add visible keypoints
                                            gt_keypoints.append({'name': kp_names[i], 'x': kp_x, 'y': kp_y})
                                    except (ValueError, IndexError):
                                        pass
        except Exception as e:
            print(f"Warning: Failed to load GT from {gt_label_path}: {e}")

    # Run inference
    print(f"\nRunning {equipment_type} keypoint inference...")
    tensor = preprocess_fn(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        heatmaps = torch.sigmoid(logits)[0].detach().cpu().numpy()

    pred_keypoints = []
    for idx, hm in enumerate(heatmaps):
        y_int, x_int = np.unravel_index(np.argmax(hm), hm.shape)
        y_sub, x_sub = float(y_int), float(x_int)
        conf = float(hm[y_int, x_int])
        x_px = x_sub / max(hm.shape[1] - 1, 1) * (w - 1) if w > 1 else x_sub
        y_px = y_sub / max(hm.shape[0] - 1, 1) * (h - 1) if h > 1 else y_sub
        pred_keypoints.append({'name': kp_names[idx], 'x': x_px, 'y': y_px, 'conf': conf})

    line_len = max(10, w // 20)
    panel1 = create_keypoint_overlay_with_overlap(img_rgb, pred_keypoints, gt_keypoints, line_len)

    img_resized = cv2.resize(img_rgb, (config['resize_width'], config['resize_height']))
    scale_x = config['resize_width'] / w
    scale_y = config['resize_height'] / h
    resized_pred = [{'name': p['name'], 'x': p['x'] * scale_x, 'y': p['y'] * scale_y} for p in pred_keypoints]
    resized_gt = [{'name': g['name'], 'x': g['x'] * scale_x, 'y': g['y'] * scale_y} for g in gt_keypoints]
    panel2 = create_keypoint_overlay_with_overlap(
        img_resized, resized_pred, resized_gt, config['resize_width']
    )

    heatmap_comb = np.max(heatmaps, axis=0)
    heatmap_resized = cv2.resize(heatmap_comb, (img_resized.shape[1], img_resized.shape[0]), interpolation=cv2.INTER_LINEAR)
    hm_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    hm_colored = (plt.cm.hot(hm_norm)[:, :, :3] * 255).astype(np.uint8)
    panel3 = (0.5 * img_resized.astype(np.float32) / 255.0 + 0.5 * hm_colored.astype(np.float32) / 255.0) * 255
    panel3 = panel3.astype(np.uint8)

    show_keypoint_inference_3panel(
        panel1, panel2, panel3,
        title1='GT vs Predicted Keypoints',
        title2=f'Resized ({config["resize_width"]}x{config["resize_height"]}) with Predicted Keypoints',
        title3='Heatmap Overlay on Image',
        show_legend=bool(gt_keypoints),
    )

    # Summary with inches error
    keypoint_errors_inch = []
    if gt_ppi and gt_ppi > 0:
        for pred in pred_keypoints:
            gt_match = next((g for g in gt_keypoints if g['name'] == pred['name']), None)
            if gt_match:
                pixel_err = abs(pred['y'] - gt_match['y'])
                keypoint_errors_inch.append((pred['name'], pixel_err / gt_ppi))

    print_inference_summary(
        random_image,
        gt_bbox_count=0,
        gt_keypoint_count=len(gt_keypoints),
        pred_bboxes=[],
        pred_keypoints=pred_keypoints,
        keypoint_errors_inch=keypoint_errors_inch,
    )


def _extract_equipment_crop_from_pole_bbox(
    img: np.ndarray,
    pole_bbox: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract equipment region from pole bbox: upper 70% with 2:5 aspect ratio (same as training).
    Returns (crop_bgr, (x1, y1, x2, y2)) where the tuple is crop bounds in original image.
    """
    x1, y1, x2, y2 = pole_bbox
    crop_w = x2 - x1
    crop_h_full = y2 - y1
    crop_h = int(crop_h_full * 0.7)
    if crop_h < 10 or crop_w < 10:
        return None, None
    target_width = int(crop_h * (2 / 5))
    center_x = (x1 + x2) / 2
    x1_new = max(0, int(center_x - target_width / 2))
    x2_new = min(img.shape[1], int(center_x + target_width / 2))
    if x2_new - x1_new < 10:
        return None, None
    crop = img[y1 : y1 + crop_h, x1_new:x2_new]
    return crop, (x1_new, y1, x2_new, y1 + crop_h)


def run_end_to_end_inference(
    pole_detector,
    equip_detector,
    kp_models_dict: Dict,
    images_dir: Path,
    equipment_names: Dict,
    equipment_colors: Dict,
    keypoint_colors: Dict,
    device,
    pole_conf: float = INFERENCE_POLE_CONF_THRESHOLD,
):
    """
    Run end-to-end inference on full images:
    1. Predict pole bbox with pole detector
    2. Extract upper 70% with 2:5 ratio crop (as in training)
    3. Predict equipment in that crop
    4. Run keypoint models on each equipment crop
    5. Assemble all predictions on original image
    """
    import matplotlib.pyplot as plt

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
    pole_res = pole_detector(img_bgr, conf=pole_conf, max_det=1, verbose=False,
                            imgsz=960)[0]
    if pole_res.boxes is None or len(pole_res.boxes) == 0:
        print("  ✗ No pole detected. Cannot run equipment inference.")
        return
    px1, py1, px2, py2 = map(int, pole_res.boxes.xyxy[0].cpu().numpy())
    pole_bbox = (px1, py1, px2, py2)
    print(f"  ✓ Pole bbox: ({px1}, {py1}) -> ({px2}, {py2})")

    # 2. Extract 70% + 2:5 equipment crop
    print("\nStep 2: Extract equipment crop (upper 70%, 2:5 ratio)...")
    equip_crop_bgr, crop_bounds = _extract_equipment_crop_from_pole_bbox(img_bgr, pole_bbox)
    if equip_crop_bgr is None:
        print("  ✗ Crop too small")
        return
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    equip_crop_rgb = cv2.cvtColor(equip_crop_bgr, cv2.COLOR_BGR2RGB)
    crop_h, crop_w = equip_crop_rgb.shape[:2]
    print(f"  ✓ Crop: {crop_w}x{crop_h} at ({crop_x1}, {crop_y1})")

    # 3. Equipment detection on crop
    print("\nStep 3: Equipment detection on crop...")
    # Use lowest per-class threshold as base to get all candidates
    base_conf = min(INFERENCE_EQUIPMENT_CONF_PER_CLASS.values()) if INFERENCE_EQUIPMENT_CONF_PER_CLASS else INFERENCE_EQUIPMENT_CONF_THRESHOLD
    results = equip_detector(equip_crop_bgr, conf=base_conf, max_det=20, verbose=False,
                             imgsz=EQUIPMENT_DETECTION_CONFIG['imgsz'])[0]

    crop_area = crop_h * crop_w
    min_bbox_area = crop_area * INFERENCE_EQUIPMENT_MIN_BBOX_AREA_FRAC
    pred_detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        for i in range(len(results.boxes)):
            bbox = results.boxes.xyxy[i].cpu().numpy()
            conf = float(results.boxes.conf[i].cpu().numpy())
            cls_id = int(results.boxes.cls[i].cpu().numpy())
            cls_name = equipment_names.get(cls_id, f'class_{cls_id}')
            # Per-class confidence filter
            cls_thresh = INFERENCE_EQUIPMENT_CONF_PER_CLASS.get(cls_name, INFERENCE_EQUIPMENT_CONF_THRESHOLD)
            if conf < cls_thresh:
                continue
            ex1, ey1, ex2, ey2 = map(int, bbox)
            # Min bbox area filter
            bbox_area = (ex2 - ex1) * (ey2 - ey1)
            if bbox_area < min_bbox_area:
                continue
            # Map from crop coords to full image coords
            x1_full = crop_x1 + ex1
            y1_full = crop_y1 + ey1
            x2_full = crop_x1 + ex2
            y2_full = crop_y1 + ey2
            pred_detections.append({
                'cls_id': cls_id,
                'cls_name': cls_name,
                'bbox': (x1_full, y1_full, x2_full, y2_full),
                'conf': conf,
                'keypoints': []
            })
            print(f"  ✓ {cls_name} (conf={conf:.3f})")

    # Apply max detection limits per class
    # Limit secondary_drip_loop to max 1 (keep highest conf)
    sdl_preds = [d for d in pred_detections if d["cls_name"] == "secondary_drip_loop"]
    if len(sdl_preds) > INFERENCE_SECONDARY_DRIP_LOOP_MAX_DET:
        sdl_preds.sort(key=lambda d: d["conf"], reverse=True)
        keep = {id(d) for d in sdl_preds[:INFERENCE_SECONDARY_DRIP_LOOP_MAX_DET]}
        pred_detections = [d for d in pred_detections if d["cls_name"] != "secondary_drip_loop" or id(d) in keep]

    # 4. Keypoint detection for each equipment
    print("\nStep 4: Keypoint detection on equipment crops...")
    for det in pred_detections:
        eq_type = det['cls_name']
        x1, y1, x2, y2 = det['bbox']
        eq_crop = img_rgb[y1:y2, x1:x2]
        if eq_crop.shape[0] < 10 or eq_crop.shape[1] < 10:
            print(f"  ⚠ {eq_type} crop too small, skipping")
            continue
        if eq_type not in kp_models_dict:
            print(f"  ⚠ No keypoint model for {eq_type}, skipping")
            continue
        kp_cfg = kp_models_dict[eq_type]
        model = kp_cfg['model']
        kp_names = kp_cfg['kp_names']
        tensor = kp_cfg['preprocess'](eq_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            heatmaps = torch.sigmoid(logits)[0].detach().cpu().numpy()
        eq_h, eq_w = eq_crop.shape[:2]
        for idx, hm in enumerate(heatmaps):
            y_int, x_int = np.unravel_index(np.argmax(hm), hm.shape)
            y_sub, x_sub = float(y_int), float(x_int)
            conf = float(hm[y_int, x_int])
            x_px = x1 + x_sub / max(hm.shape[1] - 1, 1) * (eq_w - 1) if eq_w > 1 else x1
            y_px = y1 + y_sub / max(hm.shape[0] - 1, 1) * (eq_h - 1) if eq_h > 1 else y1
            det['keypoints'].append({'name': kp_names[idx], 'x': x_px, 'y': y_px, 'conf': conf})
            print(f"  ✓ {eq_type} - {kp_names[idx]}: ({x_px:.1f}, {y_px:.1f}), conf={conf:.3f}")

    # 5. Visualize GT vs Predicted side by side (on full image)
    print("\nStep 5: Assemble and visualize GT vs Predicted")

    # Load GT bboxes AND keypoints from Labels/*_location.txt (coords in percent 0-100)
    _GT_KP_NAMES = {
        'riser':               ['riser_top'],
        'transformer':         ['top_bolt', 'bottom'],
        'street_light':        ['upper_bracket', 'lower_bracket', 'drip_loop'],
        'secondary_drip_loop': ['lowest_point'],
    }
    lbl_path = POLE_LABELS_DIR / f'{random_image.stem}_location.txt'
    gt_bboxes = []
    gt_keypoints = []
    if lbl_path.exists():
        for eq in parse_equipment_with_keypoints(lbl_path):
            cls = eq['class_name']
            bbox = eq['bbox']
            x1 = int(bbox['left'] / 100.0 * w_img)
            y1 = int(bbox['top'] / 100.0 * h_img)
            x2 = int(bbox['right'] / 100.0 * w_img)
            y2 = int(bbox['bottom'] / 100.0 * h_img)
            gt_bboxes.append({'bbox': (x1, y1, x2, y2), 'label': cls})
            color = equipment_colors.get(cls)
            kp_names = _GT_KP_NAMES.get(cls, [])
            for i, kp_key in enumerate(['kp0', 'kp1', 'kp2']):
                if i >= len(kp_names):
                    break
                kp_pct = eq.get(kp_key)
                if kp_pct is not None:
                    gt_keypoints.append({
                        'x': kp_pct[0] / 100.0 * w_img,
                        'y': kp_pct[1] / 100.0 * h_img,
                        'name': kp_names[i],
                        'color': color,
                    })

    pred_bboxes = [
        {'bbox': det['bbox'], 'label': det['cls_name'], 'conf': det['conf']}
        for det in pred_detections
    ]
    pred_keypoints = [
        {'x': kp['x'], 'y': kp['y'], 'name': kp['name'],
         'color': equipment_colors.get(det['cls_name'])}
        for det in pred_detections for kp in det['keypoints']
    ]

    gt_viz = {'bboxes': gt_bboxes, 'keypoints': gt_keypoints}
    pred_viz = {'bboxes': pred_bboxes, 'keypoints': pred_keypoints}
    show_gt_vs_pred(
        img_rgb,
        gt_viz,
        pred_viz,
        title=f'End-to-end: {random_image.name}',
        figsize=(24, 16),
        line_length=max(20, w_img // 20),
        bbox_color_per_label_gt=equipment_colors,
        bbox_color_per_label_pred=equipment_colors,
        legend_items=equipment_colors,
        show_labels=True,
        kp_line_width=VIZ_DETECTION_BBOX_THICKNESS,
        show_plot=True,
    )

    print("\n" + "="*60)
    print("END-TO-END INFERENCE SUMMARY")
    print("="*60)
    print(f"Image: {random_image.name} ({w_img}x{h_img})")
    print(f"Pole: ({px1}, {py1}) -> ({px2}, {py2})")
    print(f"Equipment crop: ({crop_x1}, {crop_y1}) -> ({crop_x2}, {crop_y2})")
    print(f"Detections: {len(pred_detections)}")
    for det in pred_detections:
        print(f"  • {det['cls_name']} (conf={det['conf']:.3f})")
        for kp in det['keypoints']:
            print(f"    {kp['name']}: ({kp['x']:.1f}, {kp['y']:.1f})")
    print("="*60)
