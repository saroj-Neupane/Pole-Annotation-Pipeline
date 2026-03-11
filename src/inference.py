"""
High-level inference functions for convenient notebook usage.

This module provides simplified, high-level functions that wrap the lower-level
utilities from inference_utils.py, making notebooks more concise and readable.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import matplotlib.pyplot as plt

from .inference_utils import (
    load_trained_keypoint_model,
    load_pole_top_model,
    predict_keypoints,
    predict_pole_top,
    run_end_to_end_inference,
    feet_to_feet_inches
)
from .data_utils import (
    load_ground_truth_keypoints,
    load_ground_truth_pole_top
)
from .visualization import (
    draw_keypoints,
    show_pole_detection,
    show_ruler_detection,
    parse_yolo_label,
    show_gt_vs_pred,
    inference_results_to_viz_data,
    gt_data_to_viz_data,
    create_keypoint_overlay_with_overlap,
    show_keypoint_inference_3panel,
    put_text_with_border,
)
from typing import Tuple, Optional as Opt
from .config import (
    path_relative_to_project,
    DATASETS_DIR,
    KEYPOINT_NAMES,
    RESIZE_HEIGHT,
    RESIZE_WIDTH,
    POLE_TOP_RESIZE_HEIGHT,
    POLE_TOP_RESIZE_WIDTH,
    INFERENCE_POLE_WEIGHTS,
    INFERENCE_RULER_WEIGHTS,
    INFERENCE_RULER_MARKING_WEIGHTS,
    INFERENCE_POLE_TOP_WEIGHTS,
    INFERENCE_POLE_CONF_THRESHOLD,
    INFERENCE_RULER_CONF_THRESHOLD,
    INFERENCE_MAX_DETECTIONS,
    INFERENCE_USE_TTA,
    INFERENCE_USE_INTERPOLATION,
    POLE_LABELS_DIR,
    MIDSPAN_LABELS_DIR
)


def setup_matplotlib_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['legend.fontsize'] = 11


def get_device() -> torch.device:
    """Get the appropriate torch device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_all_models(
    device: Optional[torch.device] = None,
    keypoint_weights: Optional[str] = None,
    pole_top_weights: Optional[str] = None,
    pole_detector_weights: Optional[str] = None,
    ruler_detector_weights: Optional[str] = None,
    skip_missing: bool = True,
) -> Dict[str, Any]:
    """
    Load all models needed for inference.
    
    Args:
        device: torch device (defaults to CUDA if available)
        keypoint_weights: Path to ruler marking keypoint model weights (default: from config)
        pole_top_weights: Path to pole top keypoint model weights (default: from config)
        pole_detector_weights: Path to pole detection YOLO model (default: from config)
        ruler_detector_weights: Path to ruler detection YOLO model (default: from config)
        skip_missing: If True, skip models whose weights don't exist (return None). Default True.
    
    Returns:
        Dictionary with keys: 'device', 'keypoint_model', 'pole_top_model', 
        'pole_detector', 'ruler_detector'. Missing models are None when skip_missing=True.
    """
    from ultralytics import YOLO
    
    if device is None:
        device = get_device()
    
    # Use config defaults if not provided
    if pole_detector_weights is None:
        pole_detector_weights = str(INFERENCE_POLE_WEIGHTS)
    if ruler_detector_weights is None:
        ruler_detector_weights = str(INFERENCE_RULER_WEIGHTS)
    if keypoint_weights is None:
        keypoint_weights = str(INFERENCE_RULER_MARKING_WEIGHTS)
    if pole_top_weights is None:
        pole_top_weights = str(INFERENCE_POLE_TOP_WEIGHTS)
    
    print("Loading models...")
    pole_detector = None
    ruler_detector = None
    keypoint_model = None
    pole_top_model = None
    
    # Load YOLO pole detector
    if Path(pole_detector_weights).exists():
        pole_detector = YOLO(pole_detector_weights)
        if torch.cuda.is_available() and hasattr(pole_detector.model, 'float'):
            pole_detector.model.float()
        print(f"✓ Pole detector loaded from {path_relative_to_project(pole_detector_weights)}")
    elif skip_missing:
        print(f"⚠ Pole detector weights not found: {pole_detector_weights} (skipped)")
    else:
        raise FileNotFoundError(f"Pole detector weights not found: {pole_detector_weights}")
    
    # Load YOLO ruler detector
    if Path(ruler_detector_weights).exists():
        ruler_detector = YOLO(ruler_detector_weights)
        if torch.cuda.is_available() and hasattr(ruler_detector.model, 'float'):
            ruler_detector.model.float()
        print(f"✓ Ruler detector loaded from {path_relative_to_project(ruler_detector_weights)}")
    elif skip_missing:
        print(f"⚠ Ruler detector weights not found: {ruler_detector_weights} (skipped)")
    else:
        raise FileNotFoundError(f"Ruler detector weights not found: {ruler_detector_weights}")
    
    # Load keypoint model (ruler marking)
    if Path(keypoint_weights).exists():
        keypoint_model = load_trained_keypoint_model(keypoint_weights, device=device)
        print(f"✓ Keypoint model loaded from {path_relative_to_project(keypoint_weights)}")
    elif skip_missing:
        print(f"⚠ Keypoint weights not found: {keypoint_weights} (skipped)")
    else:
        raise FileNotFoundError(f"Keypoint weights not found: {keypoint_weights}")
    
    # Load pole top model
    if Path(pole_top_weights).exists():
        pole_top_model = load_pole_top_model(pole_top_weights, device=device)
        print(f"✓ Pole top model loaded from {path_relative_to_project(pole_top_weights)}")
    elif skip_missing:
        print(f"⚠ Pole top weights not found: {pole_top_weights} (skipped)")
    else:
        raise FileNotFoundError(f"Pole top weights not found: {pole_top_weights}")
    
    loaded = sum(1 for m in [pole_detector, ruler_detector, keypoint_model, pole_top_model] if m is not None)
    print(f"✓ Loaded {loaded}/4 models")
    
    return {
        'device': device,
        'keypoint_model': keypoint_model,
        'pole_top_model': pole_top_model,
        'pole_detector': pole_detector,
        'ruler_detector': ruler_detector
    }


def load_pole_detector(device: Optional[torch.device] = None):
    """Load YOLO pole detection model from runs/pole_detection/weights/best.pt."""
    from ultralytics import YOLO
    from .config import INFERENCE_POLE_WEIGHTS

    if device is None:
        device = get_device()

    weights_path = INFERENCE_POLE_WEIGHTS
    if not weights_path.exists():
        raise FileNotFoundError(f"Pole detector weights not found: {weights_path}")

    detector = YOLO(str(weights_path))
    print(f"✓ Pole detector loaded from {path_relative_to_project(weights_path)}")
    return detector


def load_equipment_detector(device: Optional[torch.device] = None):
    """Load YOLO equipment detection model (Riser, Transformer, Street Light)."""
    from ultralytics import YOLO
    from .config import INFERENCE_EQUIPMENT_WEIGHTS

    if device is None:
        device = get_device()

    weights_path = INFERENCE_EQUIPMENT_WEIGHTS
    if not weights_path.exists():
        raise FileNotFoundError(f"Equipment detector weights not found: {weights_path}")

    detector = YOLO(str(weights_path))
    print(f"✓ Equipment detector loaded from {path_relative_to_project(weights_path)}")
    return detector


def load_attachment_detector(device: Optional[torch.device] = None):
    """Load YOLO attachment detection model (comm, down_guy)."""
    from ultralytics import YOLO
    from .config import INFERENCE_ATTACHMENT_WEIGHTS

    if device is None:
        device = get_device()

    weights_path = INFERENCE_ATTACHMENT_WEIGHTS
    if not weights_path.exists():
        raise FileNotFoundError(f"Attachment detector weights not found: {weights_path}")

    detector = YOLO(str(weights_path))
    print(f"✓ Attachment detector loaded from {path_relative_to_project(weights_path)}")
    return detector


def load_keypoint_detector(equipment_type: str, device: Optional[torch.device] = None):
    """Load HRNet keypoint detection model for given equipment type (riser, transformer, street_light)."""
    from torchvision import transforms
    from .models import KeypointDetector
    from .config import (
        HRNET_WEIGHTS_PATH, EQUIPMENT_KEYPOINT_CONFIGS, IMAGENET_MEAN, IMAGENET_STD,
        RISER_KEYPOINT_NAMES, TRANSFORMER_KEYPOINT_NAMES, STREET_LIGHT_KEYPOINT_NAMES,
        SECONDARY_DRIP_LOOP_KEYPOINT_NAMES,
    )

    if device is None:
        device = get_device()

    if equipment_type not in EQUIPMENT_KEYPOINT_CONFIGS:
        raise ValueError(f"Unknown equipment_type: {equipment_type}. Must be one of {list(EQUIPMENT_KEYPOINT_CONFIGS.keys())}")

    cfg, num_kp, weights_path = EQUIPMENT_KEYPOINT_CONFIGS[equipment_type]
    if not weights_path.exists():
        raise FileNotFoundError(f"{equipment_type} keypoint weights not found: {weights_path}")

    # Create and load model
    model = KeypointDetector(
        num_keypoints=num_kp,
        heatmap_size=(cfg['heatmap_height'], cfg['heatmap_width']),
        weights_path=HRNET_WEIGHTS_PATH
    )
    ckpt = torch.load(str(weights_path), map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).float().eval()

    # Create preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg['resize_height'], cfg['resize_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    print(f"✓ {equipment_type.title()} keypoint detector loaded from {path_relative_to_project(weights_path)}")

    _kp_names_map = {
        'riser': RISER_KEYPOINT_NAMES,
        'transformer': TRANSFORMER_KEYPOINT_NAMES,
        'street_light': STREET_LIGHT_KEYPOINT_NAMES,
        'secondary_drip_loop': SECONDARY_DRIP_LOOP_KEYPOINT_NAMES,
    }
    return {
        'model': model,
        'num_kp': num_kp,
        'kp_names': _kp_names_map[equipment_type],
        'preprocess': preprocess,
        'config': cfg
    }


def load_attachment_keypoint_detector(attachment_type: str, device: Optional[torch.device] = None):
    """Load HRNet keypoint detection model for attachment type (comm, down_guy)."""
    from torchvision import transforms
    from .models import KeypointDetector
    from .config import (
        HRNET_WEIGHTS_PATH, ATTACHMENT_KEYPOINT_CONFIGS, IMAGENET_MEAN, IMAGENET_STD,
        COMM_KEYPOINT_NAMES, DOWN_GUY_KEYPOINT_NAMES,
    )

    if device is None:
        device = get_device()

    if attachment_type not in ATTACHMENT_KEYPOINT_CONFIGS:
        raise ValueError(f"Unknown attachment_type: {attachment_type}. Must be one of {list(ATTACHMENT_KEYPOINT_CONFIGS.keys())}")

    cfg, num_kp, weights_path = ATTACHMENT_KEYPOINT_CONFIGS[attachment_type]
    if not weights_path.exists():
        raise FileNotFoundError(f"{attachment_type} keypoint weights not found: {weights_path}")

    model = KeypointDetector(
        num_keypoints=num_kp,
        heatmap_size=(cfg['heatmap_height'], cfg['heatmap_width']),
        weights_path=HRNET_WEIGHTS_PATH
    )
    ckpt = torch.load(str(weights_path), map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).float().eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg['resize_height'], cfg['resize_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    print(f"✓ {attachment_type} keypoint detector loaded from {path_relative_to_project(weights_path)}")

    _kp_names_map = {
        'comm': COMM_KEYPOINT_NAMES, 'down_guy': DOWN_GUY_KEYPOINT_NAMES,
        'primary': COMM_KEYPOINT_NAMES, 'secondary': COMM_KEYPOINT_NAMES,
        'neutral': COMM_KEYPOINT_NAMES, 'guy': COMM_KEYPOINT_NAMES,
    }
    return {
        'model': model,
        'num_kp': num_kp,
        'kp_names': _kp_names_map[attachment_type],
        'preprocess': preprocess,
        'config': cfg
    }


def run_ruler_marking_inference(
    image_path: Path,
    keypoint_model: Any,
    device: torch.device,
    use_tta: bool = True,
    show_visualization: bool = True
) -> Dict[str, Any]:
    """
    Run ruler marking keypoint inference on a single image.
    
    Args:
        image_path: Path to ruler crop image
        keypoint_model: Loaded keypoint model
        device: torch device
        use_tta: Use test-time augmentation
        show_visualization: Show matplotlib visualization
    
    Returns:
        Dictionary with 'rgb_image', 'resized_image', 'predictions', 'heatmaps', 
        'gt_keypoints', 'used_interp'
    """
    # Run inference
    rgb_image, resized_image, predictions, heatmaps, used_interp = predict_keypoints(
        keypoint_model, image_path, device=device, use_tta=use_tta
    )
    
    # Load ground truth
    gt_keypoints = load_ground_truth_keypoints(image_path, KEYPOINT_NAMES)
    
    # Visualization
    if show_visualization:
        max_heatmap = np.max(heatmaps, axis=0)
        max_heatmap_resized = cv2.resize(
            max_heatmap,
            (rgb_image.shape[1], rgb_image.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

        line_length = rgb_image.shape[1]
        combined_overlay = create_keypoint_overlay_with_overlap(
            rgb_image, predictions, gt_keypoints, line_length
        )

        resized_h, resized_w = resized_image.shape[:2]
        resized_points = []
        for kp in predictions:
            resized_points.append({
                'name': kp['name'],
                'x': int(kp['x'] / rgb_image.shape[1] * resized_w),
                'y': int(kp['y'] / rgb_image.shape[0] * resized_h),
                'conf': kp['conf'],
            })
        resized_overlay = draw_keypoints(
            resized_image, resized_points, (255, 0, 0), line_length=resized_w, style='line'
        )

        heatmap_normalized = (max_heatmap_resized - max_heatmap_resized.min()) / (
            max_heatmap_resized.max() - max_heatmap_resized.min() + 1e-8
        )
        heatmap_colored = plt.cm.hot(heatmap_normalized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        rgb_image_float = rgb_image.astype(np.float32) / 255.0
        heatmap_float = heatmap_colored.astype(np.float32) / 255.0
        heatmap_overlay = (0.5 * rgb_image_float + 0.5 * heatmap_float)
        heatmap_overlay = (heatmap_overlay * 255).astype(np.uint8)

        show_keypoint_inference_3panel(
            combined_overlay,
            resized_overlay,
            heatmap_overlay,
            title1='GT vs Predicted Keypoints',
            title2=f'Resized ({RESIZE_HEIGHT}x{RESIZE_WIDTH}) with Predicted Keypoints',
            title3='Heatmap Overlay on Image',
            show_legend=bool(gt_keypoints),
        )
        
        # Print results
        print("\nPredicted keypoints (with sub-pixel precision):")
        for kp in predictions:
            interp_note = " (interpolated)" if kp.get('interpolated') else ""
            display_conf = kp.get('original_conf', kp['conf'])
            print(
                f"  {kp['name']:>4} ft -> "
                f"(x={kp['x']:6.2f}, y={kp['y']:7.2f}), "
                f"conf={display_conf:.3f}{interp_note}"
            )
        
        if gt_keypoints:
            print("\nPrediction errors:")
            for pred_kp in predictions:
                for gt_kp in gt_keypoints:
                    if pred_kp['name'] == gt_kp['name']:
                        error = abs(pred_kp['y'] - gt_kp['y'])
                        print(f"  {pred_kp['name']:>4} ft: {error:.2f} pixels error")
        
        if used_interp:
            print(f"\n⚠ Interpolation was applied (threshold={CONF_THRESHOLD}).")
        else:
            print(f"\n✓ All keypoints above confidence threshold ({CONF_THRESHOLD}).")
    
    return {
        'rgb_image': rgb_image,
        'resized_image': resized_image,
        'predictions': predictions,
        'heatmaps': heatmaps,
        'gt_keypoints': gt_keypoints,
        'used_interp': used_interp
    }


def run_pole_top_inference(
    image_path: Path,
    pole_top_model: Any,
    device: torch.device,
    use_tta: bool = True,
    show_visualization: bool = True,
) -> Dict[str, Any]:
    """
    Run pole top keypoint inference on a single image.
    
    Args:
        image_path: Path to full pole image
        pole_top_model: Loaded pole top model
        device: torch device
        use_tta: Use test-time augmentation
        show_visualization: Show matplotlib visualization
    
    Returns:
        Dictionary with 'rgb_cropped', 'resized_image', 'prediction', 'heatmap', 
        'gt_keypoint'
    """
    # Run inference
    rgb_cropped, resized_image, prediction, heatmap = predict_pole_top(
        pole_top_model, image_path, device=device, use_tta=use_tta
    )
    
    # Load ground truth
    gt_keypoint = load_ground_truth_pole_top(image_path)
    
    # Visualization (3-panel only: overlay, resized, heatmap)
    if show_visualization:
        h_cropped, w_cropped = rgb_cropped.shape[:2]
        line_length = w_cropped
        pred_list = [prediction]
        gt_list = [gt_keypoint] if gt_keypoint else []

        panel1 = create_keypoint_overlay_with_overlap(
            rgb_cropped, pred_list, gt_list, line_length
        )
        full_pole_bgr = cv2.imread(str(image_path))
        if full_pole_bgr is not None:
            full_pole_rgb = cv2.cvtColor(full_pole_bgr, cv2.COLOR_BGR2RGB)
            full_pole_w = full_pole_rgb.shape[1]
            panel1 = create_keypoint_overlay_with_overlap(
                full_pole_rgb, pred_list, gt_list, full_pole_w
            )

        scale_x = (POLE_TOP_RESIZE_WIDTH - 1) / max(w_cropped - 1, 1)
        scale_y = (POLE_TOP_RESIZE_HEIGHT - 1) / max(h_cropped - 1, 1)
        resized_pred = {'name': 'pole_top', 'x': prediction['x'] * scale_x, 'y': prediction['y'] * scale_y, 'conf': prediction['conf']}
        resized_gt = {'name': 'pole_top', 'x': gt_keypoint['x'] * scale_x, 'y': gt_keypoint['y'] * scale_y} if gt_keypoint else None
        panel2 = create_keypoint_overlay_with_overlap(
            resized_image,
            [resized_pred],
            [resized_gt] if resized_gt else [],
            POLE_TOP_RESIZE_WIDTH,
        )

        heatmap_resized = cv2.resize(heatmap, (resized_image.shape[1], resized_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
        heatmap_colored = plt.cm.hot(heatmap_normalized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        heatmap_overlay = (0.5 * resized_image.astype(np.float32) / 255.0 + 0.5 * heatmap_colored.astype(np.float32) / 255.0) * 255
        panel3 = heatmap_overlay.astype(np.uint8)

        show_keypoint_inference_3panel(
            panel1, panel2, panel3,
            title1='GT vs Predicted Pole Top Keypoint (Full Pole Crop)',
            title2=f'Upper 10% Crop Resized ({POLE_TOP_RESIZE_HEIGHT}x{POLE_TOP_RESIZE_WIDTH})',
            title3='Heatmap Overlay on Resized Image',
            show_legend=bool(gt_keypoint),
        )
        
        # Print results
        print(f"\nPrediction:")
        print(f"  Position: ({prediction['x']:.1f}, {prediction['y']:.1f})")
        print(f"  Confidence: {prediction['conf']:.3f}")
        
        if gt_keypoint:
            pixel_error = abs(prediction['y'] - gt_keypoint['y'])  # Vertical distance
            print(f"\nGround Truth:")
            print(f"  Position: ({gt_keypoint['x']:.1f}, {gt_keypoint['y']:.1f})")
            print(f"  Vertical Pixel Error: {pixel_error:.1f} px")
            if 'ppi' in gt_keypoint and gt_keypoint['ppi'] > 0:
                inch_error = pixel_error / gt_keypoint['ppi']
                print(f"  Vertical Inch Error: {inch_error:.3f} inches")
    
    return {
        'rgb_cropped': rgb_cropped,
        'resized_image': resized_image,
        'prediction': prediction,
        'heatmap': heatmap,
        'gt_keypoint': gt_keypoint
    }


def load_all_ground_truth(
    image_path: Path,
    image_rgb: np.ndarray,
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Load all ground truth annotations for end-to-end inference visualization.
    
    Prioritizes location files (most accurate, global coordinates) over YOLO labels.
    Location files contain: pole bbox, ruler bbox, ruler marking keypoints, and pole top.
    
    Args:
        image_path: Path to the full raw image
        image_rgb: RGB image array (for dimensions)
        results: Inference results dictionary with 'pole', 'ruler', 'keypoints', 'pole_top'
    
    Returns:
        Dictionary with 'gt_pole_bbox', 'gt_ruler_bbox', 'gt_keypoints', 'gt_pole_top'
        All coordinates are in global pixel coordinates.
    """
    h_img, w_img = image_rgb.shape[:2]
    gt_pole_bbox = None
    gt_ruler_bbox = None
    gt_keypoints = []
    gt_pole_top = None
    
    # Location file path (contains all ground truth in global coordinates)
    # Check both pole and midspan locations based on image path
    location_file_path = None
    location_paths = [
        POLE_LABELS_DIR / f"{image_path.stem}_location.txt",
        MIDSPAN_LABELS_DIR / f"{image_path.stem}_location.txt",
    ]
    
    # Try to find location file in either directory
    for loc_path in location_paths:
        if loc_path.exists():
            location_file_path = loc_path
            break
    
    # Load all ground truth from location file first (most accurate, global coordinates)
    if location_file_path and location_file_path.exists():
        from .data_utils import (
            load_pole_bbox_from_location_file,
            load_ruler_bbox_from_location_file,
            load_ruler_marking_keypoints_from_location_file,
            load_pole_top_from_location_file
        )
        
        # Load pole bbox from location file
        gt_pole_bbox = load_pole_bbox_from_location_file(location_file_path, w_img, h_img)
        
        # Load ruler bbox from location file
        gt_ruler_bbox = load_ruler_bbox_from_location_file(location_file_path, w_img, h_img)
        
        # Load keypoints from location file
        gt_keypoints_dict = load_ruler_marking_keypoints_from_location_file(location_file_path, w_img, h_img)
        if gt_keypoints_dict:
            # Convert dictionary format to list format expected by visualization
            # Format height as string to match KEYPOINT_NAMES format (e.g., '2.5', '6.5')
            gt_keypoints = []
            for height, coords in sorted(gt_keypoints_dict.items()):
                # Format height to match KEYPOINT_NAMES format exactly (e.g., '2.5', '6.5', '10.5')
                # Use .1f format to ensure consistent formatting (2.5 not 2.5000000)
                height_str = f'{height:.1f}'
                # Validate coordinates are within image bounds
                x_coord = max(0, min(w_img - 1, coords['x']))
                y_coord = max(0, min(h_img - 1, coords['y']))
                gt_keypoints.append({
                    'name': height_str,
                    'x': x_coord,
                    'y': y_coord,
                    'conf': 1.0,
                    'ppi': None  # PPI not available from location file
                })
        
        # Load pole top from location file
        gt_pole_top_dict = load_pole_top_from_location_file(location_file_path, w_img, h_img)
        if gt_pole_top_dict:
            gt_pole_top = {
                'name': 'pole_top',
                'x': gt_pole_top_dict['x'],
                'y': gt_pole_top_dict['y'],
                'conf': 1.0,
                'ppi': None  # PPI not available from location file
            }
    
    # Fall back to YOLO labels if location file doesn't exist or didn't have the data
    # Try to load pole ground truth bbox from YOLO labels
    if gt_pole_bbox is None:
        pole_label_path = DATASETS_DIR / 'pole_detection' / 'labels' / 'val' / f"{image_path.stem}.txt"
        if pole_label_path.exists():
            pole_boxes = parse_yolo_label(pole_label_path, w_img, h_img)
            if pole_boxes:
                # Assuming class 0 is pole
                for box in pole_boxes:
                    if box['class_id'] == 0:
                        gt_pole_bbox = box['bbox']
                        break
    
    # Try to load ruler ground truth bbox from YOLO labels
    if gt_ruler_bbox is None:
        ruler_label_path = DATASETS_DIR / 'ruler_detection' / 'labels' / 'val' / f"{image_path.stem}.txt"
        if ruler_label_path.exists():
            ruler_boxes = parse_yolo_label(ruler_label_path, w_img, h_img)
            if ruler_boxes:
                # Assuming class 0 is ruler
                for box in ruler_boxes:
                    if box['class_id'] == 0:
                        gt_ruler_bbox = box['bbox']
                        break
    
    # Fall back to ruler crop labels for keypoints if location file didn't have them
    if not gt_keypoints and results.get('ruler'):
        rx1, ry1, rx2, ry2 = results['ruler']
        # Try to find ruler crop image in ruler_marking_detection dataset
        ruler_crop_paths = [
            DATASETS_DIR / 'ruler_marking_detection' / 'images' / 'val' / f"{image_path.stem}.jpg",
            DATASETS_DIR / 'ruler_marking_detection_midspan_yolo' / 'images' / 'val' / f"{image_path.stem}.jpg",
        ]
        for ruler_crop_path in ruler_crop_paths:
            if ruler_crop_path.exists():
                gt_keypoints = load_ground_truth_keypoints(ruler_crop_path, KEYPOINT_NAMES)
                # Convert keypoints from ruler crop coordinates to full image coordinates
                if gt_keypoints and results.get('ruler'):
                    rx1, ry1, rx2, ry2 = results['ruler']
                    ruler_h = ry2 - ry1
                    ruler_w = rx2 - rx1
                    # Load ruler crop to get its dimensions
                    ruler_crop_img = cv2.imread(str(ruler_crop_path))
                    if ruler_crop_img is not None:
                        crop_h, crop_w = ruler_crop_img.shape[:2]
                        # Scale keypoints from crop to full image
                        scale_x = ruler_w / crop_w
                        scale_y = ruler_h / crop_h
                        for kp in gt_keypoints:
                            kp['x'] = rx1 + kp['x'] * scale_x
                            kp['y'] = ry1 + kp['y'] * scale_y
                break
    
    # Fall back to YOLO labels for pole top if location file didn't have it
    if gt_pole_top is None:
        gt_pole_top = load_ground_truth_pole_top(image_path)
    
    return {
        'gt_pole_bbox': gt_pole_bbox,
        'gt_ruler_bbox': gt_ruler_bbox,
        'gt_keypoints': gt_keypoints,
        'gt_pole_top': gt_pole_top
    }


def create_end_to_end_visualization(
    image_rgb: np.ndarray,
    results: Dict[str, Any],
    gt_data: Dict[str, Any],
    show_plot: bool = True
) -> np.ndarray:
    """
    Single-image overlay: GT (green) + Predicted (red) on same image.
    No text labels. Rectangles for bboxes, horizontal lines for keypoints.
    """
    from .visualization import (
        draw_bboxes, COLOR_GT, COLOR_PRED, VIZ_BBOX_THICKNESS,
        VIZ_FIG_DEFAULTS, rgb_tuple_to_hex,
    )
    from matplotlib.patches import Patch

    gt_viz = gt_data_to_viz_data(gt_data)
    pred_viz = inference_results_to_viz_data(results)
    h, w = image_rgb.shape[:2]
    line_length = (results.get('ruler') and (results['ruler'][2] - results['ruler'][0])) or max(10, w // 20)

    vis = image_rgb.copy()

    # Draw GT (green) — bboxes + keypoints
    if gt_viz.get('bboxes'):
        vis = draw_bboxes(vis, gt_viz['bboxes'], color=COLOR_GT, show_label=True, line_width=VIZ_BBOX_THICKNESS)
    if gt_viz.get('keypoints'):
        vis = draw_keypoints(vis, gt_viz['keypoints'], COLOR_GT, line_length=line_length, style='line', show_labels=True)

    # Draw Pred (red) — bboxes + keypoints with confidence
    if pred_viz.get('bboxes'):
        vis = draw_bboxes(vis, pred_viz['bboxes'], color=COLOR_PRED, show_label=True, show_conf=True, line_width=VIZ_BBOX_THICKNESS)
    if pred_viz.get('keypoints'):
        vis = draw_keypoints(vis, pred_viz['keypoints'], COLOR_PRED, line_length=line_length, style='line', show_labels=True)

    if show_plot:
        setup_matplotlib_style()
        fig, ax = plt.subplots(1, 1, figsize=(12, 16), facecolor=VIZ_FIG_DEFAULTS['facecolor'], dpi=VIZ_FIG_DEFAULTS['dpi'])
        fig.patch.set_facecolor(VIZ_FIG_DEFAULTS['facecolor'])
        ax.imshow(vis)
        ax.axis('off')
        ax.set_title('End-to-End Inference: GT vs Predicted', color='black',
                     fontsize=VIZ_FIG_DEFAULTS['title_fontsize_large'], fontweight=VIZ_FIG_DEFAULTS['title_fontweight'], pad=15)
        ax.set_facecolor(VIZ_FIG_DEFAULTS['facecolor'])
        legend_elements = [
            Patch(facecolor=rgb_tuple_to_hex(COLOR_GT), edgecolor='black', linewidth=1.5, label='Ground Truth'),
            Patch(facecolor=rgb_tuple_to_hex(COLOR_PRED), edgecolor='black', linewidth=1.5, label='Predicted'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True, fancybox=True)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        plt.show()

    return vis


def print_detailed_summary(
    results: Dict[str, Any],
    gt_data: Dict[str, Any],
    image_path=None,
    ppi: float = None,
) -> None:
    """
    Print detailed summary of inference results with ground truth comparison.

    Args:
        results: Inference results dictionary
        gt_data: Ground truth data dictionary from load_all_ground_truth()
        image_path: Optional path for image name and relative path display
        ppi: Optional PPI for inches error on keypoints
    """
    from .config import path_relative_to_project

    gt_pole_bbox = gt_data.get('gt_pole_bbox')
    gt_ruler_bbox = gt_data.get('gt_ruler_bbox')
    gt_keypoints = gt_data.get('gt_keypoints', [])
    gt_pole_top = gt_data.get('gt_pole_top')

    if image_path:
        path = Path(image_path)
        print(f"\nImage: {path.name}")
        print(f"Path: {path_relative_to_project(path)}")

    gt_bbox_count = (1 if gt_pole_bbox else 0) + (1 if gt_ruler_bbox else 0)
    gt_kp_count = len(gt_keypoints) + (1 if gt_pole_top else 0)
    pred_bbox_count = (1 if results.get('pole') else 0) + (1 if results.get('ruler') else 0)
    pred_kps = (results.get('keypoints') or []) + ([results['pole_top']] if results.get('pole_top') else [])
    print(f"GT objects: {gt_bbox_count} bboxes, {gt_kp_count} keypoints")
    print(f"Pred objects: {pred_bbox_count} bboxes, {len(pred_kps)} keypoints")

    print(f"\n{'='*60}")
    print("Detailed Results")
    print(f"{'='*60}")
    
    if results.get('pole'):
        px1, py1, px2, py2 = results['pole']
        print(f"✓ Pole detected: ({px1}, {py1}) -> ({px2}, {py2})")
        print(f"  Size: {px2-px1}x{py2-py1} pixels")
        if gt_pole_bbox:
            px1_gt, py1_gt, px2_gt, py2_gt = gt_pole_bbox
            print(f"  Ground Truth: ({px1_gt}, {py1_gt}) -> ({px2_gt}, {py2_gt})")
    else:
        print("✗ Pole not detected")
    
    if results.get('ruler'):
        rx1, ry1, rx2, ry2 = results['ruler']
        print(f"✓ Ruler detected: ({rx1}, {ry1}) -> ({rx2}, {ry2})")
        print(f"  Size: {rx2-rx1}x{ry2-ry1} pixels")
        if gt_ruler_bbox:
            rx1_gt, ry1_gt, rx2_gt, ry2_gt = gt_ruler_bbox
            print(f"  Ground Truth: ({rx1_gt}, {ry1_gt}) -> ({rx2_gt}, {ry2_gt})")
    else:
        print("✗ Ruler not detected")
    
    if results.get('keypoints'):
        print(f"✓ Ruler markings detected: {len(results['keypoints'])} keypoints")
        for kp in results['keypoints']:
            err_in = ""
            if gt_keypoints and ppi and ppi > 0:
                gt_k = next((g for g in gt_keypoints if str(g['name']) == str(kp['name'])), None)
                if gt_k:
                    err_px = abs(kp['y'] - gt_k['y'])
                    err_in = f", error={err_px / ppi:.3f} in"
            print(f"  - {kp['name']} ft: y={kp['y']:.1f} px, conf={kp['conf']:.3f}{err_in}")
        if gt_keypoints:
            print(f"  Ground Truth: {len(gt_keypoints)} keypoints")
    else:
        print("✗ No ruler markings detected")
    
    if results.get('pole_top'):
        pt = results['pole_top']
        err_in = ""
        if gt_pole_top and ppi and ppi > 0:
            err_px = abs(pt['y'] - gt_pole_top['y'])
            err_in = f", error={err_px / ppi:.3f} in"
        print(f"✓ Pole top detected: ({pt['x']:.1f}, {pt['y']:.1f}), conf={pt['conf']:.3f}{err_in}")
        if gt_pole_top:
            print(f"  Ground Truth: ({gt_pole_top['x']:.1f}, {gt_pole_top['y']:.1f})")
    else:
        print("✗ Pole top not detected")
    
    print(f"{'='*60}")


def run_end_to_end_inference_simple(
    image_path: Path,
    models: Dict[str, Any],
    use_tta: bool = True,
    show_visualization: bool = True
) -> Dict[str, Any]:
    """
    Run complete end-to-end inference pipeline on a single image.
    
    Args:
        image_path: Path to full raw image
        models: Dictionary from load_all_models()
        use_tta: Use test-time augmentation
        show_visualization: Show matplotlib visualization
    
    Returns:
        Dictionary with inference results
    """
    device = models['device']
    pole_detector = models['pole_detector']
    ruler_detector = models['ruler_detector']
    keypoint_model = models['keypoint_model']
    pole_top_model = models['pole_top_model']
    
    # Load image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Run end-to-end inference
    results = run_end_to_end_inference(
        image_bgr, image_rgb,
        pole_detector, ruler_detector,
        keypoint_model, pole_top_model,
        device
    )
    
    # Visualization
    if show_visualization:
        # Load ground truth data
        gt_data = load_all_ground_truth(image_path, image_rgb, results)
        
        # Create visualization
        create_end_to_end_visualization(image_rgb, results, gt_data, show_plot=True)
        
        # Print summary
        print("\n" + "="*60)
        print("Inference Results Summary")
        print("="*60)
        print(f"Pole detected: {'Yes' if results.get('pole') else 'No'}")
        print(f"Ruler detected: {'Yes' if results.get('ruler') else 'No'}")
        if results.get('keypoints'):
            print(f"Keypoints detected: {len(results['keypoints'])}")
            for kp in results['keypoints']:
                print(f"  {kp['name']}: ({kp['x']:.1f}, {kp['y']:.1f}), conf={kp['conf']:.3f}")
        print(f"Pole top detected: {'Yes' if results.get('pole_top') else 'No'}")
        if results.get('pole_top'):
            pt = results['pole_top']
            print(f"  Position: ({pt['x']:.1f}, {pt['y']:.1f}), conf={pt['conf']:.3f}")
            # Show ground truth if available
            gt_pole_top = gt_data.get('gt_pole_top')
            if gt_pole_top:
                print(f"  Ground Truth: ({gt_pole_top['x']:.1f}, {gt_pole_top['y']:.1f})")
                error = abs(pt['y'] - gt_pole_top['y'])
                print(f"  Error: {error:.1f} pixels")
        print("="*60)
    
    return results


def _convert_to_json_serializable(obj):
    """Convert numpy and torch types to native Python types for JSON serialization."""
    # Handle None
    if obj is None:
        return None
    
    # Check for numpy types by checking the type name (more robust)
    obj_type = type(obj).__name__
    if 'int' in obj_type and 'numpy' in str(type(obj)):
        return int(obj)
    elif 'float' in obj_type and 'numpy' in str(type(obj)):
        return float(obj)
    elif 'bool' in obj_type and 'numpy' in str(type(obj)):
        return bool(obj)
    
    # Handle numpy integer types (compatible with NumPy 2.0)
    if isinstance(obj, np.integer):
        return int(obj)
    # Handle numpy floating types (compatible with NumPy 2.0)
    elif isinstance(obj, np.floating):
        return float(obj)
    # Handle numpy boolean
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Handle numpy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle torch tensor types
    elif hasattr(obj, 'item') and not isinstance(obj, (dict, list, tuple, str)):  # torch scalar
        try:
            return obj.item()
        except (AttributeError, ValueError, RuntimeError):
            pass
    elif hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):  # torch tensor
        try:
            return obj.cpu().numpy().tolist()
        except (AttributeError, ValueError, RuntimeError):
            pass
    # Handle collections
    elif isinstance(obj, dict):
        return {key: _convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        # Last resort: try to convert if it looks like a numpy scalar
        try:
            if hasattr(obj, 'dtype'):
                if np.issubdtype(obj.dtype, np.integer):
                    return int(obj)
                elif np.issubdtype(obj.dtype, np.floating):
                    return float(obj)
        except (AttributeError, TypeError):
            pass
        return obj


def run_batch_inference(
    images_dir: Path,
    output_dir: Path,
    models: Dict[str, Any],
    use_tta: bool = True,
    save_annotated: bool = True,
    save_labels: bool = True
) -> List[Dict[str, Any]]:
    """
    Run batch inference on multiple images.
    
    Args:
        images_dir: Directory containing input images
        output_dir: Directory to save results
        models: Dictionary from load_all_models()
        use_tta: Use test-time augmentation
        save_annotated: Save annotated images
        save_labels: Save label files (JSON)
    
    Returns:
        List of inference result dictionaries
    """
    from tqdm import tqdm
    import json
    import warnings
    warnings.filterwarnings('ignore')
    
    device = models['device']
    pole_detector = models['pole_detector']
    ruler_detector = models['ruler_detector']
    keypoint_model = models['keypoint_model']
    pole_top_model = models['pole_top_model']
    
    # Create output directories
    annotated_dir = output_dir / 'annotated_photos'
    labels_dir = output_dir / 'labels'
    if save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)
    if save_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image files
    image_files = sorted(images_dir.glob('*.jpg'))
    if not image_files:
        raise RuntimeError(f"No images found in {images_dir}")
    
    print(f"Processing {len(image_files)} images...")
    
    all_results = []
    
    for image_path in tqdm(image_files, desc="Running inference"):
        try:
            # Load image
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"Warning: Could not load {image_path.name}")
                continue
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = run_end_to_end_inference(
                image_bgr, image_rgb,
                pole_detector, ruler_detector,
                keypoint_model, pole_top_model,
                device
            )
            
            # Create annotated image
            if save_annotated:
                vis_image = image_rgb.copy()
                
                # Draw pole bbox (RED for predictions)
                if results['pole']:
                    px1, py1, px2, py2 = results['pole']
                    cv2.rectangle(vis_image, (px1, py1), (px2, py2), (255, 0, 0), 4)  # Red in RGB
                    put_text_with_border(vis_image, 'Pole', (px1, py1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Draw ruler bbox (RED for predictions)
                if results['ruler']:
                    rx1, ry1, rx2, ry2 = results['ruler']
                    cv2.rectangle(vis_image, (rx1, ry1), (rx2, ry2), (255, 0, 0), 4)  # Red in RGB
                    put_text_with_border(vis_image, 'Ruler', (rx1, ry1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Calculate ruler bbox width for keypoint line length
                ruler_line_length = 20  # Default fallback
                if results['ruler']:
                    rx1, ry1, rx2, ry2 = results['ruler']
                    ruler_line_length = rx2 - rx1
                
                # Draw keypoints with lines extending to ruler bbox width (RED for predictions)
                if results['keypoints']:
                    for kp in results['keypoints']:
                        x_int = int(round(kp['x']))
                        y_int = int(round(kp['y']))
                        # Draw line extending to the right (no dot at tip)
                        cv2.line(vis_image, (x_int, y_int), (x_int + ruler_line_length, y_int), (255, 0, 0), 2)
                        # Add label
                        put_text_with_border(vis_image, kp['name'], (x_int + ruler_line_length + 5, y_int - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Use ruler line length for pole top lines (same as ruler marking lines)
                # pole_line_length is now the same as ruler_line_length
                
                # Draw pole top with line extending to ruler bbox width (same length as ruler marking lines)
                if results['pole_top']:
                    pt = results['pole_top']
                    x_int = int(round(pt['x']))
                    y_int = int(round(pt['y']))
                    # Draw line extending to the right (no dot at tip) - same length as ruler marking lines
                    cv2.line(vis_image, (x_int, y_int), (x_int + ruler_line_length, y_int), (255, 0, 0), 2)
                    put_text_with_border(vis_image, 'Pole Top', (x_int + 10, y_int),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Save annotated image
                vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                output_path = annotated_dir / image_path.name
                cv2.imwrite(str(output_path), vis_bgr)
            
            # Save labels
            if save_labels:
                label_data = {
                    'image': image_path.name,
                    'pole': results['pole'],
                    'ruler': results['ruler'],
                    'keypoints': results['keypoints'],
                    'pole_top': results['pole_top']
                }
                # Convert numpy types to native Python types for JSON serialization
                label_data = _convert_to_json_serializable(label_data)
                label_path = labels_dir / f"{image_path.stem}.json"
                
                # Custom JSON encoder as fallback
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, np.bool_):
                            return bool(obj)
                        return super().default(obj)
                
                try:
                    with open(label_path, 'w') as f:
                        json.dump(label_data, f, indent=2, cls=NumpyEncoder)
                except (TypeError, ValueError) as e:
                    # If conversion failed, try one more aggressive pass
                    print(f"Warning: JSON serialization issue for {image_path.name}: {e}")
                    label_data = _convert_to_json_serializable(label_data)
                    with open(label_path, 'w') as f:
                        json.dump(label_data, f, indent=2, cls=NumpyEncoder)
            
            results['image_path'] = image_path
            all_results.append(results)
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue
    
    print(f"\n✓ Processed {len(all_results)}/{len(image_files)} images")
    print(f"  Annotated images saved to: {annotated_dir}")
    print(f"  Labels saved to: {labels_dir}")
    
    return all_results
