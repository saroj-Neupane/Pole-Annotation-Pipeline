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

from .inference_utils import (
    load_trained_keypoint_model,
    load_pole_top_model,
    predict_keypoints,
    predict_pole_top,
    run_end_to_end_inference,
)
from .data_utils import (
    load_ground_truth_keypoints,
    load_ground_truth_pole_top
)
from .config import (
    path_relative_to_project,
    KEYPOINT_NAMES,
    INFERENCE_POLE_WEIGHTS,
    INFERENCE_RULER_WEIGHTS,
    INFERENCE_RULER_MARKING_WEIGHTS,
    INFERENCE_POLE_TOP_WEIGHTS,
)


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
) -> Dict[str, Any]:
    """
    Run ruler marking keypoint inference on a single image.
    
    Args:
        image_path: Path to ruler crop image
        keypoint_model: Loaded keypoint model
        device: torch device
        use_tta: Use test-time augmentation
    
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
) -> Dict[str, Any]:
    """
    Run pole top keypoint inference on a single image.
    
    Args:
        image_path: Path to full pole image
        pole_top_model: Loaded pole top model
        device: torch device
        use_tta: Use test-time augmentation
    
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
    
    return {
        'rgb_cropped': rgb_cropped,
        'resized_image': resized_image,
        'prediction': prediction,
        'heatmap': heatmap,
        'gt_keypoint': gt_keypoint
    }


def run_end_to_end_inference_simple(
    image_path: Path,
    models: Dict[str, Any],
    use_tta: bool = True,
) -> Dict[str, Any]:
    """
    Run complete end-to-end inference pipeline on a single image.
    
    Args:
        image_path: Path to full raw image
        models: Dictionary from load_all_models()
        use_tta: Use test-time augmentation
    
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
