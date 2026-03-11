"""
Inference utilities for model loading and prediction.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from typing import Optional, Dict, List, Tuple, Any

from .models import KeypointDetector
from .config import (
    KEYPOINT_NAMES, NUM_KEYPOINTS, RESIZE_HEIGHT, RESIZE_WIDTH,
    IMAGENET_MEAN, IMAGENET_STD,
    HEATMAP_HEIGHT, HEATMAP_WIDTH, HRNET_WEIGHTS_PATH,
    POLE_TOP_RESIZE_HEIGHT, POLE_TOP_RESIZE_WIDTH,
    POLE_TOP_HEATMAP_HEIGHT, POLE_TOP_HEATMAP_WIDTH, POLE_TOP_NUM_KEYPOINTS,
    POLE_DETECTION_CONFIG, RULER_DETECTION_CONFIG,
    INFERENCE_POLE_CONF_THRESHOLD,
    INFERENCE_RULER_CONF_THRESHOLD,
    INFERENCE_MAX_DETECTIONS,
    INFERENCE_USE_TTA,
    INFERENCE_USE_INTERPOLATION,
    RULER_MARKING_WEIGHTS,
    POLE_TOP_WEIGHT_ALONE,
    POLE_PHOTO_CONFIDENCE_WEIGHTS,
    INFERENCE_RULER_MARKING_WEIGHTS,
    INFERENCE_POLE_TOP_WEIGHTS
)

# Only import interpolate_keypoints if interpolation is enabled
# (training_utils imports tensorboard which is a training-only dependency)
if INFERENCE_USE_INTERPOLATION:
    from .training_utils import interpolate_keypoints


# Preprocessing transforms
PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

POLE_TOP_PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((POLE_TOP_RESIZE_HEIGHT, POLE_TOP_RESIZE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def calculate_weighted_confidence(keypoints: List[Dict[str, Any]], weights: Dict[str, float]) -> float:
    """
    Calculate weighted confidence score across multiple keypoints.

    Weights are based on correlation between confidence and prediction error.
    Keypoints with stronger correlation (more reliable) get higher weights.

    Args:
        keypoints: List of keypoint dicts with 'name' and 'conf' keys
        weights: Dict mapping keypoint names to weight values

    Returns:
        float: Weighted average confidence score (0.0-1.0)
    """
    if not keypoints:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0

    for kp in keypoints:
        kp_name = kp.get('name')
        if kp_name in weights:
            weighted_sum += kp['conf'] * weights[kp_name]
            total_weight += weights[kp_name]

    if total_weight > 0:
        return weighted_sum / total_weight

    # Fallback: simple average if weights not found
    return np.mean([kp['conf'] for kp in keypoints])


@torch.no_grad()
def load_trained_keypoint_model(
    weights_path: str = None,
    device: Optional[torch.device] = None
) -> KeypointDetector:
    """Load trained keypoint model with FP32 precision for maximum accuracy."""
    if weights_path is None:
        weights_path = INFERENCE_RULER_MARKING_WEIGHTS
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = KeypointDetector(
        num_keypoints=NUM_KEYPOINTS,
        heatmap_size=(HEATMAP_HEIGHT, HEATMAP_WIDTH),
        weights_path=HRNET_WEIGHTS_PATH
    )
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.float()  # Explicitly set to FP32 precision for high accuracy
    model.eval()
    print("Ruler marking keypoint model loaded with FP32 precision")
    return model


@torch.no_grad()
def load_pole_top_model(
    weights_path: str = None,
    device: Optional[torch.device] = None
) -> KeypointDetector:
    """Load trained pole top keypoint model with FP32 precision for maximum accuracy."""
    if weights_path is None:
        weights_path = INFERENCE_POLE_TOP_WEIGHTS
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = KeypointDetector(
        num_keypoints=POLE_TOP_NUM_KEYPOINTS,
        heatmap_size=(POLE_TOP_HEATMAP_HEIGHT, POLE_TOP_HEATMAP_WIDTH),
        weights_path=HRNET_WEIGHTS_PATH
    )
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.float()  # Explicitly set to FP32 precision for high accuracy
    model.eval()
    print("Pole top keypoint model loaded with FP32 precision")
    return model


def infer_keypoints_on_crop(
    model: KeypointDetector,
    crop_rgb: np.ndarray,
    device: torch.device,
    use_tta: bool = False,
    use_interpolation: bool = True
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Infer keypoints with sub-pixel refinement on ruler crop.
    
    Args:
        model: trained KeypointDetector for ruler markings
        crop_rgb: RGB image array of ruler crop
        device: torch device
        use_tta: if True, use test-time augmentation with vertical shifts
        use_interpolation: if True, apply keypoint interpolation
    
    Returns:
        Tuple of (keypoints list, heatmaps array)
    """
    h, w = crop_rgb.shape[:2]
    
    # Check if model is in FP16 mode
    is_fp16 = next(model.parameters()).dtype == torch.float16
    
    if use_tta:
        # Improved TTA: Average heatmaps instead of coordinates, use better border handling
        shifts = [-2, 0, 2]
        accumulated_heatmaps_list = None  # Will store list of heatmap arrays for each keypoint
        
        for shift in shifts:
            # Apply vertical shift with better border handling (reflect/replicate instead of black borders)
            # Use BORDER_REFLECT_101 (mirror padding) to avoid edge artifacts
            M = np.float32([[1, 0, 0], [0, 1, shift]])
            shifted = cv2.warpAffine(
                crop_rgb, M, (w, h),
                borderMode=cv2.BORDER_REFLECT_101,  # Mirror padding avoids edge artifacts
                flags=cv2.INTER_LINEAR
            )
            
            resized_rgb = cv2.resize(shifted, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            tensor = PREPROCESS(resized_rgb).unsqueeze(0).to(device)
            if is_fp16:
                tensor = tensor.half()
            with torch.no_grad():
                logits = model(tensor)
                heatmaps = torch.sigmoid(logits)[0].detach().cpu().numpy()
            
            # Transform heatmaps back to account for shift before accumulating
            # For each keypoint heatmap, align it with unshifted coordinate space
            aligned_heatmaps = []
            for idx, hm in enumerate(heatmaps):
                if shift != 0:
                    # Shift heatmap to align with unshifted coordinate space
                    # The heatmap corresponds to shifted image, so we need to shift it back
                    shift_px_in_resized = shift / h * RESIZE_HEIGHT
                    M_heatmap = np.float32([[1, 0, 0], [0, 1, -shift_px_in_resized]])
                    hm_aligned = cv2.warpAffine(
                        hm, M_heatmap, (hm.shape[1], hm.shape[0]),
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
                        flags=cv2.INTER_LINEAR
                    )
                else:
                    hm_aligned = hm
                aligned_heatmaps.append(hm_aligned)
            
            # Initialize or accumulate heatmaps
            if accumulated_heatmaps_list is None:
                accumulated_heatmaps_list = [[hm] for hm in aligned_heatmaps]
            else:
                for idx, hm in enumerate(aligned_heatmaps):
                    accumulated_heatmaps_list[idx].append(hm)
        
        # Average heatmaps across all shifts for each keypoint
        averaged_heatmaps = []
        for heatmap_list in accumulated_heatmaps_list:
            averaged_hm = np.mean(heatmap_list, axis=0)
            averaged_heatmaps.append(averaged_hm)
        
        # Extract peaks from averaged heatmaps
        keypoints = []
        for idx, hm in enumerate(averaged_heatmaps):
            y_int, x_int = np.unravel_index(np.argmax(hm), hm.shape)
            y_sub, x_sub = float(y_int), float(x_int)
            conf = float(hm[y_int, x_int])
            
            # Scale to original image with sub-pixel precision
            x_px = x_sub / max(hm.shape[1] - 1, 1) * (w - 1) if w > 1 else x_sub
            y_px = y_sub / max(hm.shape[0] - 1, 1) * (h - 1) if h > 1 else y_sub
            
            keypoints.append({
                'name': KEYPOINT_NAMES[idx],
                'x': x_px,
                'y': y_px,
                'conf': conf
            })
        
        # Use averaged heatmaps for visualization
        heatmaps = np.array(averaged_heatmaps)
    else:
        # No TTA - single inference
        tensor = PREPROCESS(crop_rgb).unsqueeze(0).to(device)
        if is_fp16:
            tensor = tensor.half()
        with torch.no_grad():
            logits = model(tensor)
            heatmaps = torch.sigmoid(logits)[0].detach().cpu().numpy()

        keypoints = []
        for idx, hm in enumerate(heatmaps):
            y_int, x_int = np.unravel_index(np.argmax(hm), hm.shape)
            y_sub, x_sub = float(y_int), float(x_int)
            confidence = float(hm[y_int, x_int])
            
            x_px = x_sub / max(hm.shape[1] - 1, 1) * (w - 1) if w > 1 else x_sub
            y_px = y_sub / max(hm.shape[0] - 1, 1) * (h - 1) if h > 1 else y_sub
            
            keypoints.append({
                'name': KEYPOINT_NAMES[idx],
                'x': x_px,
                'y': y_px,
                'conf': confidence
            })

    # Use interpolation if available
    if use_interpolation:
        try:
            keypoints, _ = interpolate_keypoints(keypoints, threshold=0.0)
        except Exception:
            pass  # interpolate_keypoints not available, skip

    # Calculate weighted confidence (new metric)
    weighted_conf = calculate_weighted_confidence(keypoints, RULER_MARKING_WEIGHTS)

    # Add weighted confidence to keypoints list metadata
    for kp in keypoints:
        kp['weighted_conf'] = weighted_conf  # Same for all keypoints in this crop

    return keypoints, heatmaps


def infer_pole_top_on_crop(
    model: KeypointDetector,
    pole_crop_rgb: np.ndarray,
    device: torch.device,
    use_tta: bool = False
) -> Optional[Dict[str, float]]:
    """Infer pole top keypoint with sub-pixel refinement.
    
    Crops to upper 10% of pole crop (where pole top is always located) before inference.
    Coordinates are transformed back to original pole crop space.
    
    Args:
        model: trained KeypointDetector for pole top
        pole_crop_rgb: RGB image array of pole crop
        device: torch device
        use_tta: if True, use test-time augmentation with vertical shifts
    
    Returns:
        Dictionary with 'x', 'y', 'conf' keys, or None if model is None
    """
    if model is None:
        return None
    h_original, w_original = pole_crop_rgb.shape[:2]
    
    # Check if model is in FP16 mode
    is_fp16 = next(model.parameters()).dtype == torch.float16
    
    # Crop to upper 10% of image (pole top is always in upper 10%)
    crop_height = int(h_original * 0.1)
    pole_crop_cropped = pole_crop_rgb[0:crop_height, :]
    h_cropped, w_cropped = pole_crop_cropped.shape[:2]
    
    if use_tta:
        # Improved TTA: Average heatmaps instead of coordinates, use better border handling
        shifts = [-2, 0, 2]
        accumulated_heatmaps = []
        
        for shift in shifts:
            # Apply vertical shift with better border handling (reflect/replicate instead of black borders)
            # Use BORDER_REFLECT_101 (mirror padding) to avoid edge artifacts
            M = np.float32([[1, 0, 0], [0, 1, shift]])
            shifted = cv2.warpAffine(
                pole_crop_cropped, M, (w_cropped, h_cropped),
                borderMode=cv2.BORDER_REFLECT_101,  # Mirror padding avoids edge artifacts
                flags=cv2.INTER_LINEAR
            )
            
            resized_rgb = cv2.resize(shifted, (POLE_TOP_RESIZE_WIDTH, POLE_TOP_RESIZE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            tensor = POLE_TOP_PREPROCESS(resized_rgb).unsqueeze(0).to(device)
            if is_fp16:
                tensor = tensor.half()
            with torch.no_grad():
                logits = model(tensor)
                heatmap = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            
            # Transform heatmap coordinates back to account for shift before accumulating
            # Create inverse transform to align heatmaps in original coordinate space
            if shift != 0:
                # Shift heatmap to align with unshifted coordinate space
                # The heatmap corresponds to shifted image, so we need to shift it back
                shift_px_in_resized = shift / h_cropped * POLE_TOP_RESIZE_HEIGHT
                M_heatmap = np.float32([[1, 0, 0], [0, 1, -shift_px_in_resized]])
                heatmap = cv2.warpAffine(
                    heatmap, M_heatmap, (heatmap.shape[1], heatmap.shape[0]),
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
                    flags=cv2.INTER_LINEAR
                )
            
            accumulated_heatmaps.append(heatmap)
        
        # Average heatmaps across all shifts
        averaged_heatmap = np.mean(accumulated_heatmaps, axis=0)
        
        # Extract peak from averaged heatmap
        y_int, x_int = np.unravel_index(np.argmax(averaged_heatmap), averaged_heatmap.shape)
        y_sub, x_sub = float(y_int), float(x_int)
        conf = float(averaged_heatmap[y_int, x_int])
        
        # Transform coordinates: heatmap -> resized -> cropped -> original pole crop
        x_resized = x_sub / max(averaged_heatmap.shape[1] - 1, 1) * (POLE_TOP_RESIZE_WIDTH - 1)
        y_resized = y_sub / max(averaged_heatmap.shape[0] - 1, 1) * (POLE_TOP_RESIZE_HEIGHT - 1)
        x_cropped = x_resized / POLE_TOP_RESIZE_WIDTH * w_cropped
        y_cropped = y_resized / POLE_TOP_RESIZE_HEIGHT * h_cropped
        # y in cropped = y in original since crop starts at y=0
        x_px = x_cropped
        y_px = y_cropped

        # Note: For pole top (single keypoint), weighted_conf = conf
        return {'x': x_px, 'y': y_px, 'conf': conf, 'weighted_conf': conf}
    else:
        # No TTA - single inference
        tensor = POLE_TOP_PREPROCESS(pole_crop_cropped).unsqueeze(0).to(device)
        if is_fp16:
            tensor = tensor.half()
        with torch.no_grad():
            logits = model(tensor)
            heatmaps = torch.sigmoid(logits)[0].detach().cpu().numpy()
        
        hm = heatmaps[0]
        y_int, x_int = np.unravel_index(np.argmax(hm), hm.shape)
        y_sub, x_sub = float(y_int), float(x_int)
        confidence = float(hm[y_int, x_int])
        
        # Transform coordinates: heatmap -> resized -> cropped -> original pole crop
        x_resized = x_sub / max(hm.shape[1] - 1, 1) * (POLE_TOP_RESIZE_WIDTH - 1)
        y_resized = y_sub / max(hm.shape[0] - 1, 1) * (POLE_TOP_RESIZE_HEIGHT - 1)
        x_cropped = x_resized / POLE_TOP_RESIZE_WIDTH * w_cropped
        y_cropped = y_resized / POLE_TOP_RESIZE_HEIGHT * h_cropped
        # y in cropped = y in original since crop starts at y=0
        x_px = x_cropped
        y_px = y_cropped

        # Note: For pole top (single keypoint), weighted_conf = conf
        return {'x': x_px, 'y': y_px, 'conf': confidence, 'weighted_conf': confidence}


def predict_keypoints(
    model: KeypointDetector,
    image_path: Path,
    device: torch.device,
    use_tta: Optional[bool] = None,
    use_interpolation: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray, List[Dict], np.ndarray, bool]:
    """
    Predict keypoints on a ruler crop image with sub-pixel refinement.
    
    Args:
        model: trained KeypointDetector
        image_path: path to ruler crop image
        device: torch device
        use_tta: if True, use test-time augmentation with vertical shifts
    
    Returns:
        Tuple of (rgb, resized_rgb, points, heatmaps, used_interp)
        - rgb: original RGB image
        - resized_rgb: resized RGB image
        - points: list of keypoint dictionaries
        - heatmaps: heatmaps from model
        - used_interp: whether interpolation was used
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    # Use config defaults if not provided
    if use_tta is None:
        use_tta = INFERENCE_USE_TTA
    if use_interpolation is None:
        use_interpolation = INFERENCE_USE_INTERPOLATION
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb.shape[:2]

    if use_tta:
        # Test-time augmentation with small vertical shifts
        shifts = [-2, 0, 2]
        all_points = []
        
        for shift in shifts:
            # Apply vertical shift
            M = np.float32([[1, 0, 0], [0, 1, shift]])
            shifted = cv2.warpAffine(rgb, M, (orig_w, orig_h))
            
            resized_rgb = cv2.resize(shifted, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            tensor = PREPROCESS(resized_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                heatmaps = torch.sigmoid(logits)[0].detach().cpu().numpy()
            
            shift_points = []
            for idx, hm in enumerate(heatmaps):
                y_int, x_int = np.unravel_index(np.argmax(hm), hm.shape)
                # Sub-pixel refinement
                y_sub, x_sub = float(y_int), float(x_int)
                conf = float(hm[y_int, x_int])
                
                # Scale to original image with sub-pixel precision
                # Use consistent scaling with training: (size - 1) for pixel-perfect alignment
                x_px = x_sub / max(hm.shape[1] - 1, 1) * (orig_w - 1) if orig_w > 1 else x_sub
                y_px = (y_sub / max(hm.shape[0] - 1, 1) * (orig_h - 1) if orig_h > 1 else y_sub) - shift  # Undo shift
                shift_points.append({'x': x_px, 'y': y_px, 'conf': conf})
            
            all_points.append(shift_points)
        
        # Average predictions across shifts
        points = []
        for idx in range(NUM_KEYPOINTS):
            avg_x = np.mean([p[idx]['x'] for p in all_points])
            avg_y = np.mean([p[idx]['y'] for p in all_points])
            avg_conf = np.mean([p[idx]['conf'] for p in all_points])
            points.append({
                'name': KEYPOINT_NAMES[idx],
                'x': avg_x,
                'y': avg_y,
                'conf': avg_conf
            })
        
        # Use original image for return (no shift)
        resized_rgb = cv2.resize(rgb, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        tensor = PREPROCESS(resized_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            heatmaps = torch.sigmoid(logits)[0].detach().cpu().numpy()
    else:
        # Standard inference with sub-pixel refinement
        resized_rgb = cv2.resize(rgb, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        tensor = PREPROCESS(resized_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            heatmaps = torch.sigmoid(logits)[0].detach().cpu().numpy()

        points = []
        for idx, hm in enumerate(heatmaps):
            # Find integer peak
            y_int, x_int = np.unravel_index(np.argmax(hm), hm.shape)
            # Sub-pixel refinement for precise localization
            y_sub, x_sub = float(y_int), float(x_int)
            conf = float(hm[y_int, x_int])
            
            # Scale to original image with sub-pixel precision (keep as float)
            # Use consistent scaling with training: (size - 1) for pixel-perfect alignment
            x_px = x_sub / max(hm.shape[1] - 1, 1) * (orig_w - 1) if orig_w > 1 else x_sub
            y_px = y_sub / max(hm.shape[0] - 1, 1) * (orig_h - 1) if orig_h > 1 else y_sub
            
            points.append({
                'name': KEYPOINT_NAMES[idx], 
                'x': x_px,  # Now has sub-pixel precision
                'y': y_px,  # Now has sub-pixel precision
                'conf': conf
            })

    # Apply interpolation if enabled
    used_interp = False
    if use_interpolation:
        points, used_interp = interpolate_keypoints(points, threshold=0.0)
    return rgb, resized_rgb, points, heatmaps, used_interp


def predict_pole_top(
    model: KeypointDetector,
    image_path: Path,
    device: torch.device,
    use_tta: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray]:
    """
    Predict pole top keypoint with sub-pixel refinement.
    
    Crops to upper 10% of image (where pole top is always located) before inference.
    Coordinates are transformed back to original image space.
    
    Args:
        model: trained KeypointDetector for pole top
        image_path: path to full pole image
        device: torch device
        use_tta: if True, use test-time augmentation with vertical shifts
    
    Returns:
        Tuple of (rgb_cropped, resized_rgb, point, heatmap)
        - rgb_cropped: original cropped RGB image (upper 10%)
        - resized_rgb: resized cropped region
        - point: keypoint in original image coordinates
        - heatmap: heatmap from model
    """
    # Use config default if not provided
    if use_tta is None:
        use_tta = INFERENCE_USE_TTA
    
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb.shape[:2]
    
    # Crop to upper 10% of image (pole top is always in upper 10%)
    crop_height = int(orig_h * 0.1)
    rgb_cropped = rgb[0:crop_height, :]
    h_cropped, w_cropped = rgb_cropped.shape[:2]

    if use_tta:
        # Test-time augmentation with small vertical shifts
        shifts = [-2, 0, 2]
        all_x, all_y, all_conf = [], [], []
        
        for shift in shifts:
            # Apply vertical shift to cropped region
            M = np.float32([[1, 0, 0], [0, 1, shift]])
            shifted = cv2.warpAffine(rgb_cropped, M, (w_cropped, h_cropped))
            
            resized_rgb = cv2.resize(shifted, (POLE_TOP_RESIZE_WIDTH, POLE_TOP_RESIZE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            tensor = POLE_TOP_PREPROCESS(resized_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                heatmap = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            
            y_int, x_int = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            y_sub, x_sub = float(y_int), float(x_int)
            conf = float(heatmap[y_int, x_int])
            
            # Transform coordinates: heatmap -> resized -> cropped -> original
            # Step 1: Heatmap to resized image coordinates
            x_resized = x_sub / max(heatmap.shape[1] - 1, 1) * (POLE_TOP_RESIZE_WIDTH - 1)
            y_resized = y_sub / max(heatmap.shape[0] - 1, 1) * (POLE_TOP_RESIZE_HEIGHT - 1)
            # Step 2: Resized to cropped image coordinates
            x_cropped = x_resized / POLE_TOP_RESIZE_WIDTH * w_cropped
            y_cropped = (y_resized / POLE_TOP_RESIZE_HEIGHT * h_cropped) - shift  # Undo shift
            # Step 3: Cropped to original image coordinates (y in cropped = y in original since crop starts at y=0)
            x_px = x_cropped
            y_px = y_cropped
            
            all_x.append(x_px)
            all_y.append(y_px)
            all_conf.append(conf)
        
        # Average predictions across shifts
        point = {
            'name': 'pole_top',
            'x': np.mean(all_x),
            'y': np.mean(all_y),
            'conf': np.mean(all_conf)
        }
        
        # Use cropped region (no shift) for visualization
        resized_rgb = cv2.resize(rgb_cropped, (POLE_TOP_RESIZE_WIDTH, POLE_TOP_RESIZE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        tensor = POLE_TOP_PREPROCESS(resized_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            heatmap = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    else:
        # Standard inference with sub-pixel refinement
        resized_rgb = cv2.resize(rgb_cropped, (POLE_TOP_RESIZE_WIDTH, POLE_TOP_RESIZE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        tensor = POLE_TOP_PREPROCESS(resized_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            heatmap = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        # Find integer peak
        y_int, x_int = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        # Sub-pixel refinement for precise localization
        y_sub, x_sub = float(y_int), float(x_int)
        conf = float(heatmap[y_int, x_int])
        
        # Transform coordinates: heatmap -> resized -> cropped -> original
        # Step 1: Heatmap to resized image coordinates
        x_resized = x_sub / max(heatmap.shape[1] - 1, 1) * (POLE_TOP_RESIZE_WIDTH - 1)
        y_resized = y_sub / max(heatmap.shape[0] - 1, 1) * (POLE_TOP_RESIZE_HEIGHT - 1)
        # Step 2: Resized to cropped image coordinates
        x_cropped = x_resized / POLE_TOP_RESIZE_WIDTH * w_cropped
        y_cropped = y_resized / POLE_TOP_RESIZE_HEIGHT * h_cropped
        # Step 3: Cropped to original image coordinates (y in cropped = y in original since crop starts at y=0)
        x_px = x_cropped
        y_px = y_cropped
        
        point = {
            'name': 'pole_top',
            'x': x_px,
            'y': y_px,
            'conf': conf
        }

    return rgb_cropped, resized_rgb, point, heatmap


def feet_to_feet_inches(decimal_feet: float) -> str:
    """
    Convert decimal feet to feet'inches" format.
    
    Args:
        decimal_feet: Height in decimal feet (e.g., 2.5 feet)
        
    Returns:
        String in format "feet'inches\"" (e.g., "2'6\"")
    """
    feet = int(decimal_feet)
    inches = round((decimal_feet - feet) * 12)
    if inches == 12:
        feet += 1
        inches = 0
    return f"{feet}'{inches}\""


def run_end_to_end_inference(
    img_bgr: np.ndarray,
    img_rgb: np.ndarray,
    pole_detector: Optional[Any],  # YOLO model (should be None unless image is from pole_detection dataset)
    ruler_detector: Optional[Any],  # YOLO model (None if not trained yet)
    keypoint_model: Optional[KeypointDetector],
    pole_top_model: Optional[KeypointDetector],
    device: torch.device
) -> Dict[str, Any]:
    """Run end-to-end inference: pole and ruler detected independently -> keypoints -> pole top.
    
    Uses imgsz parameters to match training configuration:
    - Pole detection: imgsz from POLE_DETECTION_CONFIG - matches training with rect=True
    - Ruler detection: imgsz from RULER_DETECTION_CONFIG - matches training with rect=True
    Note: Pole and ruler detection run independently on the full image (not two-stage)
    
    Pole detection should only be enabled for images from pole_detection dataset.
    For other datasets (ruler_detection, midspan, etc.), pole_detector should be None.
    
    Args:
        img_bgr: BGR image array (for YOLO detectors)
        img_rgb: RGB image array (for keypoint models)
        pole_detector: YOLO pole detector (should be None unless image is from pole_detection dataset)
        ruler_detector: YOLO ruler detector
        keypoint_model: KeypointDetector for ruler markings (can be None)
        pole_top_model: KeypointDetector for pole top (can be None)
        device: torch device
    
    Returns:
        Dictionary with 'pole', 'ruler', 'keypoints', 'pole_top' keys
    """
    h_img, w_img = img_rgb.shape[:2]
    
    # Ruler detection required for calibration; return early if missing
    if ruler_detector is None:
        return {'pole': None, 'ruler': None, 'keypoints': None, 'pole_top': None}
    
    # Independent detection: Detect pole on full image (skip if pole_detector is None)
    pole_bbox = None
    pole_crop_rgb = None
    if pole_detector is not None:
        pole_res = pole_detector(img_bgr, conf=INFERENCE_POLE_CONF_THRESHOLD, max_det=INFERENCE_MAX_DETECTIONS, verbose=False, imgsz=POLE_DETECTION_CONFIG['imgsz'])[0]
        if pole_res.boxes and len(pole_res.boxes) > 0:
            px1, py1, px2, py2 = pole_res.boxes.xyxy[0].cpu().numpy().astype(int)
            # Validate bounding box and clamp to image boundaries
            px1 = max(0, min(px1, w_img - 1))
            py1 = max(0, min(py1, h_img - 1))
            px2 = max(px1 + 1, min(px2, w_img))
            py2 = max(py1 + 1, min(py2, h_img))
            # Store validated bbox
            pole_bbox = (px1, py1, px2, py2)
            # Only proceed if crop is valid and non-empty
            if px2 > px1 and py2 > py1:
                pole_crop_bgr = img_bgr[py1:py2, px1:px2]
                if pole_crop_bgr.size > 0:
                    pole_crop_rgb = cv2.cvtColor(pole_crop_bgr, cv2.COLOR_BGR2RGB)
    
    # Independent detection: Detect ruler on full image (not on pole crop)
    ruler_res = ruler_detector(img_bgr, conf=INFERENCE_RULER_CONF_THRESHOLD, max_det=INFERENCE_MAX_DETECTIONS, verbose=False, imgsz=RULER_DETECTION_CONFIG['imgsz'])[0]
    
    if not ruler_res.boxes or len(ruler_res.boxes) == 0:
        return {'pole': pole_bbox, 'ruler': None, 'keypoints': None, 'pole_top': None}
    
    # Ruler bbox is already in full image coordinates
    rx1_full, ry1_full, rx2_full, ry2_full = ruler_res.boxes.xyxy[0].cpu().numpy().astype(int)
    
    # Validate bounding box and clamp to image boundaries
    rx1_full = max(0, min(rx1_full, w_img - 1))
    ry1_full = max(0, min(ry1_full, h_img - 1))
    rx2_full = max(rx1_full + 1, min(rx2_full, w_img))
    ry2_full = max(ry1_full + 1, min(ry2_full, h_img))
    
    # Crop ruler from full image for keypoint detection (only if valid)
    keypoints_pred_global = None
    if rx2_full > rx1_full and ry2_full > ry1_full:
        ruler_crop_rgb = img_rgb[ry1_full:ry2_full, rx1_full:rx2_full]
        if ruler_crop_rgb.size > 0 and keypoint_model is not None:
            keypoints_pred_crop, _ = infer_keypoints_on_crop(keypoint_model, ruler_crop_rgb, device, use_tta=INFERENCE_USE_TTA, use_interpolation=INFERENCE_USE_INTERPOLATION)
            # Convert to global coordinates
            # Keypoints are in ruler crop coordinates, so we add rx1_full and ry1_full
            # to transform: ruler crop -> full image coordinates
            keypoints_pred_global = []
            for kp in keypoints_pred_crop:
                keypoints_pred_global.append({
                    'name': kp['name'],
                    'x': kp['x'] + rx1_full,
                    'y': kp['y'] + ry1_full,
                    'conf': kp['conf']
                })
    
    # Detect pole top (if pole was detected)
    pole_top_global = None
    if pole_top_model is not None and pole_crop_rgb is not None:
        px1, py1, px2, py2 = pole_bbox
        pred_pole_top = infer_pole_top_on_crop(pole_top_model, pole_crop_rgb, device, use_tta=INFERENCE_USE_TTA)
        if pred_pole_top:  # Always accept (fixed keypoint count: exactly 1 per pole)
            # Transform pole top from pole crop coordinates to global coordinates
            pole_top_x_global = pred_pole_top['x'] + px1
            pole_top_y_global = pred_pole_top['y'] + py1
            
            # Validate that pole top is within pole bounding box and image bounds
            # Note: Pole top is detected in upper 10% of pole crop, so it should be within bounds
            # But we add validation to catch edge cases where pole detection might be off
            pole_top_x_global = max(px1, min(px2, pole_top_x_global))  # Clamp to pole bbox x bounds
            pole_top_y_global = max(py1, min(py2, pole_top_y_global))  # Clamp to pole bbox y bounds
            pole_top_x_global = max(0, min(w_img - 1, pole_top_x_global))  # Clamp to image bounds
            pole_top_y_global = max(0, min(h_img - 1, pole_top_y_global))  # Clamp to image bounds
            
            pole_top_global = {
                'x': pole_top_x_global,
                'y': pole_top_y_global,
                'conf': pred_pole_top['conf']
            }
    
    return {
        'pole': pole_bbox,
        'ruler': (rx1_full, ry1_full, rx2_full, ry2_full),  # Full image coordinates
        'keypoints': keypoints_pred_global,
        'pole_top': pole_top_global
    }


def get_weighted_confidence_from_dict(keypoints_dict: Dict[str, float], use_ruler_marking: bool = True) -> float:
    """
    Calculate weighted confidence from individual keypoint confidences.

    These weights are derived from correlation analysis between confidence scores
    and prediction errors on the test dataset. Keypoints with stronger negative
    correlation (higher abs(r)) get higher weights.

    Date: 2026-02-07, Improvement: +17.93% correlation vs average confidence

    Args:
        keypoints_dict: Dict mapping keypoint names to confidence scores
        use_ruler_marking: If True, use ruler marking weights; else pole top

    Returns:
        float: Weighted confidence score (0.0-1.0)
    """
    weights = RULER_MARKING_WEIGHTS if use_ruler_marking else {'pole_top': POLE_TOP_WEIGHT_ALONE}

    if not keypoints_dict:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0

    for kp_name, conf in keypoints_dict.items():
        if kp_name in weights:
            weight = weights[kp_name]
            weighted_sum += conf * weight
            total_weight += weight

    if total_weight > 0:
        return weighted_sum / total_weight

    # Fallback
    return sum(keypoints_dict.values()) / len(keypoints_dict) if keypoints_dict else 0.0


def get_pole_photo_combined_confidence(pole_top_conf: float, ruler_marking_weighted_conf: float) -> float:
    """
    Calculate combined confidence for POLE PHOTOS (with both ruler and pole top).

    For pole photos only! Combines pole_top and ruler_marking confidence with
    optimal weights based on correlation analysis.

    Args:
        pole_top_conf: Confidence score from pole_top detection
        ruler_marking_weighted_conf: Weighted confidence from ruler markings

    Returns:
        float: Combined confidence score (0.0-1.0)
    """
    pole_weight = POLE_PHOTO_CONFIDENCE_WEIGHTS['pole_top']
    ruler_weight = POLE_PHOTO_CONFIDENCE_WEIGHTS['ruler_marking']

    return pole_top_conf * pole_weight + ruler_marking_weighted_conf * ruler_weight
