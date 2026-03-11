"""
Dataset classes for keypoint detection training.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .config import (
    HEATMAP_HEIGHT, HEATMAP_WIDTH, RESIZE_HEIGHT, RESIZE_WIDTH,
    GAUSSIAN_SIGMA_X, GAUSSIAN_SIGMA_Y, NUM_KEYPOINTS,
    POLE_TOP_HEATMAP_HEIGHT, POLE_TOP_HEATMAP_WIDTH,
    POLE_TOP_RESIZE_HEIGHT, POLE_TOP_RESIZE_WIDTH,
    RULER_DETECTION_CONFIG,
    INFERENCE_RULER_CONF_THRESHOLD,
    INFERENCE_MAX_DETECTIONS,
    INFERENCE_RULER_WEIGHTS
)


def _parse_ppi_from_label(label_path: Path) -> float:
    """Parse PPI value from label file comment line."""
    if not label_path.exists():
        return 0.0
    try:
        for line in label_path.read_text().strip().split('\n'):
            if line.strip().startswith('# PPI='):
                return float(line.split('=')[1])
    except (ValueError, IndexError):
        pass
    return 0.0


def _apply_transform_to_keypoint(kp_x: float, kp_y: float, M: np.ndarray, img_w: int, img_h: int) -> Tuple[float, float]:
    """Apply affine transformation matrix to a single keypoint."""
    kp_homogeneous = np.array([kp_x, kp_y, 1.0])
    kp_transformed = M @ kp_homogeneous
    return np.clip(kp_transformed[0], 0, img_w - 1), np.clip(kp_transformed[1], 0, img_h - 1)


def _build_augmentation_matrix(img_w: int, img_h: int, aug_params: dict) -> np.ndarray:
    """Build affine transformation matrix from augmentation parameters."""
    import cv2

    translate_x = aug_params.get('translate_x', 0.0)
    translate_y = aug_params.get('translate_y', 0.0)
    scale_min = aug_params.get('scale_min', 1.0)
    scale_max = aug_params.get('scale_max', 1.0)
    rotate_degrees = aug_params.get('rotate', 0.0)

    tx = np.random.uniform(-translate_x, translate_x) * img_w if translate_x > 0 else 0
    ty = np.random.uniform(-translate_y, translate_y) * img_h if translate_y > 0 else 0
    scale = np.random.uniform(scale_min, scale_max) if scale_min < scale_max else 1.0
    angle = np.random.uniform(-rotate_degrees, rotate_degrees) if rotate_degrees > 0 else 0.0

    center_x, center_y = img_w / 2, img_h / 2
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return M

class KeypointDataset(Dataset):
    """Dataset for keypoint heatmap regression training and validation."""
    def __init__(self, image_dir, label_dir, transform=None, min_visible_keypoints=5,
                 use_yolo_crop=False, ruler_detection_model_path=None, original_image_dir=None, original_label_dir=None,
                 geometric_augmentations=None):
        """
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing YOLO format labels
            transform: Optional image transforms
            min_visible_keypoints: Minimum number of visible keypoints required (default: 5, all keypoints)
                                  Set to lower value to include samples with missing keypoints
            use_yolo_crop: If True, use YOLO ruler detection with 75% probability to create crops
            ruler_detection_model_path: Path to trained ruler detection YOLO model
            original_image_dir: Directory containing original pole crop images (for YOLO cropping)
            original_label_dir: Directory containing ruler detection labels (for GT bbox to transform keypoints)
            geometric_augmentations: Dict with 'translate', 'scale', 'rotate' to simulate YOLO detection variations
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.min_visible_keypoints = min_visible_keypoints
        self.use_yolo_crop = use_yolo_crop
        self.geometric_augmentations = geometric_augmentations or {}
        
        # Get all image files
        all_image_files = sorted(self.image_dir.glob('*.jpg'))
        if not all_image_files:
            raise RuntimeError(f"No images found in {self.image_dir}")
        
        # Filter images based on number of visible keypoints
        self.image_files = []
        filtered_count = 0
        for img_path in all_image_files:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                filtered_count += 1
                continue
            
            # Count visible keypoints
            num_visible = self._count_visible_keypoints(label_path)
            if num_visible >= self.min_visible_keypoints:
                self.image_files.append(img_path)
            else:
                filtered_count += 1
        
        if not self.image_files:
            raise RuntimeError(
                f"No images with at least {self.min_visible_keypoints} visible keypoints found in {self.image_dir}. "
                f"Filtered out {filtered_count} images."
            )
        
        # Setup for YOLO-based cropping
        self.ruler_detector = None
        if use_yolo_crop:
            if ruler_detection_model_path is None:
                ruler_detection_model_path = INFERENCE_RULER_WEIGHTS
            if not Path(ruler_detection_model_path).exists():
                raise FileNotFoundError(f"Ruler detection model not found at {ruler_detection_model_path}")
            # Load YOLO model lazily (will be loaded on first use)
            self.ruler_detection_model_path = ruler_detection_model_path
            if original_image_dir is None:
                # Infer from image_dir: ruler_marking_detection/images/train -> ruler_detection/images/train
                original_image_dir = str(self.image_dir).replace('ruler_marking_detection', 'ruler_detection')
            if original_label_dir is None:
                # Infer from label_dir: ruler_marking_detection/labels/train -> ruler_detection/labels/train
                original_label_dir = str(self.label_dir).replace('ruler_marking_detection', 'ruler_detection')
            self.original_image_dir = Path(original_image_dir)
            self.original_label_dir = Path(original_label_dir)
        
        aug_str = " (with YOLO crop augmentation)" if use_yolo_crop else ""
        print(f"KeypointDataset: Loaded {len(self.image_files)} images "
              f"(filtered {filtered_count} with <{self.min_visible_keypoints} visible keypoints){aug_str}")
        
        x_range = np.arange(HEATMAP_WIDTH)
        y_range = np.arange(HEATMAP_HEIGHT)
        self.x_grid, self.y_grid = np.meshgrid(x_range, y_range)
        self.x_grid = self.x_grid.astype(np.float32)
        self.y_grid = self.y_grid.astype(np.float32)
        self.sigma_x = GAUSSIAN_SIGMA_X
        self.sigma_y = GAUSSIAN_SIGMA_Y
    
    def _count_visible_keypoints(self, label_path):
        """Count the number of visible keypoints in a label file."""
        if not label_path.exists():
            return 0
        
        try:
            parts = label_path.read_text().strip().split()
            if len(parts) < 5 + NUM_KEYPOINTS * 3:
                return 0
            
            count = 0
            for i in range(NUM_KEYPOINTS):
                kp_idx = 5 + i * 3
                if kp_idx + 2 < len(parts):
                    v = float(parts[kp_idx + 2])
                    if v > 0:  # Visible (v=2) or occluded (v=1)
                        count += 1
            return count
        except Exception:
            return 0

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Determine if we should use YOLO crop (75% probability when enabled)
        use_yolo = self.use_yolo_crop and np.random.rand() < 0.75
        
        if use_yolo:
            # Load original pole crop image for YOLO detection
            original_img_path = self.original_image_dir / img_path.name
            if not original_img_path.exists():
                # Fallback to ruler crop image if original not found
                use_yolo = False
                img = cv2.imread(str(img_path))
            else:
                img = cv2.imread(str(original_img_path))
        else:
            # Use ruler crop image directly
            img = cv2.imread(str(img_path))
        
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        label_path = self.label_dir / f"{img_path.stem}.txt"
        
        # Use YOLO ruler detection to create crop (75% probability)
        if use_yolo:
            # Load YOLO model lazily
            if self.ruler_detector is None:
                from ultralytics import YOLO
                self.ruler_detector = YOLO(str(self.ruler_detection_model_path))
            
            # Get GT ruler bbox from ruler_detection labels to transform keypoint coordinates
            # The keypoints in ruler_marking_detection label are relative to the GT crop
            # We need to transform them to pole crop coordinates first
            pole_crop_h, pole_crop_w = h, w  # Store pole crop dimensions
            gt_ruler_label_path = self.original_label_dir / f"{img_path.stem}.txt"
            gt_crop_x1, gt_crop_y1, gt_crop_w, gt_crop_h = 0, 0, w, h
            if gt_ruler_label_path.exists():
                with open(gt_ruler_label_path, 'r') as f:
                    gt_label_line = None
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            gt_label_line = line
                            break
                    if gt_label_line:
                        parts = gt_label_line.split()
                        if len(parts) >= 5:  # class + bbox(4)
                            # YOLO format: class x_center y_center width height (normalized)
                            x_center_norm = float(parts[1])
                            y_center_norm = float(parts[2])
                            width_norm = float(parts[3])
                            height_norm = float(parts[4])
                            # Convert to absolute bbox coordinates using pole crop dimensions
                            gt_crop_w = int(width_norm * pole_crop_w)
                            gt_crop_h = int(height_norm * pole_crop_h)
                            gt_crop_x1 = int((x_center_norm - width_norm / 2) * pole_crop_w)
                            gt_crop_y1 = int((y_center_norm - height_norm / 2) * pole_crop_h)
            
            # Parse keypoints and PPI from ruler_marking_detection label
            # Keypoints are normalized relative to GT ruler crop, not pole crop
            keypoints_crop = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
            visibilities = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
            ppi = 0.0
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    # Parse PPI from comment line if present
                    for line in lines:
                        line = line.strip()
                        if line.startswith('# PPI='):
                            try:
                                ppi = float(line.split('=')[1])
                            except:
                                ppi = 0.0
                            break
                    # Find the actual label line (non-comment)
                    label_line = None
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            label_line = line
                            break
                    if label_line:
                        parts = label_line.split()
                        if len(parts) >= 5 + NUM_KEYPOINTS * 3:
                            # Parse keypoints relative to GT ruler crop (normalized coordinates)
                            for i in range(NUM_KEYPOINTS):
                                kp_idx = 5 + i * 3
                                if kp_idx + 2 < len(parts):
                                    kp_x_norm = float(parts[kp_idx])
                                    kp_y_norm = float(parts[kp_idx + 1])
                                    vis = float(parts[kp_idx + 2])
                                    # Convert normalized keypoint to absolute coordinates in GT crop space
                                    keypoints_crop[i, 0] = kp_x_norm * gt_crop_w
                                    keypoints_crop[i, 1] = kp_y_norm * gt_crop_h
                                    visibilities[i] = vis
            
            # Transform keypoints from crop space to pole crop space
            keypoints_pole_crop = keypoints_crop.copy()
            keypoints_pole_crop[:, 0] += gt_crop_x1
            keypoints_pole_crop[:, 1] += gt_crop_y1
            
            # Run YOLO ruler detection on pole crop image
            results = self.ruler_detector(img, conf=INFERENCE_RULER_CONF_THRESHOLD, max_det=INFERENCE_MAX_DETECTIONS, verbose=False, imgsz=RULER_DETECTION_CONFIG['imgsz'])
            if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                # Get YOLO prediction bbox
                yolo_box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
                yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_box
                # Clip to image bounds using pole crop dimensions
                yolo_x1 = max(0, min(yolo_x1, pole_crop_w - 1))
                yolo_y1 = max(0, min(yolo_y1, pole_crop_h - 1))
                yolo_x2 = max(yolo_x1 + 1, min(yolo_x2, pole_crop_w))
                yolo_y2 = max(yolo_y1 + 1, min(yolo_y2, pole_crop_h))
                
                # Crop image using YOLO prediction
                img = img[yolo_y1:yolo_y2, yolo_x1:yolo_x2]
                
                # Transform keypoints from pole crop space to YOLO prediction crop space
                # This ensures keypoint coordinates are relative to the YOLO-predicted ruler crop
                keypoints = keypoints_pole_crop.copy()
                keypoints[:, 0] -= yolo_x1
                keypoints[:, 1] -= yolo_y1
                
                # Update dimensions
                h, w = img.shape[:2]
                
                # Clamp keypoints to valid range
                keypoints[:, 0] = np.clip(keypoints[:, 0], 0, w - 1)
                keypoints[:, 1] = np.clip(keypoints[:, 1], 0, h - 1)
                
                # Check if keypoints are still visible after YOLO crop
                for i in range(NUM_KEYPOINTS):
                    if visibilities[i] > 0:
                        if keypoints[i, 0] < 0 or keypoints[i, 0] >= w or keypoints[i, 1] < 0 or keypoints[i, 1] >= h:
                            visibilities[i] = 0.0
            else:
                # YOLO didn't detect ruler, fall back to using original crop
                # Reload the ruler crop image
                img = cv2.imread(str(img_path))
                if img is None:
                    raise FileNotFoundError(f"Unable to read: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                # Keypoints are already in crop space, re-parse from label
                keypoints, visibilities, ppi = self._parse_yolo_labels(label_path, w, h)
        else:
            # Not using YOLO crop - parse keypoints normally (they're relative to ruler crop image)
            keypoints, visibilities, ppi = self._parse_yolo_labels(label_path, w, h)
        
        # Apply geometric augmentations to simulate YOLO detection variations
        if self.geometric_augmentations and self.transform:  # Only apply during training
            img, keypoints = self._apply_geometric_augmentations(img, keypoints, w, h)
            h, w = img.shape[:2]
        
        heatmaps = self._create_heatmaps(keypoints, visibilities, w, h)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # FIXED: Use consistent scaling with heatmap creation (with -1 for pixel-perfect alignment)
        scale_x = (RESIZE_WIDTH - 1) / max(w - 1, 1)
        scale_y = (RESIZE_HEIGHT - 1) / max(h - 1, 1)
        keypoints_resized = keypoints.copy()
        keypoints_resized[:, 0] = keypoints[:, 0] * scale_x
        keypoints_resized[:, 1] = keypoints[:, 1] * scale_y

        # Return original image dimensions and PPI for inch-based PCK calculation
        return (
            img,
            torch.from_numpy(heatmaps),
            torch.from_numpy(keypoints_resized.astype(np.float32)),
            torch.from_numpy(visibilities.astype(np.float32)),
            torch.tensor([h, w], dtype=torch.float32),  # original height, width
            torch.tensor([ppi], dtype=torch.float32),  # pixels per inch
        )

    def _parse_yolo_labels(self, label_path, img_w, img_h):
        keypoints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        visibilities = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        ppi = 0.0  # Default PPI if not found

        if label_path.exists():
            # Parse PPI from comment line if present
            lines = label_path.read_text().strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('# PPI='):
                    try:
                        ppi = float(line.split('=')[1])
                    except:
                        ppi = 0.0
                    break
            # Find the actual label line (non-comment)
            label_line = None
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    label_line = line
                    break
            if not label_line:
                return keypoints, visibilities, ppi
            parts = label_line.split()
            if len(parts) >= 5 + NUM_KEYPOINTS * 3:
                for i in range(NUM_KEYPOINTS):
                    kp_idx = 5 + i * 3
                    if kp_idx + 2 < len(parts):
                        x = float(parts[kp_idx]) * img_w
                        y = float(parts[kp_idx + 1]) * img_h
                        v = float(parts[kp_idx + 2])
                        keypoints[i] = (x, y)
                        visibilities[i] = v
        return keypoints, visibilities, ppi

    def _create_heatmaps(self, keypoints, visibilities, img_w, img_h):
        heatmaps = np.zeros((NUM_KEYPOINTS, HEATMAP_HEIGHT, HEATMAP_WIDTH), dtype=np.float32)
        scale_x = (HEATMAP_WIDTH - 1) / max(img_w - 1, 1)
        scale_y = (HEATMAP_HEIGHT - 1) / max(img_h - 1, 1)

        for i, (kp, vis) in enumerate(zip(keypoints, visibilities)):
            if vis > 0:
                cx = kp[0] * scale_x
                cy = kp[1] * scale_y
                heatmaps[i] = np.exp(-(((self.x_grid - cx) ** 2) / (2 * self.sigma_x ** 2) + ((self.y_grid - cy) ** 2) / (2 * self.sigma_y ** 2)))
        return heatmaps
    
    def _apply_geometric_augmentations(self, img, keypoints, img_w, img_h):
        """Apply geometric augmentations to simulate YOLO detection variations."""
        import cv2

        M = _build_augmentation_matrix(img_w, img_h, self.geometric_augmentations)
        img_transformed = cv2.warpAffine(img, M, (img_w, img_h),
                                        borderMode=cv2.BORDER_REFLECT_101,
                                        flags=cv2.INTER_LINEAR)

        if len(keypoints) > 0:
            kp_homogeneous = np.ones((len(keypoints), 3))
            kp_homogeneous[:, :2] = keypoints
            kp_transformed = (M @ kp_homogeneous.T).T
            kp_transformed[:, 0] = np.clip(kp_transformed[:, 0], 0, img_w - 1)
            kp_transformed[:, 1] = np.clip(kp_transformed[:, 1], 0, img_h - 1)
            keypoints = kp_transformed[:, :2]

        return img_transformed, keypoints


class PoleTopKeypointDataset(Dataset):
    """Dataset for pole top keypoint detection using pole_top_detection_yolo dataset.

    Automatically crops images to upper 10% since pole top is always in this region.
    Keypoint coordinates are transformed to cropped image space during training.
    """
    def __init__(self, image_dir, label_dir, transform=None, geometric_augmentations=None):
        """
        Args:
            image_dir: Directory containing pole crop images
            label_dir: Directory containing YOLO format labels
            transform: Image transforms to apply
            geometric_augmentations: Dict with 'translate', 'scale', 'rotate' to simulate YOLO detection variations
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.geometric_augmentations = geometric_augmentations or {}
        self.image_files = sorted(self.image_dir.glob('*.jpg'))
        if not self.image_files:
            raise RuntimeError(f"No images found in {self.image_dir}")
        
        x_range = np.arange(POLE_TOP_HEATMAP_WIDTH)
        y_range = np.arange(POLE_TOP_HEATMAP_HEIGHT)
        self.x_grid, self.y_grid = np.meshgrid(x_range, y_range)
        # Use pole top-specific Gaussian sigmas (sharper for single keypoint)
        self.sigma_x = POLE_TOP_HEATMAP_WIDTH / 8
        self.sigma_y = POLE_TOP_HEATMAP_HEIGHT / 32
        
        print(f"PoleTopKeypointDataset: {len(self.image_files)} images (upper 10% crop)")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Unable to read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Crop to upper 10% of image (pole top is always in upper 10%)
        crop_height = int(h * 0.1)
        img_cropped = img[0:crop_height, :]
        h_cropped, w_cropped = img_cropped.shape[:2]
        
        label_path = self.label_dir / f"{img_path.stem}.txt"
        kp_x, kp_y, vis = w // 2, 0, 0.0
        ppi = _parse_ppi_from_label(label_path)
        if label_path.exists():
            for line in label_path.read_text().strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 8:
                        vis = float(parts[7])
                        kp_x = float(parts[5]) * w
                        kp_y = float(parts[6]) * h
                    break
        
        # Apply geometric augmentations to simulate YOLO detection variations
        if self.geometric_augmentations and self.transform:  # Only apply during training
            img_cropped, kp_x, kp_y = self._apply_geometric_augmentations_pole_top(
                img_cropped, kp_x, kp_y, w_cropped, h_cropped
            )
            h_cropped, w_cropped = img_cropped.shape[:2]
        
        heatmap = np.zeros((POLE_TOP_HEATMAP_HEIGHT, POLE_TOP_HEATMAP_WIDTH), dtype=np.float32)
        if vis > 0:
            scale_x = (POLE_TOP_HEATMAP_WIDTH - 1) / max(w_cropped - 1, 1)
            scale_y = (POLE_TOP_HEATMAP_HEIGHT - 1) / max(h_cropped - 1, 1)
            cx = kp_x * scale_x
            cy = kp_y * scale_y
            heatmap = np.exp(-(((self.x_grid - cx) ** 2) / (2 * self.sigma_x ** 2) + 
                              ((self.y_grid - cy) ** 2) / (2 * self.sigma_y ** 2)))
        
        if self.transform:
            img_cropped = self.transform(img_cropped)
        else:
            img_cropped = transforms.ToTensor()(img_cropped)
        
        scale_x = (POLE_TOP_RESIZE_WIDTH - 1) / max(w_cropped - 1, 1)
        scale_y = (POLE_TOP_RESIZE_HEIGHT - 1) / max(h_cropped - 1, 1)
        # Return keypoints with shape (1, 2) to match KeypointDataset format (batch, num_keypoints, 2)
        kp_resized = np.array([[kp_x * scale_x, kp_y * scale_y]], dtype=np.float32)
        
        return (img_cropped, torch.from_numpy(heatmap[None]), 
                torch.from_numpy(kp_resized), torch.tensor([[vis]], dtype=torch.float32),
                torch.tensor([h_cropped, w_cropped, h], dtype=torch.float32),
                torch.tensor([ppi], dtype=torch.float32))
    
    def _apply_geometric_augmentations_pole_top(self, img, kp_x, kp_y, img_w, img_h):
        """Apply geometric augmentations to simulate YOLO detection variations for pole top."""
        import cv2

        M = _build_augmentation_matrix(img_w, img_h, self.geometric_augmentations)
        img_transformed = cv2.warpAffine(img, M, (img_w, img_h),
                                        borderMode=cv2.BORDER_REFLECT_101,
                                        flags=cv2.INTER_LINEAR)
        kp_x_new, kp_y_new = _apply_transform_to_keypoint(kp_x, kp_y, M, img_w, img_h)
        return img_transformed, kp_x_new, kp_y_new


class EquipmentKeypointDataset(Dataset):
    """Dataset for equipment keypoint detection (riser, transformer, street_light).

    Images are pre-cropped to the equipment bounding box.
    Labels use YOLO pose format: class cx cy w h kp0_x kp0_y kp0_v [kp1_x kp1_y kp1_v]
    """
    def __init__(self, image_dir, label_dir, num_keypoints, resize_height, resize_width,
                 heatmap_height, heatmap_width, transform=None, geometric_augmentations=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.num_keypoints = num_keypoints
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.heatmap_height = heatmap_height
        self.heatmap_width = heatmap_width
        self.transform = transform
        self.geometric_augmentations = geometric_augmentations or {}
        self.image_files = sorted(self.image_dir.glob('*.jpg'))
        if not self.image_files:
            raise RuntimeError(f"No images found in {self.image_dir}")

        x_range = np.arange(heatmap_width)
        y_range = np.arange(heatmap_height)
        self.x_grid, self.y_grid = np.meshgrid(x_range, y_range)
        self.sigma_x = heatmap_width / 8
        self.sigma_y = heatmap_height / 32

        print(f"EquipmentKeypointDataset: {len(self.image_files)} images, "
              f"{num_keypoints} keypoint(s), resize={resize_height}x{resize_width}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Unable to read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Parse label
        label_path = self.label_dir / f"{img_path.stem}.txt"
        kp_coords = []  # list of (x_px, y_px)
        kp_vis = []     # list of float
        ppi = _parse_ppi_from_label(label_path)

        if label_path.exists():
            for line in label_path.read_text().strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    for ki in range(self.num_keypoints):
                        base = 5 + ki * 3
                        if base + 2 < len(parts):
                            kp_coords.append((float(parts[base]) * w, float(parts[base + 1]) * h))
                            kp_vis.append(float(parts[base + 2]))
                        else:
                            kp_coords.append((w / 2, h / 2))
                            kp_vis.append(0.0)
                    break

        # Fill missing keypoints
        while len(kp_coords) < self.num_keypoints:
            kp_coords.append((w / 2, h / 2))
            kp_vis.append(0.0)

        # Apply geometric augmentations
        if self.geometric_augmentations and self.transform:
            img, kp_coords = self._apply_geometric_augmentations(img, kp_coords, w, h)
            h, w = img.shape[:2]

        # Generate heatmaps
        heatmaps = np.zeros((self.num_keypoints, self.heatmap_height, self.heatmap_width), dtype=np.float32)
        for ki in range(self.num_keypoints):
            if kp_vis[ki] > 0:
                scale_x = (self.heatmap_width - 1) / max(w - 1, 1)
                scale_y = (self.heatmap_height - 1) / max(h - 1, 1)
                cx = kp_coords[ki][0] * scale_x
                cy = kp_coords[ki][1] * scale_y
                heatmaps[ki] = np.exp(-(((self.x_grid - cx) ** 2) / (2 * self.sigma_x ** 2) +
                                        ((self.y_grid - cy) ** 2) / (2 * self.sigma_y ** 2)))

        # Apply image transforms
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # Keypoints in resized image space
        scale_x = (self.resize_width - 1) / max(w - 1, 1)
        scale_y = (self.resize_height - 1) / max(h - 1, 1)
        kp_resized = np.array([[kp[0] * scale_x, kp[1] * scale_y] for kp in kp_coords], dtype=np.float32)
        vis_array = np.array(kp_vis, dtype=np.float32).reshape(self.num_keypoints, 1)

        return (img, torch.from_numpy(heatmaps),
                torch.from_numpy(kp_resized), torch.from_numpy(vis_array),
                torch.tensor([h, w, h], dtype=torch.float32),
                torch.tensor([ppi], dtype=torch.float32))

    def _apply_geometric_augmentations(self, img, kp_coords, img_w, img_h):
        """Apply geometric augmentations to keypoint coordinates."""
        M = _build_augmentation_matrix(img_w, img_h, self.geometric_augmentations)
        img_transformed = cv2.warpAffine(img, M, (img_w, img_h),
                                         borderMode=cv2.BORDER_REFLECT_101,
                                         flags=cv2.INTER_LINEAR)
        new_coords = [_apply_transform_to_keypoint(kp_x, kp_y, M, img_w, img_h)
                      for kp_x, kp_y in kp_coords]
        return img_transformed, new_coords