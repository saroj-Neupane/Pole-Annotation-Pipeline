"""
Training utility functions for model training.
"""

import sys
import yaml
import numpy as np
import torch
import torch.optim as optim
import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    YOLO = None  # type: ignore

from .config import (
    IMAGENET_MEAN, IMAGENET_STD,
    POLE_DETECTION_CONFIG, RULER_DETECTION_CONFIG, EQUIPMENT_DETECTION_CONFIG,
    ATTACHMENT_DETECTION_CONFIG, ATTACHMENT_AUGMENT_PARAMS,
    YOLO_MODEL_PATHS, YOLO_MODELS_DIR, PROJECT_ROOT, RUNS_DIR,
    RULER_AUGMENT_PARAMS, POLE_AUGMENT_PARAMS, EQUIPMENT_AUGMENT_PARAMS,
    POLE_TOP_AUGMENT_PARAMS, AUGMENT_PARAMS,
    DATASETS_DIR, DATASET_DIRS, POLE_DETECTION, RULER_DETECTION,
    EQUIPMENT_DETECTION, EQUIPMENT_CLASS_NAMES,
    ATTACHMENT_CLASS_NAMES, EQUIPMENT_KEYPOINT_CONFIGS, ATTACHMENT_KEYPOINT_CONFIGS,
    RULER_MARKING_DETECTION_CONFIG, POLE_TOP_DETECTION_CONFIG,
)


def clear_yolo_disk_cache_from_other_datasets(current_model: str, dry_run: bool = False) -> int:
    """
    Remove YOLO *.npy disk cache from other datasets to free disk space before training.

    Ultralytics caches decoded images as *.npy alongside images. With limited disk,
    clearing cache from already-trained models frees space for the current model's cache.

    Returns:
        Total bytes freed (0 if dry_run).
    """
    current_dir = DATASET_DIRS.get(current_model)
    if not current_dir:
        return 0
    freed = 0
    for name, dataset_dir in DATASET_DIRS.items():
        if name == current_model or not isinstance(dataset_dir, Path):
            continue
        images_dir = dataset_dir / "images"
        if not images_dir.exists():
            continue
        for split in ("train", "val", "test"):
            split_dir = images_dir / split
            if not split_dir.exists():
                continue
            for npy_path in split_dir.glob("*.npy"):
                if dry_run:
                    freed += npy_path.stat().st_size
                else:
                    try:
                        freed += npy_path.stat().st_size
                        npy_path.unlink()
                    except OSError:
                        pass
    if freed and not dry_run:
        print(f"Cleared {freed / (1 << 30):.1f} GB YOLO cache from other datasets (kept {current_model})")
    return freed
from .losses import FocalHeatmapLoss, UnimodalHeatmapLoss


def _parse_height(name):
    """Parse height from keypoint name."""
    try:
        return float(name)
    except (TypeError, ValueError):
        return np.nan


def _interpolate_sequence(values, indices, confident_indices):
    """Interpolate missing values in a sequence using confident points."""
    result = values.astype(np.float32).copy()
    confident_set = set(confident_indices.tolist())
    for idx in indices:
        if idx in confident_set:
            continue
        lower = confident_indices[confident_indices < idx]
        upper = confident_indices[confident_indices > idx]
        if len(lower) and len(upper):
            l = lower[-1]
            u = upper[0]
            t = (idx - l) / (u - l)
            result[idx] = result[l] * (1 - t) + result[u] * t
        elif len(lower) >= 2:
            l1 = lower[-1]
            l2 = lower[-2]
            slope = (result[l1] - result[l2]) / (l1 - l2)
            result[idx] = result[l1] + slope * (idx - l1)
        elif len(upper) >= 2:
            u1 = upper[0]
            u2 = upper[1]
            slope = (result[u2] - result[u1]) / (u2 - u1)
            result[idx] = result[u1] + slope * (idx - u1)
        elif len(lower) == 1:
            result[idx] = result[lower[-1]]
        elif len(upper) == 1:
            result[idx] = result[upper[0]]
    return result


def interpolate_keypoints(points, threshold=0.0):
    """
    Interpolate missing keypoints using linear regression on confident points.
    
    Args:
        points: List of keypoint dictionaries with 'name', 'x', 'y', 'conf' keys
        threshold: Confidence threshold for considering points as confident
        
    Returns:
        Tuple of (interpolated_points, was_interpolated)
    """
    if not points:
        return points, False

    for p in points:
        p['interpolated'] = False

    indices = np.arange(len(points))
    confidences = np.array([p['conf'] for p in points], dtype=np.float32)
    confident_mask = confidences >= threshold
    if confident_mask.sum() < 2:
        return points, False

    heights = np.array([_parse_height(p['name']) for p in points], dtype=np.float32)
    valid_height_mask = ~np.isnan(heights)
    confident_height_mask = confident_mask & valid_height_mask

    xs = np.array([p['x'] for p in points], dtype=np.float32)
    ys = np.array([p['y'] for p in points], dtype=np.float32)

    interpolated = False

    # Calibrate using known heights when we have enough confident points
    calibration_mask = confident_height_mask.copy()
    if calibration_mask.sum() < 2 and valid_height_mask.sum() >= 2:
        # Use the two most confident keypoints with valid heights as backup anchors
        valid_indices = np.where(valid_height_mask)[0]
        top_indices = valid_indices[np.argsort(confidences[valid_indices])[::-1][:2]]
        calibration_mask[top_indices] = True

    if calibration_mask.sum() >= 2:
        h_conf = heights[calibration_mask]
        x_conf = xs[calibration_mask]
        y_conf = ys[calibration_mask]

        # Fit linear model height -> pixel position
        x_coeffs = np.polyfit(h_conf, x_conf, 1)
        y_coeffs = np.polyfit(h_conf, y_conf, 1)

        for i, p in enumerate(points):
            if not valid_height_mask[i]:
                continue
            if confidences[i] < threshold:
                x_pred = np.polyval(x_coeffs, heights[i])
                y_pred = np.polyval(y_coeffs, heights[i])
                original_conf = p.get('original_conf', p['conf'])
                p['x'] = float(x_pred)
                p['y'] = float(y_pred)
                p['original_conf'] = original_conf
                p['conf'] = float(threshold)
                p['interpolated'] = True
                interpolated = True
        return points, interpolated

    # Fall back to index-based interpolation if heights are insufficient
    confident_indices = indices[confident_mask]
    xs_interp = _interpolate_sequence(xs, indices, confident_indices)
    ys_interp = _interpolate_sequence(ys, indices, confident_indices)

    for i, p in enumerate(points):
        if confidences[i] < threshold:
            original_conf = p.get('original_conf', p['conf'])
            p['x'] = float(xs_interp[i])
            p['y'] = float(ys_interp[i])
            p['original_conf'] = original_conf
            p['conf'] = float(threshold)
            p['interpolated'] = True
            interpolated = True
    return points, interpolated


def _get_completed_epochs(stage: str) -> Optional[int]:
    """
    Get actual completed epoch count from results.csv or last.pt.
    args.yaml stores target epochs, not completed - we need results.csv.
    """
    results_csv = RUNS_DIR / stage / 'results.csv'
    if results_csv.exists():
        try:
            with results_csv.open('r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last_epoch = int(float(rows[-1]['epoch']))
                    return last_epoch
        except (ValueError, KeyError):
            pass
    # Fallback: last.pt checkpoint stores epoch (0-indexed last completed)
    last_pt = RUNS_DIR / stage / 'weights' / 'last.pt'
    if last_pt.exists():
        try:
            ckpt = torch.load(str(last_pt), map_location='cpu', weights_only=False)
            return ckpt.get('epoch', 0) + 1  # 0-indexed -> count
        except Exception:
            pass
    return None


def find_resume_weights(stage: str, target_epochs: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Decide whether to resume training or warm-start based on existing weights and epoch counts.

    Args:
        stage: Training stage name (e.g., 'pole_detection')
        target_epochs: Target number of epochs for training

    Returns:
        Tuple of (weights_path, mode) where mode is 'resume', 'warm_start', 'already_complete', or None

    Note:
        - Returns 'resume' if training is incomplete (completed_epochs < target) or we want more epochs
        - Returns 'warm_start' if training is complete and target_epochs > completed
        - Returns 'already_complete' if training is complete and target_epochs <= completed (skip)
    """
    weights_dir = RUNS_DIR / stage / 'weights'
    last_path = weights_dir / 'last.pt'
    best_path = weights_dir / 'best.pt'

    completed_epochs = _get_completed_epochs(stage)  # actual completed, not target

    if last_path.exists():
        if completed_epochs is not None and completed_epochs >= target_epochs:
            return None, 'already_complete'
        return str(last_path), 'resume'

    if best_path.exists():
        if completed_epochs is not None and target_epochs > completed_epochs:
            return str(best_path), 'warm_start'
        if completed_epochs is not None and completed_epochs >= target_epochs:
            return None, 'already_complete'
    return None, None


def _yolo_training_args(config: dict, extra_overrides: Optional[Dict] = None) -> dict:
    """Build YOLO training args from config dict. extra_overrides applied last."""
    args = {
        'optimizer': 'auto',
        'lr0': config['lr0'],
        'lrf': config['lrf'],
        'momentum': config['momentum'],
        'weight_decay': config['weight_decay'],
        'cos_lr': config.get('cos_lr', False),
        'warmup_epochs': config['warmup_epochs'],
        'patience': config['patience'],
        'amp': config['amp'],
        'dropout': config['dropout'],
    }
    if 'cls' in config:
        args['cls'] = config['cls']
    if extra_overrides:
        args.update(extra_overrides)
    return args


def _validate_yolo_dataset(train_dir: str) -> None:
    """Raise if YOLO dataset dir is missing or has no train images."""
    p = Path(train_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {train_dir}\n"
            "Run: python scripts/prepare_dataset.py [--equipment] etc."
        )
    train_imgs = p / 'images' / 'train'
    if not train_imgs.exists():
        raise FileNotFoundError(
            f"Training images not found: {train_imgs}\n"
            "Ensure dataset has images/train/ structure. Run: python scripts/prepare_dataset.py"
        )
    n = len(list(train_imgs.glob('*.jpg')) + list(train_imgs.glob('*.png')))
    if n == 0:
        raise FileNotFoundError(
            f"No images in {train_imgs}\n"
            "Prepare the dataset first: python scripts/prepare_dataset.py"
        )


def prepare_data_yaml(train_dir: str, names: List[str]) -> Path:
    """
    Prepare YOLO data.yaml file.
    
    Args:
        train_dir: Directory containing training data
        names: List of class names
        
    Returns:
        Path to created data.yaml file
    """
    data_yaml_path = Path(train_dir) / 'data.yaml'
    abs_train_dir = Path(train_dir).resolve()
    data_yaml = {
        'path': str(abs_train_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(names),
        'names': names,
    }
    with data_yaml_path.open('w') as f:
        yaml.dump(data_yaml, f, default_flow_style=None, sort_keys=False)
    return data_yaml_path


def _log_yolo_row_to_tensorboard(row: dict, writer: SummaryWriter, lock: Optional[threading.Lock] = None) -> None:
    """Log a single YOLO results.csv row to TensorBoard."""
    # SummaryWriter is not thread-safe; use lock when writing from monitor thread
    def _do_log():
        epoch = int(float(row['epoch']))
        writer.add_scalar('loss/train_box', float(row['train/box_loss']), epoch)
        writer.add_scalar('loss/train_cls', float(row['train/cls_loss']), epoch)
        writer.add_scalar('loss/train_dfl', float(row['train/dfl_loss']), epoch)
        writer.add_scalar('loss/val_box', float(row['val/box_loss']), epoch)
        writer.add_scalar('loss/val_cls', float(row['val/cls_loss']), epoch)
        writer.add_scalar('loss/val_dfl', float(row['val/dfl_loss']), epoch)
        writer.add_scalar('metrics/precision', float(row['metrics/precision(B)']), epoch)
        writer.add_scalar('metrics/recall', float(row['metrics/recall(B)']), epoch)
        writer.add_scalar('metrics/mAP50', float(row['metrics/mAP50(B)']), epoch)
        writer.add_scalar('metrics/mAP50-95', float(row['metrics/mAP50-95(B)']), epoch)
        writer.add_scalar('learning_rate', float(row['lr/pg0']), epoch)
        writer.flush()

    try:
        if lock:
            with lock:
                _do_log()
        else:
            _do_log()
    except (ValueError, KeyError) as e:
        import warnings
        warnings.warn(f"[TensorBoard] Failed to log row (epoch={row.get('epoch')}): {e}")


def _monitor_yolo_results(csv_path: Path, writer: SummaryWriter, stop_event: threading.Event,
                          lock: threading.Lock) -> None:
    """Monitor results.csv in real-time and log to TensorBoard."""
    logged_epochs = set()
    while not stop_event.is_set():
        try:
            if csv_path.exists():
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        epoch = int(float(row.get('epoch', -1)))
                        if epoch >= 0 and epoch not in logged_epochs:
                            _log_yolo_row_to_tensorboard(row, writer, lock=lock)
                            logged_epochs.add(epoch)
        except Exception as e:
            print(f"[TensorBoard Monitor] Error: {e}")
        time.sleep(2)


def _log_yolo_results_to_tensorboard(csv_path: Path, writer: SummaryWriter,
                                     lock: Optional[threading.Lock] = None) -> None:
    """Parse YOLO results.csv and log metrics to TensorBoard (final pass)."""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                _log_yolo_row_to_tensorboard(row, writer, lock=lock)
    except Exception as e:
        print(f"Warning: Could not log YOLO results to TensorBoard: {e}")


def _train_yolo_detector(stage: str, train_dir: str, names: List[str], imgsz: Tuple[int, int] | int, 
                         use_rect: bool, default_batch_size: int, epochs: int, 
                         batch_size: Optional[int], extra_args: Optional[Dict],
                         augment_params: Optional[Dict] = None, model_size: str = 'nano',
                         resume: bool = False, warm_start: bool = False,
                         device: Optional[str] = None):
    """
    Shared training function for YOLO detectors.
    
    Args:
        stage: Training stage name
        train_dir: Directory containing training data
        names: List of class names
        imgsz: Image size (tuple or int)
        use_rect: Whether to use rectangular training
        default_batch_size: Default batch size
        epochs: Number of training epochs
        batch_size: Batch size (overrides default if provided)
        extra_args: Additional training arguments
        augment_params: Augmentation parameters (default: None, uses AUGMENT_PARAMS)
        resume: If True, force resume from last.pt
        warm_start: If True, force warm-start from last.pt (fresh run with pretrained)
        device: Device override (cuda/cpu); None uses auto
    """
    if not HAS_YOLO:
        raise ImportError("ultralytics package is required for YOLO training. Install with: pip install ultralytics")
    
    _validate_yolo_dataset(train_dir)
    data_yaml_path = prepare_data_yaml(train_dir, names)
    print(f"\n=== Training {stage} ===")
    # User override takes precedence over auto-detect
    if resume or warm_start:
        last_pt = RUNS_DIR / stage / 'weights' / 'last.pt'
        if last_pt.exists():
            resume_path, mode = str(last_pt), ('resume' if resume else 'warm_start')
        else:
            resume_path, mode = find_resume_weights(stage, target_epochs=epochs)
            if resume and resume_path:
                mode = 'resume'
            elif warm_start and resume_path:
                mode = 'warm_start'
    else:
        resume_path, mode = find_resume_weights(stage, target_epochs=epochs)

    # Get base model from config
    if model_size not in YOLO_MODEL_PATHS:
        raise ValueError(f"Invalid model_size: {model_size}. Must be one of: {list(YOLO_MODEL_PATHS.keys())}")

    base_model_path = YOLO_MODEL_PATHS[model_size]
    # Ensure models directory exists
    YOLO_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Get model name for download (YOLO will auto-download if it doesn't exist)
    model_name = f'yolo11{model_size[0]}.pt'  # 'n' for nano, 's' for small, 'm' for medium
    base_model = str(base_model_path)

    if batch_size is None:
        batch_size = default_batch_size

    # Skip if training already completed for target epochs
    if mode == 'already_complete':
        print(f"Training already complete for {stage} ({epochs} epochs). Skipping.")
        sys.exit(0)

    # Clean up old weights if starting fresh (to avoid YOLO trying to resume)
    if not resume_path and not mode:
        weights_dir = RUNS_DIR / stage / 'weights'
        if weights_dir.exists():
            import shutil
            print(f"Cleaning up old weights to start fresh training...")
            shutil.rmtree(weights_dir)
            weights_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    if resume_path and mode == 'resume':
        print(f"Resuming training for {stage} from {resume_path}")
        detector = YOLO(resume_path)
        resume_flag = True
    elif resume_path and mode == 'warm_start':
        print(f"Warm-starting {stage} from {resume_path}")
        detector = YOLO(resume_path)
        resume_flag = False
    else:
        print(f"Training {stage} from scratch using {model_name} (from config: model_size={model_size})")
        # YOLO will automatically download the model if it doesn't exist
        detector = YOLO(model_name)
        resume_flag = False
    
    # Use provided augment_params or default to AUGMENT_PARAMS
    if augment_params is None:
        augment_params = AUGMENT_PARAMS
    
    # Build training arguments
    train_args = {
        'data': str(data_yaml_path),
        'epochs': epochs,
        'imgsz': imgsz,
        'project': str(RUNS_DIR),
        'name': stage,
        'exist_ok': True,
        'batch': batch_size,
        'cache': 'disk',       # Keep disk cache: dataset (143GB) > available RAM (10GB)
        'rect': use_rect,
        'workers': 8,          # Added: parallel data loading (CPU has 24 cores)
        'close_mosaic': 15,    # Added: disable mosaic earlier for better convergence
        'plots': True,        # Save batch images for TensorBoard
        **augment_params,
    }
    
    if resume_flag:
        train_args['resume'] = True

    if device is not None:
        train_args['device'] = device
    
    if extra_args:
        train_args.update(extra_args)

    # Setup TensorBoard — each run gets its own subdirectory so event files
    # don't accumulate into a single directory across re-runs.
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_dir = RUNS_DIR / stage / 'tensorboard' / run_ts
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer_lock = threading.Lock()

    # Log hyperparameters text
    hparams_text = 'batch_size={}\nepochs={}\nimgsz={}\nlr0={}\nlrf={}\noptimizer={}\nmodel_size={}'.format(
        batch_size, epochs, imgsz[0] if isinstance(imgsz, tuple) else imgsz,
        train_args.get('lr0', 0.001), train_args.get('lrf', 0.01),
        train_args.get('optimizer', 'auto'), model_size
    )
    with writer_lock:
        writer.add_text('hyperparameters', hparams_text)

    print(f"Using imgsz={imgsz} for {stage}" + (f" with rect={use_rect}" if use_rect else ""))

    # Start background monitoring of results.csv (SummaryWriter is not thread-safe)
    results_csv = RUNS_DIR / stage / 'results.csv'
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_yolo_results,
        args=(results_csv, writer, stop_event, writer_lock),
        daemon=True,
    )
    monitor_thread.start()

    detector.train(**train_args)

    # Stop monitoring and do final log pass
    stop_event.set()
    monitor_thread.join(timeout=5)
    if results_csv.exists():
        _log_yolo_results_to_tensorboard(results_csv, writer, lock=writer_lock)

    writer.close()
    print(f"Finished training {stage}. Best weights: runs/{stage}/weights/best.pt")


def train_pole_detector(train_dir: Optional[str] = None, names: Optional[List[str]] = None, 
                       epochs: Optional[int] = None, batch_size: Optional[int] = None, 
                       extra_args: Optional[Dict] = None,
                       resume: bool = False, warm_start: bool = False, device: Optional[str] = None):
    """
    Train YOLO model for pole detection.
    
    Optimized for ~1900 training images:
    - Uses optimizer='auto' (AdamW) for better convergence
    - Allows horizontal flips (poles can be flipped without affecting detection)
    - Linear LR decay with proper final LR factor (lrf=0.01)
    - Reduced epochs (250) appropriate for dataset size
    - Optimized patience (25 epochs = 10% of total epochs)
    
    Args:
        train_dir: Directory containing training data
        names: List of class names (default: ['pole'])
        epochs: Number of training epochs (default: from POLE_DETECTION_CONFIG)
        batch_size: Batch size for training (default: from POLE_DETECTION_CONFIG)
        extra_args: Additional training arguments to override defaults
    """
    # Use config values as defaults (single source of truth)
    if train_dir is None:
        train_dir = str(DATASET_DIRS[POLE_DETECTION])
    if epochs is None:
        epochs = POLE_DETECTION_CONFIG['epochs']
    if batch_size is None:
        batch_size = POLE_DETECTION_CONFIG['batch_size']

    pole_training_args = _yolo_training_args(
        POLE_DETECTION_CONFIG, {'cos_lr': False, **(extra_args or {})}
    )
    
    _train_yolo_detector(
        stage='pole_detection',
        train_dir=train_dir,
        names=names or ['pole'],
        imgsz=POLE_DETECTION_CONFIG['imgsz'],
        use_rect=POLE_DETECTION_CONFIG['use_rect'],
        default_batch_size=POLE_DETECTION_CONFIG['batch_size'],
        epochs=epochs,
        batch_size=batch_size,
        extra_args=pole_training_args,
        augment_params=POLE_AUGMENT_PARAMS,  # Use pole-specific augmentation params
        model_size=POLE_DETECTION_CONFIG['model_size'],
        resume=resume, warm_start=warm_start, device=device,
    )


def train_ruler_detector(train_dir: Optional[str] = None, names: Optional[List[str]] = None,
                         epochs: Optional[int] = None, batch_size: Optional[int] = None,
                         extra_args: Optional[Dict] = None,
                         resume: bool = False, warm_start: bool = False, device: Optional[str] = None):
    """
    Train YOLO model for ruler detection.
    
    Optimized for ~3600 training images (larger dataset):
    - Uses optimizer='auto' (AdamW) which works better for this task
    - Linear LR decay (cos_lr=False) instead of cosine for more stable training
    - Proper final LR factor (lrf=0.01) for meaningful learning throughout training
    - Optimized patience (30 epochs = 10% of total epochs)
    - Longer warmup for better stability with larger dataset
    
    Args:
        train_dir: Directory containing training data
        names: List of class names (default: ['ruler'])
        epochs: Number of training epochs (default: from RULER_DETECTION_CONFIG)
        batch_size: Batch size for training (default: from RULER_DETECTION_CONFIG)
        extra_args: Additional training arguments to override defaults
    """
    # Use config values as defaults (single source of truth)
    if train_dir is None:
        train_dir = str(DATASET_DIRS[RULER_DETECTION])
    if epochs is None:
        epochs = RULER_DETECTION_CONFIG['epochs']
    if batch_size is None:
        batch_size = RULER_DETECTION_CONFIG['batch_size']

    stability_args = _yolo_training_args(
        RULER_DETECTION_CONFIG, {'cos_lr': False, **(extra_args or {})}
    )
    
    _train_yolo_detector(
        stage='ruler_detection',
        train_dir=train_dir,
        names=names or ['ruler'],
        imgsz=RULER_DETECTION_CONFIG['imgsz'],
        use_rect=RULER_DETECTION_CONFIG['use_rect'],
        default_batch_size=RULER_DETECTION_CONFIG['batch_size'],
        epochs=epochs,
        batch_size=batch_size,
        extra_args=stability_args,
        augment_params=RULER_AUGMENT_PARAMS,  # Use ruler-specific augmentation params
        model_size=RULER_DETECTION_CONFIG['model_size'],
        resume=resume, warm_start=warm_start, device=device,
    )


def train_equipment_detector(train_dir: Optional[str] = None,
                             names: Optional[List[str]] = None,
                             epochs: Optional[int] = None,
                             batch_size: Optional[int] = None,
                             lr0: Optional[float] = None,
                             extra_args: Optional[Dict] = None,
                             resume: bool = False, warm_start: bool = False, device: Optional[str] = None):
    """
    Train YOLO model for equipment detection (riser, transformer, street_light).

    Args:
        train_dir: Directory containing training data (default: DATASETS_DIR/EQUIPMENT_DETECTION)
        names: List of class names (default: EQUIPMENT_CLASS_NAMES)
        epochs: Number of training epochs (default: from EQUIPMENT_DETECTION_CONFIG)
        batch_size: Batch size for training (default: from EQUIPMENT_DETECTION_CONFIG)
        lr0: Learning rate override (default: from config)
        extra_args: Additional training arguments to override defaults
    """
    if train_dir is None:
        train_dir = str(DATASETS_DIR / EQUIPMENT_DETECTION)
    if epochs is None:
        epochs = EQUIPMENT_DETECTION_CONFIG['epochs']
    if batch_size is None:
        batch_size = EQUIPMENT_DETECTION_CONFIG['batch_size']

    overrides = {'cos_lr': True}
    if lr0 is not None:
        overrides['lr0'] = lr0
    if extra_args:
        overrides.update(extra_args)
    equipment_training_args = _yolo_training_args(EQUIPMENT_DETECTION_CONFIG, overrides)

    _train_yolo_detector(
        stage='equipment_detection',
        train_dir=train_dir,
        names=names or list(EQUIPMENT_CLASS_NAMES),
        imgsz=EQUIPMENT_DETECTION_CONFIG['imgsz'],
        use_rect=EQUIPMENT_DETECTION_CONFIG['use_rect'],
        default_batch_size=EQUIPMENT_DETECTION_CONFIG['batch_size'],
        epochs=epochs,
        batch_size=batch_size,
        extra_args=equipment_training_args,
        augment_params=EQUIPMENT_AUGMENT_PARAMS,
        model_size=EQUIPMENT_DETECTION_CONFIG['model_size'],
        resume=resume, warm_start=warm_start, device=device,
    )


def train_attachment_detector(train_dir: Optional[str] = None,
                               names: Optional[List[str]] = None,
                               epochs: Optional[int] = None,
                               batch_size: Optional[int] = None,
                               lr0: Optional[float] = None,
                               extra_args: Optional[Dict] = None,
                               resume: bool = False, warm_start: bool = False, device: Optional[str] = None):
    """
    Train YOLO model for attachment detection (comm, down_guy).

    Args:
        train_dir: Directory containing training data (default: DATASETS_DIR/attachment_detection)
        names: List of class names (default: ATTACHMENT_CLASS_NAMES)
        epochs: Number of training epochs (default: from ATTACHMENT_DETECTION_CONFIG)
        batch_size: Batch size for training (default: from ATTACHMENT_DETECTION_CONFIG)
        lr0: Learning rate override (default: from config)
        extra_args: Additional training arguments to override defaults
    """
    if train_dir is None:
        train_dir = str(DATASETS_DIR / "attachment_detection")
    if epochs is None:
        epochs = ATTACHMENT_DETECTION_CONFIG['epochs']
    if batch_size is None:
        batch_size = ATTACHMENT_DETECTION_CONFIG['batch_size']

    overrides = {'cos_lr': True}
    if lr0 is not None:
        overrides['lr0'] = lr0
    if extra_args:
        overrides.update(extra_args)
    attachment_training_args = _yolo_training_args(ATTACHMENT_DETECTION_CONFIG, overrides)

    _train_yolo_detector(
        stage='attachment_detection',
        train_dir=train_dir,
        names=names or list(ATTACHMENT_CLASS_NAMES),
        imgsz=ATTACHMENT_DETECTION_CONFIG['imgsz'],
        use_rect=ATTACHMENT_DETECTION_CONFIG['use_rect'],
        default_batch_size=ATTACHMENT_DETECTION_CONFIG['batch_size'],
        epochs=epochs,
        batch_size=batch_size,
        extra_args=attachment_training_args,
        augment_params=ATTACHMENT_AUGMENT_PARAMS,
        model_size=ATTACHMENT_DETECTION_CONFIG['model_size'],
        resume=resume, warm_start=warm_start, device=device,
    )


def _print_keypoint_training_summary(
    history: dict, best_val_loss: float, best_val_acc: float,
    weights_path, title: str = "Training"
) -> None:
    """Print standardized keypoint training completion summary."""
    final_acc_3 = history.get('val_kp_acc_3inch', [])[-1] if history.get('val_kp_acc_3inch') else 0.0
    final_acc_2 = history.get('val_kp_acc_2inch', [])[-1] if history.get('val_kp_acc_2inch') else 0.0
    final_acc_1 = history.get('val_kp_acc_1inch', [])[-1] if history.get('val_kp_acc_1inch') else 0.0
    final_acc_05 = history.get('val_kp_acc_0_5inch', [])[-1] if history.get('val_kp_acc_0_5inch') else 0.0
    print(
        "\n" + "=" * 70 + "\n"
        f"✓ {title} completed!\n"
        f"Best val loss: {best_val_loss:.4f}\n"
        f"Final PCK (vertical): {final_acc_3*100:.1f}% (≤3\") | {final_acc_2*100:.1f}% (≤2\") | "
        f"{final_acc_1*100:.1f}% (≤1\") | {final_acc_05*100:.1f}% (≤0.5\")\n"
        f"Best PCK (vertical): {best_val_acc * 100:.1f}%\n"
        f"Weights: {weights_path}\n"
        + "=" * 70 + "\n"
    )


def train_model(model, train_loader, val_loader, num_epochs=50, patience=10,
                use_focal_loss=False,
                device=None, checkpoint_dir=None, learning_rate=3e-4,
                num_keypoints=None):
    """
    Train a keypoint detection model with heatmap regression.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        patience: Early stopping patience
        use_focal_loss: Whether to use FocalHeatmapLoss instead of UnimodalHeatmapLoss
        device: Device to train on (default: 'cuda' if available, else 'cpu')
        checkpoint_dir: Directory to save checkpoints (default: model-specific)
        learning_rate: Learning rate for optimizer (default: 3e-4)
        num_keypoints: Number of keypoints (default: None, infers from config NUM_KEYPOINTS)
        
    Returns:
        Tuple of (history, best_val_loss, best_val_acc)
    """
    from .config import NUM_KEYPOINTS, HEATMAP_HEIGHT
    
    # Infer number of keypoints from model if not provided
    if num_keypoints is None:
        num_keypoints = NUM_KEYPOINTS
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    if use_focal_loss:
        criterion = FocalHeatmapLoss(alpha=2, beta=4)
        print("Using FocalHeatmapLoss")
    else:
        criterion = UnimodalHeatmapLoss(
            pos_weight=8.0,
            vertical_focus_weight=0.1,
        )
        print("Using UnimodalHeatmapLoss")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.5e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.6)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-6)

    # Determine checkpoint path
    if checkpoint_dir is None:
        checkpoint_dir = RUNS_DIR / 'keypoint_detection'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = checkpoint_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weights_dir / 'last.pth'
    best_model_path = weights_dir / 'best.pth'
    
    start_epoch = 1
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [], 
        'val_kp_acc_3inch': [], 'val_kp_acc_2inch': [], 
        'val_kp_acc_1inch': [], 'val_kp_acc_0_5inch': []
    }
    
    if checkpoint_path.exists():
        print(f"Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        old_history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
        history = {
            'train_loss': old_history.get('train_loss', []),
            'val_loss': old_history.get('val_loss', []),
            'val_kp_acc_3inch': old_history.get('val_kp_acc_3inch', []),
            'val_kp_acc_2inch': old_history.get('val_kp_acc_2inch', []),
            'val_kp_acc_1inch': old_history.get('val_kp_acc_1inch', []),
            'val_kp_acc_0_5inch': old_history.get('val_kp_acc_0_5inch', old_history.get('val_kp_acc', []))
        }
        if 'plateau_scheduler_best' in checkpoint:
            plateau_scheduler.best = checkpoint['plateau_scheduler_best']
            plateau_scheduler.num_bad_epochs = checkpoint.get('plateau_scheduler_num_bad_epochs', 0)
        print(f"Resumed from epoch {start_epoch-1}, best_val_loss={best_val_loss:.4f}, best_val_acc={best_val_acc*100:.1f}%")
    else:
        print("Starting training from scratch")

    # Setup TensorBoard — each run gets its own subdirectory so event files
    # don't accumulate into a single directory across re-runs.
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_dir = checkpoint_dir / 'tensorboard' / run_ts
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    # Log hyperparameters text
    if start_epoch == 1:
        hparams_text = 'learning_rate={}\nbatch_size={}\nnum_epochs={}\npatience={}\nuse_focal_loss={}\nnum_keypoints={}'.format(
            learning_rate, train_loader.batch_size, num_epochs, patience, use_focal_loss, num_keypoints
        )
        writer.add_text('hyperparameters', hparams_text)

        # Try to log model graph
        try:
            sample_batch = next(iter(train_loader))
            writer.add_graph(model, sample_batch[0][:1].to(device))
        except Exception as e:
            pass  # Don't fail training if graph logging fails

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        running_train = 0.0
        for images, targets, _, _, _, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            pred_heatmaps = torch.sigmoid(logits)
            loss = criterion(pred_heatmaps, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train += loss.item()

        mean_train = running_train / max(len(train_loader), 1)

        model.eval()
        running_val = 0.0
        kp_hits_3inch = 0.0
        kp_hits_2inch = 0.0
        kp_hits_1inch = 0.0
        kp_hits_0_5inch = 0.0
        kp_total = 0.0
        with torch.no_grad():
            for images, targets, kp_coords, vis, orig_dims, ppi_values in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                kp_coords = kp_coords.to(device)
                vis = vis.to(device)
                orig_dims = orig_dims.to(device)
                ppi_values = ppi_values.to(device)

                logits = model(images)
                pred_heatmaps = torch.sigmoid(logits)
                loss = criterion(pred_heatmaps, targets)
                running_val += loss.item()

                pred_heatmaps_np = pred_heatmaps.cpu().numpy()
                for batch_idx in range(pred_heatmaps_np.shape[0]):
                    for kp_idx in range(num_keypoints):
                        if vis[batch_idx, kp_idx] <= 0.5:
                            continue
                        
                        hm = pred_heatmaps_np[batch_idx, kp_idx]
                        hm_h, hm_w = hm.shape[0], hm.shape[1]
                        y_int, x_int = np.unravel_index(np.argmax(hm), hm.shape)
                        y_sub, x_sub = float(y_int), float(x_int)
                        
                        # Transform from heatmap/resized space to original image space
                        # Dataset stores kp_coords in resized space; heatmap matches resize dimensions
                        orig_h = orig_dims[batch_idx, 0].item()
                        gt_y_resized = kp_coords[batch_idx, kp_idx, 1].item()
                        scale_denom = max(hm_h - 1, 1)
                        pred_y_orig = (y_sub / scale_denom) * (orig_h - 1) if orig_h > 1 else y_sub
                        gt_y_orig = (gt_y_resized / scale_denom) * (orig_h - 1) if orig_h > 1 else gt_y_resized
                        pixel_error = abs(pred_y_orig - gt_y_orig)
                        pixels_per_inch = ppi_values[batch_idx, 0].item()
                        inch_error = pixel_error / pixels_per_inch if pixels_per_inch > 0 else float('inf')
                        
                        if inch_error <= 3.0:
                            kp_hits_3inch += 1
                        if inch_error <= 2.0:
                            kp_hits_2inch += 1
                        if inch_error <= 1.0:
                            kp_hits_1inch += 1
                        if inch_error <= 0.5:
                            kp_hits_0_5inch += 1
                        kp_total += 1

        mean_val = running_val / max(len(val_loader), 1)
        kp_acc_3inch = kp_hits_3inch / kp_total if kp_total else 0.0
        kp_acc_2inch = kp_hits_2inch / kp_total if kp_total else 0.0
        kp_acc_1inch = kp_hits_1inch / kp_total if kp_total else 0.0
        kp_acc_0_5inch = kp_hits_0_5inch / kp_total if kp_total else 0.0

        history['train_loss'].append(mean_train)
        history['val_loss'].append(mean_val)
        history['val_kp_acc_3inch'].append(kp_acc_3inch)
        history['val_kp_acc_2inch'].append(kp_acc_2inch)
        history['val_kp_acc_1inch'].append(kp_acc_1inch)
        history['val_kp_acc_0_5inch'].append(kp_acc_0_5inch)

        # Log to TensorBoard
        writer.add_scalar('loss/train', mean_train, epoch)
        writer.add_scalar('loss/val', mean_val, epoch)
        writer.add_scalar('pck/3inch', kp_acc_3inch, epoch)
        writer.add_scalar('pck/2inch', kp_acc_2inch, epoch)
        writer.add_scalar('pck/1inch', kp_acc_1inch, epoch)
        writer.add_scalar('pck/0_5inch', kp_acc_0_5inch, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.flush()

        # Log sample heatmaps every 10 epochs
        if epoch % 10 == 0:
            try:
                with torch.no_grad():
                    sample_images, _, _, _, _, _ = next(iter(val_loader))
                    sample_pred = torch.sigmoid(model(sample_images[:4].to(device)))
                    writer.add_images('heatmaps/predictions', sample_pred[:4], epoch, dataformats='NCHW')
            except Exception:
                pass  # Don't fail training if visualization fails

        kp_acc = kp_acc_1inch
        scheduler.step()
        plateau_scheduler.step(mean_val)
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        best_val_acc = max(best_val_acc, kp_acc)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'plateau_scheduler_best': plateau_scheduler.best,
            'plateau_scheduler_num_bad_epochs': plateau_scheduler.num_bad_epochs,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'patience_counter': patience_counter,
            'history': history,
            'batch_size': train_loader.batch_size,
            'learning_rate': learning_rate,
        }, checkpoint_path)

        # Save metrics to JSON for easy access (no PyTorch required)
        metrics_json_path = checkpoint_dir / 'metrics.json'
        import json
        metrics_data = {
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'final_metrics': {
                'train_loss': mean_train,
                'val_loss': mean_val,
                'pck_3_inch': kp_acc_3inch,
                'pck_2_inch': kp_acc_2inch,
                'pck_1_inch': kp_acc_1inch,
                'pck_0_5_inch': kp_acc_0_5inch,
            },
            'training_config': {
                'batch_size': train_loader.batch_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'patience': patience,
                'use_focal_loss': use_focal_loss,
                'num_keypoints': num_keypoints,
            },
            'history': history,
        }
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:03d} | train {mean_train:.4f} | val {mean_val:.4f} | "
            f"PCK (vertical): {kp_acc_3inch*100:.1f}% (≤3\") | {kp_acc_2inch*100:.1f}% (≤2\") | {kp_acc_1inch*100:.1f}% (≤1\") | {kp_acc_0_5inch*100:.1f}% (≤0.5\") | "
            f"best {best_val_loss:.4f} | LR {current_lr:.2e}"
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    writer.close()
    return history, best_val_loss, best_val_acc


def train_ruler_marking_detector(train_dir: Optional[str] = None,
                                  epochs: Optional[int] = None, patience: Optional[int] = None,
                                  batch_size: Optional[int] = None, learning_rate: Optional[float] = None,
                                  use_focal_loss: Optional[bool] = None,
                                  augmentation_params: Optional[Dict] = None,
                                  geometric_augmentations: Optional[Dict] = None,
                                  resume: bool = False, device: Optional[str] = None):
    """
    Train HRNet model for ruler marking detection (keypoint detection).
    
    Args:
        train_dir: Directory containing training data
        epochs: Number of training epochs (default: 50)
        patience: Early stopping patience (default: 10)
        batch_size: Batch size for training (default: 8)
        learning_rate: Learning rate for optimizer (default: 3e-4)
        use_focal_loss: Whether to use FocalHeatmapLoss (default: False)
        augmentation_params: Dict with 'brightness', 'contrast', 'saturation' for ColorJitter (default: None)
        geometric_augmentations: Dict with 'translate_x', 'translate_y', 'scale_min', 'scale_max', 'rotate'
                                  to simulate YOLO detection variations (default: None, uses defaults)
        resume: If True, resume from best.pth checkpoint; if False, start fresh with HRNet weights
    """
    cfg = RULER_MARKING_DETECTION_CONFIG
    if train_dir is None:
        train_dir = str(DATASET_DIRS['ruler_marking_detection'])
    if epochs is None:
        epochs = cfg['epochs']
    if patience is None:
        patience = cfg['patience']
    if batch_size is None:
        batch_size = cfg['batch_size']
    if learning_rate is None:
        learning_rate = cfg['learning_rate']
    if use_focal_loss is None:
        use_focal_loss = cfg['use_focal_loss']
    if augmentation_params is None:
        augmentation_params = cfg['augmentation_params']
    if geometric_augmentations is None:
        geometric_augmentations = cfg['geometric_augmentations']

    import torch
    import torch.optim as optim
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from pathlib import Path
    from .models import KeypointDetector
    from .datasets import KeypointDataset
    from .config import (
        KEYPOINT_NAMES, NUM_KEYPOINTS, HEATMAP_HEIGHT, HEATMAP_WIDTH,
        RESIZE_HEIGHT, RESIZE_WIDTH, HRNET_WEIGHTS_PATH,
        IMAGENET_MEAN, IMAGENET_STD,
    )
    import numpy as np
    _validate_yolo_dataset(train_dir)
    image_mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    image_std = np.array(IMAGENET_STD, dtype=np.float32)
    
    device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = RUNS_DIR / 'ruler_marking_detection'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare transforms
    erasing_prob = augmentation_params.get('erasing_prob', 0.0)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
        transforms.ColorJitter(
            brightness=augmentation_params.get('brightness', 0.1),
            contrast=augmentation_params.get('contrast', 0.1),
            saturation=augmentation_params.get('saturation', 0.1),
            hue=augmentation_params.get('hue', 0.0),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
        *([transforms.RandomErasing(p=erasing_prob, scale=(0.02, 0.15), ratio=(0.3, 3.0), value=0)] if erasing_prob > 0 else []),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])
    
    # Create datasets
    train_dataset = KeypointDataset(
        image_dir=Path(f'{train_dir}/images/train'),
        label_dir=Path(f'{train_dir}/labels/train'),
        transform=train_transform,
        min_visible_keypoints=cfg.get('min_visible_keypoints', 5),
        geometric_augmentations=geometric_augmentations  # Only applied during training
    )
    
    val_dataset = KeypointDataset(
        image_dir=Path(f'{train_dir}/images/val'),
        label_dir=Path(f'{train_dir}/labels/val'),
        transform=val_transform,
        min_visible_keypoints=cfg.get('min_visible_keypoints', 5),
        geometric_augmentations=None  # No geometric augmentations for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Create model
    model = KeypointDetector(
        num_keypoints=NUM_KEYPOINTS,
        heatmap_size=(HEATMAP_HEIGHT, HEATMAP_WIDTH),
        weights_path=HRNET_WEIGHTS_PATH
    )

    # Load checkpoint if resuming
    checkpoint_path = checkpoint_dir / 'weights' / 'best.pth'
    if resume and checkpoint_path.exists():
        print(f"Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    elif resume and not checkpoint_path.exists():
        print(f"Warning: --resume specified but no checkpoint found at {checkpoint_path}")
        print("Starting fresh training instead")

    history, best_val_loss, best_val_acc = train_model(
        model, train_loader, val_loader,
        num_epochs=epochs,
        patience=patience,
        use_focal_loss=use_focal_loss,
        device=device,
        checkpoint_dir=checkpoint_dir,
        learning_rate=learning_rate,
        num_keypoints=NUM_KEYPOINTS,
    )
    
    _print_keypoint_training_summary(
        history, best_val_loss, best_val_acc,
        checkpoint_dir / 'weights' / 'best.pth', 'Ruler Marking Detection Training'
    )
    return history, best_val_loss, best_val_acc


def train_pole_top_detector(train_dir: Optional[str] = None,
                            epochs: Optional[int] = None, patience: Optional[int] = None,
                            batch_size: Optional[int] = None, learning_rate: Optional[float] = None,
                            use_focal_loss: Optional[bool] = None,
                            augmentation_params: Optional[Dict] = None,
                            geometric_augmentations: Optional[Dict] = None,
                            resume: bool = False, device: Optional[str] = None):
    """
    Train HRNet model for pole top detection (keypoint detection).
    
    Args:
        train_dir: Directory containing training data
        epochs: Number of training epochs (default: 50)
        patience: Early stopping patience (default: 10)
        batch_size: Batch size for training (default: 8)
        learning_rate: Learning rate for optimizer (default: 3e-4)
        use_focal_loss: Whether to use FocalHeatmapLoss (default: False)
        augmentation_params: Dict with 'brightness', 'contrast', 'saturation' for ColorJitter (default: None)
        geometric_augmentations: Dict with 'translate_x', 'translate_y', 'scale_min', 'scale_max', 'rotate'
                                  to simulate YOLO detection variations (default: None, uses defaults)
        resume: If True, resume from best.pth checkpoint; if False, start fresh with HRNet weights
    """
    cfg = POLE_TOP_DETECTION_CONFIG
    if train_dir is None:
        train_dir = str(DATASET_DIRS['pole_top_detection'])
    if epochs is None:
        epochs = cfg['epochs']
    if patience is None:
        patience = cfg['patience']
    if batch_size is None:
        batch_size = cfg['batch_size']
    if learning_rate is None:
        learning_rate = cfg['learning_rate']
    if use_focal_loss is None:
        use_focal_loss = cfg.get('use_focal_loss', False)
    if augmentation_params is None:
        augmentation_params = cfg['augmentation_params']
    if geometric_augmentations is None:
        geometric_augmentations = cfg['geometric_augmentations']

    import torch
    import torch.optim as optim
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from pathlib import Path
    from .models import KeypointDetector
    from .datasets import PoleTopKeypointDataset
    from .config import (
        POLE_TOP_RESIZE_HEIGHT, POLE_TOP_RESIZE_WIDTH,
        POLE_TOP_HEATMAP_HEIGHT, POLE_TOP_HEATMAP_WIDTH,
        POLE_TOP_NUM_KEYPOINTS, HRNET_WEIGHTS_PATH
    )
    
    _validate_yolo_dataset(train_dir)
    device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = RUNS_DIR / 'pole_top_detection'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare transforms
    erasing_prob = augmentation_params.get('erasing_prob', 0.0)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((POLE_TOP_RESIZE_HEIGHT, POLE_TOP_RESIZE_WIDTH)),
        transforms.ColorJitter(
            brightness=augmentation_params.get('brightness', 0.1),
            contrast=augmentation_params.get('contrast', 0.1),
            saturation=augmentation_params.get('saturation', 0.1),
            hue=augmentation_params.get('hue', 0.0),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        *([transforms.RandomErasing(p=erasing_prob, scale=(0.02, 0.15), ratio=(0.3, 3.0), value=0)] if erasing_prob > 0 else []),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((POLE_TOP_RESIZE_HEIGHT, POLE_TOP_RESIZE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Create datasets
    train_dataset = PoleTopKeypointDataset(
        image_dir=Path(f'{train_dir}/images/train'),
        label_dir=Path(f'{train_dir}/labels/train'),
        transform=train_transform,
        geometric_augmentations=geometric_augmentations  # Only applied during training
    )
    
    val_dataset = PoleTopKeypointDataset(
        image_dir=Path(f'{train_dir}/images/val'),
        label_dir=Path(f'{train_dir}/labels/val'),
        transform=val_transform,
        geometric_augmentations=None  # No geometric augmentations for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Create model
    model = KeypointDetector(
        num_keypoints=POLE_TOP_NUM_KEYPOINTS,
        heatmap_size=(POLE_TOP_HEATMAP_HEIGHT, POLE_TOP_HEATMAP_WIDTH),
        weights_path=HRNET_WEIGHTS_PATH
    )

    # Load checkpoint if resuming
    checkpoint_path = checkpoint_dir / 'weights' / 'best.pth'
    if resume and checkpoint_path.exists():
        print(f"Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    elif resume and not checkpoint_path.exists():
        print(f"Warning: --resume specified but no checkpoint found at {checkpoint_path}")
        print("Starting fresh training instead")

    history, best_val_loss, best_val_acc = train_model(
        model, train_loader, val_loader,
        num_epochs=epochs,
        patience=patience,
        use_focal_loss=use_focal_loss,
        device=device,
        checkpoint_dir=checkpoint_dir,
        learning_rate=learning_rate,
        num_keypoints=POLE_TOP_NUM_KEYPOINTS,
    )

    _print_keypoint_training_summary(
        history, best_val_loss, best_val_acc,
        checkpoint_dir / 'weights' / 'best.pth', 'Pole Top Detection Training'
    )
    return history, best_val_loss, best_val_acc


def _train_keypoint_detector_impl(
    keypoint_type: str,
    configs: Dict,
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    use_focal_loss: Optional[bool] = None,
    augmentation_params: Optional[Dict] = None,
    geometric_augmentations: Optional[Dict] = None,
    resume: bool = False,
    device: Optional[str] = None,
) -> Tuple:
    """Shared implementation for equipment and attachment keypoint training."""
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from pathlib import Path
    from .models import KeypointDetector
    from .datasets import EquipmentKeypointDataset
    from .config import HRNET_WEIGHTS_PATH

    if keypoint_type not in configs:
        raise ValueError(f"Unknown keypoint_type: {keypoint_type}. Must be one of {list(configs.keys())}")

    cfg, num_keypoints, _ = configs[keypoint_type]
    if train_dir is None:
        train_dir = str(DATASET_DIRS[f'{keypoint_type}_keypoint_detection'])
    _validate_yolo_dataset(train_dir)

    if epochs is None:
        epochs = cfg['epochs']
    if patience is None:
        patience = cfg['patience']
    if batch_size is None:
        batch_size = cfg['batch_size']
    if learning_rate is None:
        learning_rate = cfg['learning_rate']
    if use_focal_loss is None:
        use_focal_loss = cfg['use_focal_loss']
    if augmentation_params is None:
        augmentation_params = cfg['augmentation_params']
    if geometric_augmentations is None:
        geometric_augmentations = cfg['geometric_augmentations']

    resize_h = cfg['resize_height']
    resize_w = cfg['resize_width']
    heatmap_h = cfg['heatmap_height']
    heatmap_w = cfg['heatmap_width']

    device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = RUNS_DIR / f'{keypoint_type}_keypoint_detection'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    erasing_prob = augmentation_params.get('erasing_prob', 0.0)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resize_h, resize_w)),
        transforms.ColorJitter(
            brightness=augmentation_params.get('brightness', 0.1),
            contrast=augmentation_params.get('contrast', 0.1),
            saturation=augmentation_params.get('saturation', 0.1),
            hue=augmentation_params.get('hue', 0.0),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        *([transforms.RandomErasing(p=erasing_prob, scale=(0.02, 0.15), ratio=(0.3, 3.0), value=0)] if erasing_prob > 0 else []),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resize_h, resize_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_dataset = EquipmentKeypointDataset(
        image_dir=Path(f'{train_dir}/images/train'),
        label_dir=Path(f'{train_dir}/labels/train'),
        num_keypoints=num_keypoints,
        resize_height=resize_h, resize_width=resize_w,
        heatmap_height=heatmap_h, heatmap_width=heatmap_w,
        transform=train_transform,
        geometric_augmentations=geometric_augmentations,
    )
    val_dataset = EquipmentKeypointDataset(
        image_dir=Path(f'{train_dir}/images/val'),
        label_dir=Path(f'{train_dir}/labels/val'),
        num_keypoints=num_keypoints,
        resize_height=resize_h, resize_width=resize_w,
        heatmap_height=heatmap_h, heatmap_width=heatmap_w,
        transform=val_transform,
        geometric_augmentations=None,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = KeypointDetector(
        num_keypoints=num_keypoints,
        heatmap_size=(heatmap_h, heatmap_w),
        weights_path=HRNET_WEIGHTS_PATH
    )

    checkpoint_path = checkpoint_dir / 'weights' / 'best.pth'
    if resume and checkpoint_path.exists():
        print(f"Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    elif resume and not checkpoint_path.exists():
        print(f"Warning: --resume specified but no checkpoint found at {checkpoint_path}")
        print("Starting fresh training instead")

    title = f'{keypoint_type.replace("_", " ").title()} Keypoint Detection Training'
    history, best_val_loss, best_val_acc = train_model(
        model, train_loader, val_loader,
        num_epochs=epochs,
        patience=patience,
        use_focal_loss=use_focal_loss,
        device=device,
        checkpoint_dir=checkpoint_dir,
        learning_rate=learning_rate,
        num_keypoints=num_keypoints,
    )
    _print_keypoint_training_summary(
        history, best_val_loss, best_val_acc,
        checkpoint_dir / 'weights' / 'best.pth', title
    )
    return history, best_val_loss, best_val_acc


def train_equipment_keypoint_detector(equipment_type: str,
                                       train_dir: Optional[str] = None,
                                       val_dir: Optional[str] = None,
                                       epochs: Optional[int] = None,
                                       patience: Optional[int] = None,
                                       batch_size: Optional[int] = None,
                                       learning_rate: Optional[float] = None,
                                       use_focal_loss: Optional[bool] = None,
                                       augmentation_params: Optional[Dict] = None,
                                       geometric_augmentations: Optional[Dict] = None,
                                       resume: bool = False, device: Optional[str] = None):
    """Train HRNet model for equipment keypoint detection (riser, transformer, street_light)."""
    return _train_keypoint_detector_impl(
        keypoint_type=equipment_type,
        configs=EQUIPMENT_KEYPOINT_CONFIGS,
        train_dir=train_dir,
        val_dir=val_dir,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_focal_loss=use_focal_loss,
        augmentation_params=augmentation_params,
        geometric_augmentations=geometric_augmentations,
        resume=resume,
        device=device,
    )


def train_attachment_keypoint_detector(attachment_type: str,
                                       train_dir: Optional[str] = None,
                                       val_dir: Optional[str] = None,
                                       epochs: Optional[int] = None,
                                       patience: Optional[int] = None,
                                       batch_size: Optional[int] = None,
                                       learning_rate: Optional[float] = None,
                                       use_focal_loss: Optional[bool] = None,
                                       augmentation_params: Optional[Dict] = None,
                                       geometric_augmentations: Optional[Dict] = None,
                                       resume: bool = False, device: Optional[str] = None):
    """Train HRNet model for attachment keypoint detection (comm, down_guy)."""
    return _train_keypoint_detector_impl(
        keypoint_type=attachment_type,
        configs=ATTACHMENT_KEYPOINT_CONFIGS,
        train_dir=train_dir,
        val_dir=val_dir,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_focal_loss=use_focal_loss,
        augmentation_params=augmentation_params,
        geometric_augmentations=geometric_augmentations,
        resume=resume,
        device=device,
    )
