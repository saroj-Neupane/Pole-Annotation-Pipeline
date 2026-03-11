#!/usr/bin/env python
"""
Unified training script for all models in the Pole Midspan Photo Calibration project.

Supports training any model with command-line flags for easy reproducibility and automation.

Examples:
    # Train pole detector with defaults
    python train.py --model pole_detection

    # Train ruler marking detector with custom hyperparameters
    python train.py --model ruler_marking_detection --epochs 150 --batch-size 64 --lr 1e-3

    # Resume interrupted training
    python train.py --model pole_top_detection --resume

    # Warm start equipment detector
    python train.py --model equipment_detection --warm-start --epochs 50

    # Show all options
    python train.py --help
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from src.training_utils import (
    train_pole_detector,
    train_ruler_detector,
    train_equipment_detector,
    train_attachment_detector,
    train_ruler_marking_detector,
    train_pole_top_detector,
    train_equipment_keypoint_detector,
    train_attachment_keypoint_detector,
    clear_yolo_disk_cache_from_other_datasets,
)
from src.config import RUNS_DIR, KEYPOINT_MODEL_TO_TYPE, DATASET_DIRS

# Setup logging
def setup_logging(model_name: str, debug: bool = False):
    """Configure logging for training."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"train_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Training started for model: {model_name}")
    logger.info(f"Log file: {log_file}")
    return logger


def main():
    parser = argparse.ArgumentParser(
        description="Train any model in the Pole Midspan Photo Calibration project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --model pole_detection
  python train.py --model ruler_marking_detection --epochs 150 --batch-size 64
  python train.py --model pole_top_detection --resume
  python train.py --model equipment_detection --warm-start --epochs 50
        """
    )

    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=[
            'pole_detection',
            'ruler_detection',
            'ruler_marking_detection',
            'pole_top_detection',
            'equipment_detection',
            'attachment_detection',
            *KEYPOINT_MODEL_TO_TYPE.keys(),
        ],
        help='Model to train'
    )

    # Common hyperparameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides default)'
    )
    parser.add_argument(
        '--batch-size', '-bs',
        type=int,
        default=None,
        help='Batch size (overrides default)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides default)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Training device (auto=cuda if available else cpu)'
    )

    # Training mode
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint (continue training from last.pt)'
    )
    parser.add_argument(
        '--warm-start',
        action='store_true',
        help='Warm start from pretrained weights (start fresh with pretrained model)'
    )

    # Debug and logging
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging (verbose)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be trained without actually training'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        default=False,
        help='Clear YOLO *.npy cache from other datasets before training (frees disk for current model)'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.model, debug=args.debug)

    # Log parsed arguments
    logger.info(f"Arguments: {vars(args)}")

    # Validate conflicting options
    if args.resume and args.warm_start:
        logger.error("Cannot use --resume and --warm-start together!")
        sys.exit(1)

    # Model configuration and training function mapping
    model_configs = {
        # YOLO-based models (object detection)
        'pole_detection': {
            'trainer': train_pole_detector,
            'kwargs': {
                'train_dir': str(DATASET_DIRS['pole_detection']),
                'resume': args.resume,
                'warm_start': args.warm_start,
                'device': args.device if args.device != 'auto' else None,
            }
        },
        'ruler_detection': {
            'trainer': train_ruler_detector,
            'kwargs': {
                'train_dir': str(DATASET_DIRS['ruler_detection']),
                'resume': args.resume,
                'warm_start': args.warm_start,
                'device': args.device if args.device != 'auto' else None,
            }
        },
        'equipment_detection': {
            'trainer': train_equipment_detector,
            'kwargs': {
                'train_dir': str(DATASET_DIRS['equipment_detection']),
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr0': args.lr,
                'resume': args.resume,
                'warm_start': args.warm_start,
                'device': args.device if args.device != 'auto' else None,
            }
        },
        'attachment_detection': {
            'trainer': train_attachment_detector,
            'kwargs': {
                'train_dir': str(DATASET_DIRS['attachment_detection']),
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr0': args.lr,
                'resume': args.resume,
                'warm_start': args.warm_start,
                'device': args.device if args.device != 'auto' else None,
            }
        },

        # HRNet-based models (keypoint detection)
        'ruler_marking_detection': {
            'trainer': train_ruler_marking_detector,
            'kwargs': {
                'train_dir': str(DATASET_DIRS['ruler_marking_detection']),
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'resume': args.resume,
                'device': args.device if args.device != 'auto' else None,
            }
        },
        'pole_top_detection': {
            'trainer': train_pole_top_detector,
            'kwargs': {
                'train_dir': str(DATASET_DIRS['pole_top_detection']),
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'resume': args.resume,
                'device': args.device if args.device != 'auto' else None,
            }
        },

        # Keypoint detection (equipment + attachment)
        **{
            model: {
                'trainer': train_equipment_keypoint_detector if kp_type in ('riser', 'transformer', 'street_light', 'secondary_drip_loop') else train_attachment_keypoint_detector,
                'kwargs': {
                    ('equipment_type' if kp_type in ('riser', 'transformer', 'street_light', 'secondary_drip_loop') else 'attachment_type'): kp_type,
                    'train_dir': str(DATASET_DIRS[model]),
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr,
                    'resume': args.resume,
                    'device': args.device if args.device != 'auto' else None,
                }
            }
            for model, kp_type in KEYPOINT_MODEL_TO_TYPE.items()
        },
    }

    model_config = model_configs[args.model]
    trainer = model_config['trainer']
    kwargs = {k: v for k, v in model_config['kwargs'].items() if v is not None}

    # Show what will be trained
    logger.info(f"Model: {args.model}")
    logger.info(f"Training function: {trainer.__name__}")
    logger.info(f"Hyperparameters: {kwargs}")

    if args.dry_run:
        logger.info("DRY RUN: Would train with above configuration")
        return

    if args.clear_cache and args.model in (
        'pole_detection', 'ruler_detection', 'ruler_marking_detection',
        'pole_top_detection', 'equipment_detection', 'attachment_detection',
    ):
        clear_yolo_disk_cache_from_other_datasets(args.model)

    try:
        # Call the appropriate training function
        logger.info(f"🚀 Starting training for {args.model}...")

        # Call trainer (device passed via kwargs from model config)
        trainer(**kwargs)

        logger.info(f"✅ Training complete for {args.model}!")
        print(f"\n✅ Training complete! Model weights saved to {RUNS_DIR}/{args.model}/weights/")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
