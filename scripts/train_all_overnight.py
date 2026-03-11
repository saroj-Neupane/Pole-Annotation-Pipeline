#!/usr/bin/env python3
"""
Train all models sequentially. Intended for overnight runs.

Skips models that are already trained (best.pt or best.pth exists) by default.
Use --force to retrain all.

Usage:
    python scripts/train_all_overnight.py              # Train all, skip completed
    python scripts/train_all_overnight.py --force      # Retrain all
    python scripts/train_all_overnight.py --skip 3     # Skip first 3 models
    python scripts/train_all_overnight.py --models pole_detection equipment_detection
    nohup python scripts/train_all_overnight.py > train_all.log 2>&1 &
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import KEYPOINT_MODEL_TO_TYPE, RUNS_DIR
from src.training_utils import clear_yolo_disk_cache_from_other_datasets

# YOLO models that use *.npy disk cache (cleared before each to free disk)
YOLO_CACHE_MODELS = (
    'pole_detection',
    'ruler_detection',
    'ruler_marking_detection',
    'pole_top_detection',
    'equipment_detection',
    'attachment_detection',
)

# Order: calibration first, then equipment/attachment, then keypoints
MODELS = [
    *YOLO_CACHE_MODELS,
    *KEYPOINT_MODEL_TO_TYPE.keys(),
]


def is_model_trained(model: str) -> bool:
    """Check if model has completed training (best weights exist)."""
    weights_dir = RUNS_DIR / model / 'weights'
    return (weights_dir / 'best.pt').exists() or (weights_dir / 'best.pth').exists()


def run_training(model: str, args) -> tuple:
    """Run train.py for a model. Returns (model, status, returncode, elapsed_sec)."""
    if not args.force and is_model_trained(model):
        return (model, 'SKIPPED', 0, 0.0)

    if not args.no_clear_cache and model in YOLO_CACHE_MODELS:
        clear_yolo_disk_cache_from_other_datasets(model)

    start = datetime.now()
    cmd = [sys.executable, str(PROJECT_ROOT / 'train.py'), '--model', model, '--device', args.device]
    if args.resume:
        cmd.append('--resume')

    try:
        rc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        elapsed = (datetime.now() - start).total_seconds()
        status = 'OK' if rc.returncode == 0 else 'FAILED'
        return (model, status, rc.returncode, elapsed)
    except Exception as e:
        print(f'  {model}: ERROR - {e}')
        return (model, 'ERROR', -1, 0.0)


def main():
    parser = argparse.ArgumentParser(description='Train all models sequentially')
    parser.add_argument('--skip', type=int, default=0, help='Skip first N models')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Train only these models (default: all)')
    parser.add_argument('--exclude', type=str, nargs='+', default=None,
                        help='Exclude these models')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint for each')
    parser.add_argument('--force', action='store_true', help='Retrain even if already complete')
    parser.add_argument('--stop-on-error', action='store_true',
                        help='Stop on first failure (default: continue)')
    parser.add_argument('--no-clear-cache', action='store_true',
                        help='Skip clearing YOLO cache from other datasets before each model')
    parser.add_argument('--dry-run', action='store_true',
                        help='List models and which would be skipped/trained')
    args = parser.parse_args()

    models = args.models or MODELS[args.skip:]
    if args.exclude:
        models = [m for m in models if m not in set(args.exclude)]

    if args.dry_run:
        print('Models (in order):')
        for i, m in enumerate(models, 1):
            trained = is_model_trained(m)
            action = 'SKIP (already trained)' if trained and not args.force else 'TRAIN'
            print(f'  {i}. {m}: {action}')
        return

    print('=' * 60)
    print(f'Training {len(models)} models sequentially')
    print(f'Started: {datetime.now().isoformat()}')
    if not args.force:
        print('(Skipping already-trained models by default; use --force to retrain)')
    print('=' * 60)

    results = []
    for i, model in enumerate(models, 1):
        print(f'\n[{i}/{len(models)}] {model}...')
        model_result = run_training(model, args)
        results.append(model_result)

        model, status, rc, elapsed = model_result
        if status == 'SKIPPED':
            print(f'  {model}: SKIPPED (already trained)')
        else:
            mins = elapsed / 60 if elapsed else 0
            print(f'  {model}: {status} ({mins:.1f} min)')

        if status not in ('OK', 'SKIPPED') and args.stop_on_error:
            print(f'\nStopping on first failure: {model}')
            break

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for model, status, rc, elapsed in results:
        if status == 'SKIPPED':
            print(f'  {model}: SKIPPED')
        else:
            mins = f'{elapsed/60:.1f} min' if elapsed else '-'
            print(f'  {model}: {status} ({mins})')

    failed = [m for m, s, _, _ in results if s not in ('OK', 'SKIPPED')]
    if failed:
        print(f'\nFailed: {", ".join(failed)}')
        sys.exit(1)
    print('\nAll models trained successfully.')


if __name__ == '__main__':
    main()
