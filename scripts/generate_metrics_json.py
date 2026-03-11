#!/usr/bin/env python3
"""
Generate metrics.json files for existing training runs.

This extracts metrics from last.pth checkpoints and saves them as JSON
for easy access without requiring PyTorch.

Usage:
    python scripts/generate_metrics_json.py [--model MODEL_NAME]
"""
import argparse
import json
import torch
from pathlib import Path


def generate_metrics_json(run_dir: Path) -> bool:
    """Generate metrics.json from last.pth checkpoint."""
    last_checkpoint = run_dir / 'weights' / 'last.pth'
    metrics_json_path = run_dir / 'metrics.json'

    if not last_checkpoint.exists():
        print(f"  ❌ No last.pth checkpoint found")
        return False

    try:
        checkpoint = torch.load(last_checkpoint, map_location='cpu')
        if not isinstance(checkpoint, dict):
            print(f"  ❌ Checkpoint is not a dict (raw state_dict)")
            return False

        history = checkpoint.get('history', {})
        if not history:
            print(f"  ⚠️  No training history in checkpoint")
            return False

        # Get final epoch metrics
        epoch = checkpoint.get('epoch', 0)
        final_metrics = {}

        # Get last value from each history list
        if history.get('train_loss'):
            final_metrics['train_loss'] = history['train_loss'][-1]
        if history.get('val_loss'):
            final_metrics['val_loss'] = history['val_loss'][-1]

        # PCK metrics
        for key in ['val_kp_acc_3inch', 'val_kp_acc_2inch', 'val_kp_acc_1inch', 'val_kp_acc_0_5inch']:
            if history.get(key):
                metric_name = f'pck_{key.split("_")[-1]}'
                final_metrics[metric_name] = history[key][-1]

        # Build metrics data
        metrics_data = {
            'epoch': epoch,
            'best_val_loss': checkpoint.get('best_val_loss'),
            'best_val_acc': checkpoint.get('best_val_acc'),
            'final_metrics': final_metrics,
            'training_config': {
                'batch_size': checkpoint.get('batch_size'),
                'learning_rate': checkpoint.get('learning_rate'),
            },
            'history': history,
        }

        # Save to JSON
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        pck_1inch = final_metrics.get('pck_1inch', 0) * 100
        print(f"  ✅ Generated metrics.json (Epoch {epoch}, PCK@1\": {pck_1inch:.1f}%)")
        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate metrics.json files for existing training runs'
    )
    parser.add_argument('--model', help='Specific model name (e.g., riser_keypoint_detection)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    runs_dir = project_root / 'runs'

    # HRNet keypoint models
    hrnet_models = [
        'ruler_marking_detection', 'pole_top_detection',
        'riser_keypoint_detection', 'transformer_keypoint_detection',
        'street_light_keypoint_detection', 'secondary_drip_loop_keypoint_detection',
        'comm_keypoint_detection', 'down_guy_keypoint_detection',
    ]

    if args.model:
        models = [args.model]
    else:
        models = hrnet_models

    print("=" * 70)
    print("GENERATING metrics.json FOR KEYPOINT MODELS")
    print("=" * 70)
    print()

    generated = 0
    for model_name in sorted(models):
        run_dir = runs_dir / model_name
        if not run_dir.exists():
            print(f"{model_name}: Not found")
            continue

        print(f"{model_name}:")
        if generate_metrics_json(run_dir):
            generated += 1

    print()
    print("=" * 70)
    print(f"Generated {generated}/{len(models)} metrics.json files")
    print("=" * 70)


if __name__ == '__main__':
    main()
