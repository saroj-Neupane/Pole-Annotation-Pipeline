#!/usr/bin/env python3
"""
Promote trained models from runs/ to models/production/ with full metadata.

Usage:
    python scripts/promote_model.py --model pole_detection --version 1.0.0 --status production
    python scripts/promote_model.py --model pole_detection --version 1.0.0 --dry-run
"""

import argparse
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any
import subprocess
import yaml
import csv


def get_git_info() -> Dict[str, Any]:
    """Get git information about current state."""
    try:
        commit_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], text=True).strip()

        # Check if working directory is clean
        status = subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
        is_dirty = bool(status)

        # Get remote URL
        try:
            remote_url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], text=True).strip()
        except:
            remote_url = "unknown"

        return {
            'commit_sha': commit_sha,
            'branch': branch,
            'is_dirty': is_dirty,
            'remote_url': remote_url,
        }
    except Exception as e:
        print(f"Warning: Could not get git info: {e}")
        return {
            'commit_sha': 'unknown',
            'branch': 'unknown',
            'is_dirty': False,
            'remote_url': 'unknown',
        }


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b''):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_registry(registry_path: Path) -> Dict[str, Any]:
    """Load model registry."""
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            return json.load(f)
    else:
        return {
            'schema_version': '1.0.0',
            'last_updated': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'models': {},
            'deployment_history': [],
        }


def save_registry(registry_path: Path, registry: Dict[str, Any]) -> None:
    """Save model registry."""
    registry['last_updated'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)


def extract_yolo_metadata(run_dir: Path, model_name: str) -> Dict[str, Any]:
    """Extract metadata from YOLO training run."""
    metadata = {
        'model_type': 'yolo',
        'task': 'object_detection',
    }

    # Parse args.yaml for training config
    args_yaml = run_dir / 'args.yaml'
    if args_yaml.exists():
        with open(args_yaml, 'r') as f:
            args = yaml.safe_load(f)
            if isinstance(args, dict):
                metadata['training'] = {
                    'epochs_completed': int(args.get('epochs', 0)),
                    'batch_size': int(args.get('batch', 16)),
                    'base_lr': float(args.get('lr0', 0.001)),
                    'lrf': float(args.get('lrf', 0.01)),
                    'optimizer': args.get('optimizer', 'auto'),
                    'imgsz': int(args.get('imgsz', 640)) if isinstance(args.get('imgsz'), int) else args.get('imgsz', 640),
                }

                # Extract architecture info
                model_size = args.get('model', '').split('yolo11')[-1].split('.')[0] if 'yolo11' in args.get('model', '') else 'unknown'
                metadata['model_info'] = {
                    'architecture': f"yolo11{model_size}",
                    'framework': 'ultralytics',
                }

    # Parse results.csv for validation metrics
    results_csv = run_dir / 'results.csv'
    if results_csv.exists():
        try:
            with open(results_csv, 'r') as f:
                reader = list(csv.DictReader(f))
                if reader:
                    last_row = reader[-1]
                    metadata['metrics'] = {
                        'validation': {
                            'mAP50': float(last_row.get('metrics/mAP50(B)', 0)),
                            'mAP50_95': float(last_row.get('metrics/mAP50-95(B)', 0)),
                            'precision': float(last_row.get('metrics/precision(B)', 0)),
                            'recall': float(last_row.get('metrics/recall(B)', 0)),
                        }
                    }
        except Exception as e:
            print(f"Warning: Could not parse results.csv: {e}")

    return metadata


def extract_hrnet_metadata(run_dir: Path, model_name: str) -> Dict[str, Any]:
    """Extract metadata from HRNet training run."""
    import torch

    metadata = {
        'model_type': 'hrnet',
        'task': 'keypoint_detection',
    }

    # Try to load metrics.json first (fast, no PyTorch required)
    metrics_json_path = run_dir / 'metrics.json'
    if metrics_json_path.exists():
        try:
            with open(metrics_json_path, 'r') as f:
                metrics_data = json.load(f)

            metadata['training'] = {
                'epochs_completed': metrics_data.get('epoch', 0) + 1,
                'batch_size': metrics_data.get('training_config', {}).get('batch_size', 32),
                'base_lr': metrics_data.get('training_config', {}).get('learning_rate', 3e-4),
            }

            # Extract final metrics
            final = metrics_data.get('final_metrics', {})
            metadata['metrics'] = {
                'validation': {
                    'best_loss': metrics_data.get('best_val_loss'),
                    'pck_3inch': final.get('pck_3_inch'),
                    'pck_2inch': final.get('pck_2_inch'),
                    'pck_1inch': final.get('pck_1_inch'),
                    'pck_0_5inch': final.get('pck_0_5_inch'),
                }
            }
            print(f"  ✓ Loaded metrics from metrics.json")
        except Exception as e:
            print(f"  Warning: Could not load metrics.json: {e}")

    # Fallback: Load checkpoint to get training history (legacy support)
    if not metadata.get('metrics'):
        weights_dir = run_dir / 'weights'
        last_checkpoint = weights_dir / 'last.pth'  # Use last.pth (has history) not best.pth (no history)

        if last_checkpoint.exists():
            try:
                checkpoint = torch.load(last_checkpoint, map_location='cpu')
                if isinstance(checkpoint, dict):
                    # Extract training info
                    epochs = checkpoint.get('epoch', 0) + 1
                    history = checkpoint.get('history', {})

                    metadata['training'] = {
                        'epochs_completed': epochs,
                        'batch_size': checkpoint.get('batch_size', 32),
                        'base_lr': checkpoint.get('learning_rate', 3e-4),
                    }

                    # Extract metrics
                    if history:
                        val_loss = history.get('val_loss', [])
                        metrics = {
                            'validation': {
                                'best_loss': float(checkpoint.get('best_val_loss', min(val_loss) if val_loss else None)),
                            }
                        }

                        # Add PCK metrics if available
                        for key in ['val_kp_acc_3inch', 'val_kp_acc_2inch', 'val_kp_acc_1inch', 'val_kp_acc_0_5inch']:
                            if key in history:
                                acc_list = history[key]
                                if acc_list:
                                    pck_key = f'pck_{key.split("_")[-1]}'
                                    metrics['validation'][pck_key] = float(acc_list[-1])

                        metadata['metrics'] = metrics
                    print(f"  ✓ Loaded metrics from last.pth checkpoint")
            except Exception as e:
                print(f"  Warning: Could not load checkpoint: {e}")

    metadata['model_info'] = {
        'architecture': 'hrnet_custom',
        'framework': 'pytorch',
    }

    return metadata


def promote_model(
    model_name: str,
    version: str,
    status: str,
    notes: str,
    project_root: Path,
    dry_run: bool = False,
) -> bool:
    """
    Promote model from runs/ to models/production/.

    Args:
        model_name: Model name (e.g., 'pole_detection')
        version: Version string (e.g., '1.0.0')
        status: Status ('staging', 'production', 'archived')
        notes: Notes about the promotion
        project_root: Project root directory
        dry_run: If True, don't actually create files

    Returns:
        True if successful, False otherwise
    """
    runs_dir = project_root / 'runs'
    models_dir = project_root / 'models'
    production_dir = models_dir / 'production'
    registry_path = models_dir / 'registry.json'

    # Validate model exists in runs/
    run_model_dir = runs_dir / model_name
    if not run_model_dir.exists():
        print(f"❌ Error: Model directory not found: {run_model_dir}")
        return False

    weights_dir = run_model_dir / 'weights'
    if not weights_dir.exists():
        print(f"❌ Error: Weights directory not found: {weights_dir}")
        return False

    # Determine model type and weight file (aligned with config/train.py)
    yolo_models = ['pole_detection', 'ruler_detection', 'equipment_detection', 'attachment_detection']
    hrnet_models = ['ruler_marking_detection', 'pole_top_detection', 'riser_keypoint_detection',
                    'transformer_keypoint_detection', 'street_light_keypoint_detection',
                    'comm_keypoint_detection', 'down_guy_keypoint_detection',
                    'secondary_drip_loop_keypoint_detection',
                    'primary_keypoint_detection', 'secondary_keypoint_detection',
                    'neutral_keypoint_detection', 'guy_keypoint_detection']

    if model_name in yolo_models:
        model_type = 'yolo'
        weight_file = weights_dir / 'best.pt'
        extension = '.pt'
    elif model_name in hrnet_models:
        model_type = 'hrnet'
        weight_file = weights_dir / 'best.pth'
        extension = '.pth'
    else:
        print(f"❌ Error: Unknown model type for {model_name}")
        return False

    if not weight_file.exists():
        print(f"❌ Error: Weight file not found: {weight_file}")
        return False

    # Create version directory
    version_dir = production_dir / model_name / f'v{version}'
    if not dry_run:
        version_dir.mkdir(parents=True, exist_ok=True)

    # Copy weight file
    model_output_path = version_dir / f'model{extension}'
    if not dry_run:
        shutil.copy2(weight_file, model_output_path)
        print(f"✓ Copied weights: {weight_file} → {model_output_path}")

    # Copy training args
    args_file = run_model_dir / 'args.yaml'
    if args_file.exists() and not dry_run:
        shutil.copy2(args_file, version_dir / 'training_args.yaml')

    # Extract metadata
    if model_type == 'yolo':
        metadata_dict = extract_yolo_metadata(run_model_dir, model_name)
    else:
        metadata_dict = extract_hrnet_metadata(run_model_dir, model_name)

    # Merge with common metadata
    full_metadata = {
        'model_name': model_name,
        'version': version,
        'created_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'git_info': get_git_info(),
        'model_info': metadata_dict.get('model_info', {}),
        'training': metadata_dict.get('training', {}),
        'metrics': metadata_dict.get('metrics', {}),
        'deployment': {
            'status': status,
        },
        'provenance': {
            'source_run_dir': str(run_model_dir),
            'notes': notes,
        },
        'file_info': {
            'model_file': f'model{extension}',
            'model_size_mb': round(weight_file.stat().st_size / (1024 * 1024), 1) if weight_file.exists() else 0,
        }
    }

    # Add file hash if it exists
    if weight_file.exists():
        full_metadata['file_info']['sha256'] = compute_sha256(weight_file)

    # Save metadata
    metadata_path = version_dir / 'metadata.json'
    if not dry_run:
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        print(f"✓ Created metadata: {metadata_path}")

    # Update registry
    registry = load_registry(registry_path)

    # Ensure model exists in registry
    if model_name not in registry['models']:
        registry['models'][model_name] = {
            'type': model_type,
            'versions': {},
            'latest_version': None,
            'production_version': None,
        }

    # Add version to registry
    registry['models'][model_name]['versions'][version] = {
        'created_at': full_metadata['created_at'],
        'status': status,
        'mAP50': full_metadata.get('metrics', {}).get('validation', {}).get('mAP50'),
        'pck_1inch': full_metadata.get('metrics', {}).get('validation', {}).get('pck_1inch'),
        'path': f'models/production/{model_name}/v{version}',
    }

    # Update version pointers
    registry['models'][model_name]['latest_version'] = version
    if status == 'production':
        registry['models'][model_name]['production_version'] = version

    if not dry_run:
        save_registry(registry_path, registry)
        print(f"✓ Updated registry: {registry_path}")

    # Create/update symlinks
    latest_link = production_dir / model_name / 'latest'
    status_link = production_dir / model_name / status

    if not dry_run:
        for link in [latest_link, status_link]:
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                # Replace flat production/ dir (from raw cp) with symlink
                shutil.rmtree(link)
                print(f"  Removed flat directory {link} (replacing with symlink)")

        latest_link.symlink_to(f'v{version}', target_is_directory=True)
        status_link.symlink_to(f'v{version}', target_is_directory=True)
        print(f"✓ Created symlinks:")
        print(f"  {latest_link} → v{version}")
        print(f"  {status_link} → v{version}")

    print(f"\n✅ Successfully promoted {model_name} v{version} to {status}")
    if dry_run:
        print("(Dry-run mode: no files were modified)")

    return True


def bump_version(version: str) -> str:
    """Bump patch: 1.0.0 -> 1.0.1, 1.0.1 -> 1.0.2, etc."""
    parts = version.split('.')
    if len(parts) >= 3:
        try:
            patch = int(parts[2]) + 1
            return f"{parts[0]}.{parts[1]}.{patch}"
        except ValueError:
            pass
    return f"{version}.1" if version else "1.0.1"


def discover_trained_models(
    project_root: Path,
    only_missing: bool = False,
    bump_version_flag: bool = False,
) -> List[Tuple[str, str]]:
    """Discover models in runs/ that have best.pt or best.pth. Returns [(model_name, version)].
    - only_missing: only models not in registry (use 1.0.0)
    - bump_version_flag: all trained models with bumped version from current production"""
    runs_dir = project_root / 'runs'
    registry_path = project_root / 'models' / 'registry.json'
    registry = load_registry(registry_path)

    all_models = [
        'pole_detection', 'ruler_detection', 'equipment_detection', 'attachment_detection',
        'ruler_marking_detection', 'pole_top_detection', 'riser_keypoint_detection',
        'transformer_keypoint_detection', 'street_light_keypoint_detection',
        'comm_keypoint_detection', 'down_guy_keypoint_detection',
        'secondary_drip_loop_keypoint_detection',
        'primary_keypoint_detection', 'secondary_keypoint_detection',
        'neutral_keypoint_detection', 'guy_keypoint_detection',
    ]
    out = []
    for name in sorted(all_models):
        wdir = runs_dir / name / 'weights'
        if not (wdir / 'best.pt').exists() and not (wdir / 'best.pth').exists():
            continue
        if only_missing and name in registry.get('models', {}):
            continue
        if bump_version_flag:
            current = (registry.get('models', {}).get(name) or {}).get('production_version') or '1.0.0'
            version = bump_version(current)
        else:
            version = '1.0.0'
        out.append((name, version))
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Promote trained models from runs/ to models/production/.'
    )
    parser.add_argument('--model', help='Model name (e.g., pole_detection)')
    parser.add_argument('--version', default='1.0.0', help='Version string (default: 1.0.0)')
    parser.add_argument('--all', action='store_true', help='Promote models missing from registry')
    parser.add_argument('--bump', action='store_true', help='Promote all trained models as new versions (bump from current production)')
    parser.add_argument('--status', default='production', choices=['staging', 'production', 'archived'],
                       help='Model status (default: production)')
    parser.add_argument('--notes', default='', help='Notes about this promotion')
    parser.add_argument('--dry-run', action='store_true', help='Preview without modifying files')

    args = parser.parse_args()

    scripts_dir = Path(__file__).parent
    project_root = scripts_dir.parent

    if args.all or args.bump:
        models = discover_trained_models(
            project_root,
            only_missing=args.all and not args.bump,
            bump_version_flag=args.bump,
        )
        if not models:
            print('No trained models found in runs/')
            exit(1)
        print(f'Promoting {len(models)} model(s): {[(m[0], m[1]) for m in models]}')
        failed = []
        for model_name, version in models:
            if not promote_model(model_name, version, args.status, args.notes, project_root, args.dry_run):
                failed.append(model_name)
        exit(1 if failed else 0)

    if not args.model:
        parser.error('--model is required unless --all is used')
    success = promote_model(
        model_name=args.model,
        version=args.version,
        status=args.status,
        notes=args.notes,
        project_root=project_root,
        dry_run=args.dry_run,
    )
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
