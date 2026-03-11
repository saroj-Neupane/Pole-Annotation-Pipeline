#!/usr/bin/env python3
"""
Enrich model metadata with end-to-end evaluation metrics from results/ directory.

Usage:
    python scripts/enrich_model_metadata.py
"""
import json
from pathlib import Path
from datetime import datetime, timezone


def enrich_metadata():
    """Add evaluation metrics to HRNet model metadata from results/ files."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    models_dir = project_root / 'models' / 'production'

    # Mapping: result file -> model name
    mapping = {
        'equipment/riser_detection.json': 'riser_keypoint_detection',
        'equipment/transformer_detection.json': 'transformer_keypoint_detection',
        'equipment/streetlight_detection.json': 'street_light_keypoint_detection',
        'equipment/secondary_drip_loop_detection.json': 'secondary_drip_loop_keypoint_detection',
        'attachment/comm_detection.json': 'comm_keypoint_detection',
        'attachment/down_guy_detection.json': 'down_guy_keypoint_detection',
    }

    for result_file, model_name in mapping.items():
        result_path = results_dir / result_file
        if not result_path.exists():
            print(f"⚠️  {model_name}: No evaluation results at {result_file}")
            continue

        # Load evaluation results
        with open(result_path) as f:
            eval_data = json.load(f)

        # Find latest version directory
        model_dir = models_dir / model_name
        latest_link = model_dir / 'latest'
        if not latest_link.exists():
            print(f"⚠️  {model_name}: No 'latest' symlink found")
            continue

        # Resolve symlink to get actual version directory
        version_dir = latest_link.resolve()
        metadata_path = version_dir / 'metadata.json'

        if not metadata_path.exists():
            print(f"⚠️  {model_name}: No metadata.json in {version_dir}")
            continue

        # Load existing metadata
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Extract relevant metrics
        keypoint_metrics = eval_data.get('keypoint', {})
        detection_metrics = eval_data.get('detection', {})

        # Add end-to-end metrics section
        metadata['e2e_evaluation'] = {
            'evaluation_date': eval_data.get('evaluation_date'),
            'evaluation_split': eval_data.get('evaluation_split', 'test'),
            'images_evaluated': eval_data.get('images_evaluated'),
            'detection': {
                'precision': detection_metrics.get('precision'),
                'recall': detection_metrics.get('recall'),
                'f1': detection_metrics.get('f1'),
                'map_0_5': detection_metrics.get('map_0_5'),
            },
            'keypoint': {
                'pck_3_inch': keypoint_metrics.get('pck_3_inch'),
                'pck_2_inch': keypoint_metrics.get('pck_2_inch'),
                'pck_1_inch': keypoint_metrics.get('pck_1_inch'),
                'pck_0_5_inch': keypoint_metrics.get('pck_0_5_inch'),
                'mean_error_inches': keypoint_metrics.get('mean_error_inches'),
                'median_error_inches': keypoint_metrics.get('median_error_inches'),
            }
        }

        # Update timestamp
        metadata['enriched_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ {model_name} v{version_dir.name[1:]}: Added E2E metrics (PCK@1\": {keypoint_metrics.get('pck_1_inch', 0):.1f}%)")

    print("\n" + "=" * 70)
    print("Metadata enrichment complete!")


if __name__ == '__main__':
    enrich_metadata()
