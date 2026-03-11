#!/usr/bin/env python3
"""
Prepare datasets: calibration, equipment, attachment, and keypoint detection.

By default, freezes val/test from split_manifest.json and adds new samples from data/
to train only (skips samples already in manifest). Use --force-manifest to recreate.

Usage:
    python scripts/prepare_dataset.py                    # all datasets (quiet)
    python scripts/prepare_dataset.py --calibration       # calibration only
    python scripts/prepare_dataset.py --verbose          # detailed output
    python scripts/prepare_dataset.py --force-manifest   # recreate splits from scratch
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASE_DIR_POLE,
    BASE_DIR_MIDSPAN,
    POLE_LABELS_DIR,
    MIDSPAN_LABELS_DIR,
    DATASETS_DIR,
    EQUIPMENT_DATASET_DIR,
    ATTACHMENT_DATASET_DIR,
    DATASET_DIRS,
    KEYPOINT_PREPARE_SPECS,
)
from src.data_utils import (
    prepare_calibration_datasets,
    prepare_equipment_detection_dataset,
    prepare_attachment_detection_dataset,
    prepare_keypoint_detection_dataset,
    prepare_attachment_keypoint_dataset,
    create_split_manifest,
    update_split_manifest,
    load_split_manifest,
)

KEYPOINT_ARG_TO_SPEC = {
    'riser': 'riser_keypoint_dir',
    'transformer': 'transformer_keypoint_dir',
    'street_light': 'street_light_keypoint_dir',
    'secondary_drip_loop': 'secondary_drip_loop_keypoint_dir',
    'comm': 'comm_keypoint_dir',
    'down_guy': 'down_guy_keypoint_dir',
    'primary': 'primary_keypoint_dir',
    'secondary': 'secondary_keypoint_dir',
    'neutral': 'neutral_keypoint_dir',
    'guy': 'guy_keypoint_dir',
}


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets from pole and midspan photos")
    parser.add_argument("--pole-photos", type=Path, default=BASE_DIR_POLE / "Photos")
    parser.add_argument("--pole-labels", type=Path, default=POLE_LABELS_DIR)
    parser.add_argument("--midspan-photos", type=Path, default=BASE_DIR_MIDSPAN / "Photos")
    parser.add_argument("--midspan-labels", type=Path, default=MIDSPAN_LABELS_DIR)
    parser.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR)
    parser.add_argument("--equipment-dataset-dir", type=Path, default=EQUIPMENT_DATASET_DIR)
    parser.add_argument("--attachment-dataset-dir", type=Path, default=ATTACHMENT_DATASET_DIR)
    keypoint_defaults = {spec[1]: DATASET_DIRS[spec[1]] for spec in KEYPOINT_PREPARE_SPECS}
    parser.add_argument("--riser-keypoint-dir", type=Path, default=keypoint_defaults["riser_keypoint_detection"])
    parser.add_argument("--transformer-keypoint-dir", type=Path, default=keypoint_defaults["transformer_keypoint_detection"])
    parser.add_argument("--street-light-keypoint-dir", type=Path, default=keypoint_defaults["street_light_keypoint_detection"])
    parser.add_argument("--secondary-drip-loop-keypoint-dir", type=Path, default=keypoint_defaults["secondary_drip_loop_keypoint_detection"])
    parser.add_argument("--comm-keypoint-dir", type=Path, default=keypoint_defaults["comm_keypoint_detection"])
    parser.add_argument("--down-guy-keypoint-dir", type=Path, default=keypoint_defaults["down_guy_keypoint_detection"])
    parser.add_argument("--primary-keypoint-dir", type=Path, default=keypoint_defaults["primary_keypoint_detection"])
    parser.add_argument("--secondary-keypoint-dir", type=Path, default=keypoint_defaults["secondary_keypoint_detection"])
    parser.add_argument("--neutral-keypoint-dir", type=Path, default=keypoint_defaults["neutral_keypoint_detection"])
    parser.add_argument("--guy-keypoint-dir", type=Path, default=keypoint_defaults["guy_keypoint_detection"])
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--equipment", action="store_true")
    parser.add_argument("--attachment", action="store_true")
    parser.add_argument("--keypoints", action="store_true")
    parser.add_argument("--create-manifest-only", action="store_true")
    parser.add_argument("--force-manifest", action="store_true", help="Recreate manifest from scratch (resets val/test)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed output")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")

    args = parser.parse_args()

    prepare_all = not any([args.calibration, args.equipment, args.attachment, args.keypoints])
    do_calibration = prepare_all or args.calibration
    do_equipment = prepare_all or args.equipment
    do_attachment = prepare_all or args.attachment
    do_keypoints = prepare_all or args.keypoints
    verbose = args.verbose

    # Manifest (required for consistent splits)
    # Default: freeze val/test; add new samples from data/ to train only. Use --force-manifest to recreate.
    if args.create_manifest_only or do_calibration or do_equipment or do_attachment or do_keypoints:
        manifest_path = args.datasets_dir / "split_manifest.json"
        existing_manifest = load_split_manifest(manifest_path)
        if args.create_manifest_only or args.force_manifest or existing_manifest is None:
            manifest = create_split_manifest(
                pole_photos_dir=args.pole_photos,
                pole_labels_dir=args.pole_labels,
                midspan_photos_dir=args.midspan_photos,
                midspan_labels_dir=args.midspan_labels,
                output_path=manifest_path,
            )
            if verbose:
                p = manifest['pole']
                m = manifest['midspan']
                print(f"Manifest: pole {len(p['train'])}/{len(p['val'])}/{len(p['test'])} "
                       f"midspan {len(m['train'])}/{len(m['val'])}/{len(m['test'])}")
            else:
                print("Manifest ✓")
            if args.create_manifest_only:
                return
        else:
            manifest = update_split_manifest(
                pole_photos_dir=args.pole_photos,
                pole_labels_dir=args.pole_labels,
                midspan_photos_dir=args.midspan_photos,
                midspan_labels_dir=args.midspan_labels,
                output_path=manifest_path,
                manifest=existing_manifest,
            )
            if verbose:
                p = manifest['pole']
                m = manifest['midspan']
                print(f"Manifest (updated): pole train {len(p['train'])}, val {len(p['val'])}, test {len(p['test'])} "
                       f"| midspan train {len(m['train'])}, val {len(m['val'])}, test {len(m['test'])}")
            else:
                print("Manifest ✓")

    if not args.pole_photos.exists():
        print(f"❌ Pole photos not found: {args.pole_photos}", file=sys.stderr)
        sys.exit(1)
    if do_calibration and not args.midspan_photos.exists():
        print(f"❌ Midspan photos not found: {args.midspan_photos}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print("Preparing datasets" + (" (all)" if prepare_all else ""))
        print(f"  Pole: {args.pole_photos}")
        print(f"  Midspan: {args.midspan_photos}")

    try:
        if do_calibration:
            prepare_calibration_datasets(
                pole_photos_dir=args.pole_photos,
                pole_labels_dir=args.pole_labels,
                midspan_photos_dir=args.midspan_photos,
                midspan_labels_dir=args.midspan_labels,
                datasets_dir=args.datasets_dir,
                verbose=verbose,
                workers=args.workers,
            )
            if not verbose:
                print("Calibration ✓")

        if do_equipment:
            prepare_equipment_detection_dataset(
                photos_dir=args.pole_photos,
                labels_dir=args.pole_labels,
                dataset_dir=args.equipment_dataset_dir,
                verbose=verbose,
                workers=args.workers,
            )
            if not verbose:
                print("Equipment ✓")

        if do_attachment:
            prepare_attachment_detection_dataset(
                photos_dir=args.pole_photos,
                labels_dir=args.pole_labels,
                dataset_dir=args.attachment_dataset_dir,
                verbose=verbose,
                workers=args.workers,
            )
            if not verbose:
                print("Attachment ✓")

        if do_keypoints:
            for kp_type, _, prep_kind in KEYPOINT_PREPARE_SPECS:
                dataset_dir = getattr(args, KEYPOINT_ARG_TO_SPEC[kp_type])
                if prep_kind == 'equipment':
                    prepare_keypoint_detection_dataset(
                        photos_dir=args.pole_photos,
                        labels_dir=args.pole_labels,
                        eq_type=kp_type,
                        dataset_dir=dataset_dir,
                        verbose=verbose,
                        workers=args.workers,
                    )
                else:
                    prepare_attachment_keypoint_dataset(
                        photos_dir=args.pole_photos,
                        labels_dir=args.pole_labels,
                        att_type=kp_type,
                        dataset_dir=dataset_dir,
                        verbose=verbose,
                        workers=args.workers,
                    )
                if not verbose:
                    print(f"Keypoints ({kp_type}) ✓")

        if not verbose:
            print("Done ✓")

    except Exception as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
