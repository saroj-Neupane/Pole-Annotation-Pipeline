#!/usr/bin/env python3
"""
Run model evaluations from the command line.

Usage examples:
    python scripts/evaluate_models.py --calibration
    python scripts/evaluate_models.py --attachment
    python scripts/evaluate_models.py --equipment
    python scripts/evaluate_models.py --all
    python scripts/evaluate_models.py --calibration --dataset pole_detection
    python scripts/evaluate_models.py --calibration --no-tta
    python scripts/evaluate_models.py --calibration --plots-only
    python scripts/evaluate_models.py --all --results-dir /tmp/eval_results
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate pole calibration and annotation models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Which evaluations to run
    group = parser.add_argument_group('Evaluation targets')
    group.add_argument('--calibration', action='store_true',
                       help='Run calibration pipeline evaluation (pole & ruler detection)')
    group.add_argument('--attachment', action='store_true',
                       help='Run attachment E2E evaluation (comm, down_guy)')
    group.add_argument('--equipment', action='store_true',
                       help='Run equipment E2E evaluation (streetlight, transformer, riser, etc.)')
    group.add_argument('--all', dest='run_all', action='store_true',
                       help='Run all evaluations (calibration + attachment + equipment)')

    # Calibration-specific options
    calib_group = parser.add_argument_group('Calibration options')
    calib_group.add_argument(
        '--dataset',
        choices=['pole_detection', 'ruler_detection'],
        default=None,
        help='Only evaluate a specific calibration dataset (default: both)',
    )
    calib_group.add_argument('--no-tta', dest='use_tta', action='store_false', default=True,
                              help='Disable test-time augmentation for keypoint models')

    # Chart options
    chart_group = parser.add_argument_group('Chart options')
    chart_group.add_argument('--plots', action='store_true',
                              help='Generate evaluation charts after running evaluations')
    chart_group.add_argument('--plots-only', action='store_true',
                              help='Only generate charts from existing JSON results (skip inference)')

    # Output
    parser.add_argument('--results-dir', type=Path, default=None,
                        help='Override output directory for JSON results and charts')
    parser.add_argument('--device', default=None,
                        help='Torch device (e.g. "cpu", "cuda", "cuda:0"). Auto-detected if not set.')

    return parser.parse_args()


def get_device(device_str):
    import torch
    if device_str:
        return torch.device(device_str)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_calibration(args):
    from src.evaluation_utils import (
        run_full_evaluation,
        generate_evaluation_plots,
        load_calibration_evaluation_models,
    )
    from src.config import RESULTS_CALIBRATION_DIR, EVALUATION_DATASETS_CONFIG

    results_dir = args.results_dir or RESULTS_CALIBRATION_DIR

    if args.plots_only:
        print(f"Generating calibration charts from: {results_dir}")
        generate_evaluation_plots(results_dir)
        return

    datasets = [args.dataset] if args.dataset else list(EVALUATION_DATASETS_CONFIG.keys())
    print(f"Loading calibration models...")
    device = get_device(args.device)
    models = load_calibration_evaluation_models(device)

    for dataset_name in datasets:
        print(f"\n--- Evaluating: {dataset_name} ---")
        run_full_evaluation(
            dataset_name=dataset_name,
            models=models,
            results_dir=results_dir,
            use_tta=args.use_tta,
        )

    if args.plots or args.run_all:
        print("\nGenerating calibration charts...")
        generate_evaluation_plots(results_dir)


def run_attachment(args):
    from src.evaluation_attachment_equipment import run_attachment_evaluation
    from src.evaluation_charts import generate_all_attachment_charts
    from src.config import RESULTS_ATTACHMENT_DIR

    results_dir = args.results_dir or RESULTS_ATTACHMENT_DIR

    if args.plots_only:
        print(f"Generating attachment charts from: {results_dir}")
        generate_all_attachment_charts(results_dir)
        return

    device = get_device(args.device)
    print("\n--- Evaluating: attachment (comm, down_guy) ---")
    run_attachment_evaluation(results_dir=results_dir, device=device)
    print("\nGenerating attachment charts...")
    generate_all_attachment_charts(results_dir)


def run_equipment(args):
    from src.evaluation_attachment_equipment import run_equipment_evaluation
    from src.evaluation_charts import generate_all_equipment_charts
    from src.config import RESULTS_EQUIPMENT_DIR

    results_dir = args.results_dir or RESULTS_EQUIPMENT_DIR

    if args.plots_only:
        print(f"Generating equipment charts from: {results_dir}")
        generate_all_equipment_charts(results_dir)
        return

    device = get_device(args.device)
    print("\n--- Evaluating: equipment (streetlight, transformer, riser, etc.) ---")
    run_equipment_evaluation(results_dir=results_dir, device=device)
    print("\nGenerating equipment charts...")
    generate_all_equipment_charts(results_dir)


def run_combined(args):
    """Run equipment + attachment sharing a single pole-detection pass."""
    from src.evaluation_attachment_equipment import run_combined_evaluation
    from src.evaluation_charts import generate_all_equipment_charts, generate_all_attachment_charts
    from src.config import RESULTS_EQUIPMENT_DIR, RESULTS_ATTACHMENT_DIR

    equip_dir = args.results_dir or RESULTS_EQUIPMENT_DIR
    attach_dir = args.results_dir or RESULTS_ATTACHMENT_DIR

    device = get_device(args.device)
    print("\n--- Evaluating: equipment + attachment (combined pole pass) ---")
    run_combined_evaluation(equip_results_dir=equip_dir, attach_results_dir=attach_dir, device=device)

    print("\nGenerating equipment charts...")
    generate_all_equipment_charts(equip_dir)
    print("\nGenerating attachment charts...")
    generate_all_attachment_charts(attach_dir)


def main():
    args = parse_args()

    nothing_selected = not (args.calibration or args.attachment or args.equipment or args.run_all or args.plots_only)
    if nothing_selected:
        print("No evaluation target specified. Use --calibration, --attachment, --equipment, or --all.")
        print("Run with --help for full usage.")
        sys.exit(1)

    if args.run_all or args.calibration:
        run_calibration(args)

    # Run equipment + attachment together (shared pole pass) when both are needed
    if args.run_all or (args.attachment and args.equipment):
        run_combined(args)
    elif args.attachment:
        run_attachment(args)
    elif args.equipment:
        run_equipment(args)

    # --plots-only with no domain flag: generate all charts
    if args.plots_only and not (args.calibration or args.attachment or args.equipment or args.run_all):
        run_calibration(args)
        run_attachment(args)
        run_equipment(args)

    print("\nDone.")


if __name__ == '__main__':
    main()
