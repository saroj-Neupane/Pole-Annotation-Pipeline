#!/usr/bin/env python3
"""Train a midspan wire-strip HRNet on an arbitrary prepared strip dataset dir.

Thin CLI over training_utils.train_midspan_wire_strip_detector so width / config
experiments are reproducible (the deployed model was trained via an ad-hoc heredoc).
Each run writes best_f1.pth (max deployed-extractor F1) + best.pth (min val-loss) under
--checkpoint-dir; never clobbers the deployed runs/midspan_wire_strip_detection unless
you point --checkpoint-dir there.

Example (width-2 strips):
  PYTHONPATH=. python scripts/train/train_strip_variant.py \
      --train-dir datasets/midspan_wire_strip_detection_w2 \
      --checkpoint-dir runs/midspan_wire_strip_detection_w2 \
      --epochs 24 --patience 24
"""
import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from src.training_utils import train_midspan_wire_strip_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True, help="prepared strip dataset dir")
    ap.add_argument("--checkpoint-dir", required=True, help="output run dir (weights/ written here)")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--patience", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--pos-weight", type=float, default=8.0)
    ap.add_argument("--flipud-p", type=float, default=0.0)
    ap.add_argument("--sigma-y", type=float, default=None,
                    help="target-Gaussian vertical std in heatmap rows (default config 27.2; "
                         "smaller = sharper peaks = localization lever)")
    ap.add_argument("--tier-aware", action="store_true",
                    help="parse 'y TAG' labels: supervise non-primary, IGNORE a band around each "
                         "primary in the loss (no target, no penalty) — primary-exclusion direction check")
    ap.add_argument("--ignore-half-rows", type=float, default=None,
                    help="ignore-band half-height in heatmap rows around each primary (default 3*sigma_y)")
    ap.add_argument("--target-mode", choices=["blob", "ridge"], default="blob",
                    help="'blob'=legacy centred 2D Gaussian (deployed); 'ridge'=full-width horizontal "
                         "band at each wire y (2D-line target; pair with full-width readout)")
    ap.add_argument("--continue-lr", type=float, default=None,
                    help="resume from <checkpoint-dir>/weights/last.pth at this FIXED LR (schedulers "
                         "frozen, val-loss early-stop off) to drive a still-climbing deploy/fbeta to its "
                         "true plateau; best_fbeta.pth captures the peak")
    ap.add_argument("--resize-h", type=int, default=None,
                    help="input/heatmap height override (default config 3480); scale --sigma-y "
                         "proportionally (1740 -> 6.8); deploy-eval peak distance auto-scales")
    ap.add_argument("--resize-w", type=int, default=None, help="input/heatmap width override (default 96)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"[train_strip_variant] train_dir={args.train_dir} ckpt={args.checkpoint_dir} "
          f"epochs={args.epochs} pos_weight={args.pos_weight} flipud_p={args.flipud_p} "
          f"sigma_y={args.sigma_y} tier_aware={args.tier_aware} target_mode={args.target_mode}", flush=True)
    train_midspan_wire_strip_detector(
        train_dir=args.train_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        pos_weight=args.pos_weight,
        flipud_p=args.flipud_p,
        sigma_y=args.sigma_y,
        tier_aware=args.tier_aware,
        ignore_half_rows=args.ignore_half_rows,
        target_mode=args.target_mode,
        continue_lr=args.continue_lr,
        resize_height=args.resize_h,
        resize_width=args.resize_w,
        device=args.device,
    )


if __name__ == "__main__":
    main()
