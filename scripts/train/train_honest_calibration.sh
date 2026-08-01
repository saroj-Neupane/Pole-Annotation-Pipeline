#!/usr/bin/env bash
# Honest calibration-stack retrain on the honest split: pole_detection + ruler_detection (YOLO,
# ultralytics fitness-selected best.pt) + pole_top_detection + ruler_marking_detection (HRNet,
# route through train_model -> best.pth[val-loss] + best_pck.pth[PCK@2"]). Production weights backed
# up to *_preHonest. Serial (one GPU), continue-on-error. HRNet needs device='cuda' (not '0').
cd /home/saroj/Desktop/Python_Projects/Pole_Annotation
export PYTHONPATH=. USE_PHOTO_ID_LAYOUT=1
LOG=data/hard_mining/feedback
ts() { date '+%F %T'; }
echo "=== HONEST CALIBRATION RETRAIN START $(ts) ==="

for r in pole_detection ruler_detection pole_top_detection ruler_marking_detection; do
  if [ -d "runs/$r" ] && [ ! -d "runs/${r}_preHonest" ]; then
    mv "runs/$r" "runs/${r}_preHonest" && echo "backed up runs/$r -> runs/${r}_preHonest"
  fi
done

run() {  # $1=trainer fn  $2=log-name
  echo ">> $2 $(ts)"
  python -u -c "from src.training_utils import $1; $1(device='cuda')" \
    > "$LOG/train_${2}_honest.log" 2>&1 && echo "  $2 done $(ts)" || echo "  $2 FAILED $(ts)"
}

run train_pole_detector          pole_detection
run train_ruler_detector         ruler_detection
run train_pole_top_detector      pole_top_detection
run train_ruler_marking_detector ruler_marking_detection

# --- w3 midspan strip (dataset already w3=width_expand 3.0) + sharp sigma 13.6 -> runs/midspan_wire_strip_w3_honest ---
# (production runs/midspan_wire_strip_detection_w3sharp kept untouched for comparison). Saves
# best.pth[val-loss] + best_f1.pth[deployed find_peaks F1] — use best_f1 / e2e, NOT val-loss.
echo ">> midspan_wire_strip_w3_honest $(ts)"
python -u -c "from src.training_utils import train_midspan_wire_strip_detector; train_midspan_wire_strip_detector(train_dir='datasets/midspan_wire_strip_detection', checkpoint_dir='runs/midspan_wire_strip_w3_honest', sigma_y=13.6, device='cuda')" \
  > "$LOG/train_strip_w3_honest.log" 2>&1 && echo "  strip_w3 done $(ts)" || echo "  strip_w3 FAILED $(ts)"

echo "=== HONEST CALIBRATION + STRIP RETRAIN DONE $(ts) ==="
