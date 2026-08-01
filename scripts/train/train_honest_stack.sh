#!/usr/bin/env bash
# Train the remaining production stack on the HONEST split (equipment + 4 equipment keypoints + w3 strip).
# Production run dirs are backed up to <name>_preHonest first (rollback); honest models take the canonical
# paths. Strip goes to a distinct _honest dir via checkpoint_dir. Serial (one GPU), continue-on-error.
cd /home/saroj/Desktop/Python_Projects/Pole_Annotation
export PYTHONPATH=. USE_PHOTO_ID_LAYOUT=1
LOG=data/hard_mining/feedback
ts() { date '+%F %T'; }

echo "=== HONEST STACK TRAIN START $(ts) ==="

# 1) back up existing production run dirs (idempotent)
for r in equipment_detection riser_keypoint_detection transformer_keypoint_detection \
         street_light_keypoint_detection secondary_drip_loop_keypoint_detection; do
  if [ -d "runs/$r" ] && [ ! -d "runs/${r}_preHonest" ]; then
    mv "runs/$r" "runs/${r}_preHonest" && echo "backed up runs/$r -> runs/${r}_preHonest"
  fi
done

# 2) equipment detector (YOLO) -> runs/equipment_detection
echo ">> equipment_detection $(ts)"
python -u -c "from src.training_utils import train_equipment_detector; train_equipment_detector(device='0')" \
  > $LOG/train_equipment_honest.log 2>&1 && echo "  equipment done $(ts)" || echo "  equipment FAILED $(ts)"

# 3) equipment keypoints (HRNet) -> runs/<type>_keypoint_detection
for kp in riser transformer street_light secondary_drip_loop; do
  echo ">> ${kp}_keypoint $(ts)"
  python -u -c "from src.training_utils import train_equipment_keypoint_detector; train_equipment_keypoint_detector('$kp', device='0')" \
    > $LOG/train_${kp}_kp_honest.log 2>&1 && echo "  ${kp} done $(ts)" || echo "  ${kp} FAILED $(ts)"
done

# 4) midspan wire strip — w3 (dataset already w3) + sharp sigma 13.6 -> runs/midspan_wire_strip_w3_honest
echo ">> midspan_wire_strip_w3_honest $(ts)"
python -u -c "from src.training_utils import train_midspan_wire_strip_detector; train_midspan_wire_strip_detector(train_dir='datasets/midspan_wire_strip_detection', checkpoint_dir='runs/midspan_wire_strip_w3_honest', sigma_y=13.6, device='0')" \
  > $LOG/train_strip_w3_honest.log 2>&1 && echo "  strip done $(ts)" || echo "  strip FAILED $(ts)"

echo "=== HONEST STACK TRAIN DONE $(ts) ==="
