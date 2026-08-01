#!/usr/bin/env bash
# Honest keypoint retrain: the 4 equipment HRNet keypoints on the honest split -> runs/<type>_keypoint_detection
# (production weights already backed up to *_preHonest). Serial (one GPU), continue-on-error.
cd /home/saroj/Desktop/Python_Projects/Pole_Annotation
export PYTHONPATH=. USE_PHOTO_ID_LAYOUT=1
LOG=data/hard_mining/feedback
ts() { date '+%F %T'; }
echo "=== HONEST KEYPOINT RETRAIN START $(ts) ==="
for kp in riser transformer street_light secondary_drip_loop; do
  echo ">> ${kp}_keypoint $(ts)"
  python -u -c "from src.training_utils import train_equipment_keypoint_detector; train_equipment_keypoint_detector('$kp', device='cuda')" \
    > $LOG/train_${kp}_kp_honest.log 2>&1 && echo "  ${kp} done $(ts)" || echo "  ${kp} FAILED $(ts)"
done
echo "=== HONEST KEYPOINT RETRAIN DONE $(ts) ==="
