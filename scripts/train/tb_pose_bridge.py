#!/usr/bin/env python3
"""Live-bridge a YOLO run's results.csv (which carries the pose (P) metrics) into a sibling
TensorBoard event file, so an already-running train (whose in-process writer logs box-only)
still surfaces pose mAP / pose loss. Writes to <run>/tensorboard/pose_bridge/ — TB --logdir
<run> merges it. Usage: python scripts/train/tb_pose_bridge.py runs/unified_pole_honest
"""
import csv
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from torch.utils.tensorboard import SummaryWriter
from src.training_utils import _log_yolo_row_to_tensorboard

run = Path(sys.argv[1])
csv_path = run / "results.csv"
writer = SummaryWriter(str(run / "tensorboard" / "pose_bridge"))
lock = threading.Lock()
seen = set()
stale = 0
print(f"[pose-bridge] watching {csv_path}", flush=True)
for _ in range(600):                      # ~5h cap at 30s poll
    new = 0
    if csv_path.exists():
        with open(csv_path) as f:
            for raw in csv.DictReader(f):
                row = {k.strip(): v for k, v in raw.items()}
                e = row.get("epoch")
                if e and e not in seen:
                    _log_yolo_row_to_tensorboard(row, writer, lock)
                    seen.add(e)
                    new += 1
    if new:
        stale = 0
        print(f"[pose-bridge] +{new} epochs (last={max(seen, key=lambda x: float(x))})", flush=True)
    else:
        stale += 1
    if seen and max(float(x) for x in seen) >= 99:        # finished 100 epochs
        break
    if stale >= 20 and seen:                              # ~10 min no new epoch -> early-stopped/done
        break
    time.sleep(30)
writer.flush()
writer.close()
print(f"[pose-bridge] done ({len(seen)} epochs bridged)", flush=True)
