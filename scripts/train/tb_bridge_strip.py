"""Bridge the strip trainer's text log -> live TensorBoard tfevents (non-intrusive).

The HRNet strip trainer prints per-epoch lines but writes no tfevents. This tails the
log, parses each epoch line, and writes scalars so `tensorboard --logdir <out>` shows
loss + per-tolerance recall live. Exits once training is done and the final epoch is written.
"""
import re
import sys
import time
import subprocess
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/train_strip_fresh.log")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("runs/midspan_wire_strip_detection/tb_live")
OUT.mkdir(parents=True, exist_ok=True)

EP = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*train\s+([\d.]+)\s*\|\s*val\s+([\d.]+)\s*\|\s*recall:\s*"
    r"([\d.]+)%[^|]*\|\s*([\d.]+)%[^|]*\|\s*([\d.]+)%"
)

def train_running():
    return subprocess.run(["pgrep", "-f", "python train.py --model midspan"],
                          capture_output=True).returncode == 0

w = SummaryWriter(str(OUT))
written = set()
while True:
    if LOG.exists():
        for line in LOG.read_text(errors="ignore").splitlines():
            m = EP.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            if ep in written:
                continue
            written.add(ep)
            w.add_scalar("loss/train", float(m.group(2)), ep)
            w.add_scalar("loss/val", float(m.group(3)), ep)
            w.add_scalar("recall/3inch", float(m.group(4)), ep)
            w.add_scalar("recall/2inch", float(m.group(5)), ep)
            w.add_scalar("recall/1inch", float(m.group(6)), ep)
        w.flush()
    if not train_running() and written:
        # one last pass already done above; training is over
        break
    time.sleep(10)
w.close()
print(f"bridge done: wrote {len(written)} epochs to {OUT}")
