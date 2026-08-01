#!/usr/bin/env python3
"""Dataset QA: render photo + GT-label overlay to sanity-check the label store.

No model — this verifies the LABELS (keypoints, bboxes, hw/ct/arm tokens, ruler anchors)
are correctly placed on the image, catching label-generation bugs before retraining.

Usage:
  python scripts/eval/qa_render_labels.py --classes davit,arm4plus,arm3 --n 12 --out viz/qa_mined
  python scripts/eval/qa_render_labels.py --pids <pid1>,<pid2> --out viz/qa
  python scripts/eval/qa_render_labels.py --jobs MNLC006 --n 8 --out viz/qa
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("USE_PHOTO_ID_LAYOUT", "1")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
import src.photo_id_layout as PIL_LAYOUT  # noqa: E402

PHOTOS = REPO / "data" / "Photos"
DISP_H = 1400  # render height


def _font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def render(pid: str, rec: dict, out: Path):
    p = PHOTOS / f"{pid}.jpg"
    if not p.exists():
        return False
    im = Image.open(p).convert("RGB")
    W, H = im.size
    scale = DISP_H / H
    im = im.resize((round(W * scale), DISP_H))
    W, H = im.size
    d = ImageDraw.Draw(im)
    f = _font(20); fs = _font(15)

    def X(px):  # percent -> display pixels
        return px / 100.0 * W

    def Y(py):
        return py / 100.0 * H

    # ruler anchors (ticks) — cyan dots
    for a in rec.get("anchors", []):
        ft, x, y = a
        d.ellipse([X(x) - 5, Y(y) - 5, X(x) + 5, Y(y) + 5], outline=(0, 220, 220), width=2)
        d.text((X(x) + 7, Y(y) - 8), f"{ft}ft", fill=(0, 220, 220), font=fs)

    # pole top — magenta
    pt = rec.get("pole_top")
    if pt:
        d.ellipse([X(pt[0]) - 6, Y(pt[1]) - 6, X(pt[0]) + 6, Y(pt[1]) + 6], outline=(255, 0, 255), width=3)
        d.text((X(pt[0]) + 8, Y(pt[1])), "pole_top", fill=(255, 0, 255), font=fs)

    # attachment bboxes (yellow) keyed by prefix
    bb = {b["name"][:-5]: b["coords"] for b in rec.get("bboxes", []) if b["name"].endswith("_bbox")}
    hw = rec.get("hw", {}); ct = rec.get("ct", {}); arm = rec.get("arm", {})
    for m in rec.get("attachments", []):
        name = m["name"]
        cx, cy = X(m["x"]), Y(m["y"])
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(50, 220, 50))
        coords = bb.get(name)
        if coords and len(coords) == 4:
            l, r, t, b = coords
            d.rectangle([X(l), Y(t), X(r), Y(b)], outline=(255, 230, 0), width=2)
        tag = name
        extra = []
        if name in hw: extra.append(hw[name])
        if name in ct: extra.append(ct[name])
        if name in arm: extra.append(f"K={arm[name]}")
        if extra: tag += " [" + ",".join(extra) + "]"
        d.text((cx + 8, cy - 10), tag, fill=(50, 255, 50), font=fs)

    # header
    head = f"{pid[:12]}  job={rec.get('job')}  scid={rec.get('scid')}  has_height={rec.get('has_height')}  n_att={len(rec.get('attachments',[]))}"
    d.rectangle([0, 0, W, 26], fill=(0, 0, 0))
    d.text((4, 3), head, fill=(255, 255, 255), font=f)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", help="comma list to sample from the mining manifest")
    ap.add_argument("--pids", help="comma list of explicit photo_ids")
    ap.add_argument("--jobs", help="comma list of job names to sample from")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="viz/qa_labels")
    args = ap.parse_args()

    PIL_LAYOUT._ensure()
    store = PIL_LAYOUT._LABEL_BY_PID

    pids = []
    if args.pids:
        pids = args.pids.split(",")
    else:
        mani_path = REPO / "data" / "hard_mining" / "manifest.jsonl"
        rows = [json.loads(l) for l in mani_path.read_text().splitlines() if l.strip()] if mani_path.exists() else []
        want_cls = set(args.classes.split(",")) if args.classes else None
        want_jobs = set(args.jobs.split(",")) if args.jobs else None
        seen = set()
        for r in rows:
            if r["photo_id"] in seen or r["photo_id"] not in store:
                continue
            if want_cls and not (set(r["classes"]) & want_cls):
                continue
            if want_jobs and r["job"] not in want_jobs:
                continue
            seen.add(r["photo_id"])
            pids.append(r["photo_id"])
            if len(pids) >= args.n:
                break

    out_dir = REPO / args.out
    n = 0
    for pid in pids:
        rec = store.get(pid)
        if not rec:
            print(f"  {pid[:12]}: no label")
            continue
        if render(pid, rec, out_dir / f"{pid[:12]}_{rec.get('job','?').replace(' ','_')}.png"):
            n += 1
    print(f"rendered {n} -> {out_dir}")


if __name__ == "__main__":
    main()
