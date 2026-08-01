"""Midspan wire TIER classifier (Experiment 2, EXP-0001 productionization).

Classifies each detected midspan strip peak as bare / multiplex / comm (or 'none' = no
signal, the 4th VETO class that absorbs false peaks) from a PPI-normalized 40"x10" photo
patch centred on the wire crossing, via a fine-tuned resnet18. The predicted tier feeds
the matcher's ``w_mid_tier3_bonus`` term (subtracted from tier3-agreeing midspan<->pole
edges so they beat the dustbin — the rescue mechanism a veto cannot provide).

Winning validated config (fixed balanced harness, 2026-07-08): 4-class 'none'-veto
classifier + protect-bare gates (0, .7, .7) + bonus 0.6 -> e2e 0.5496 -> 0.5615 (+1.2pp;
a FLOOR given incomplete midspan GT). NOT bonus=1.5 — that was the oracle-tier point.

Patch geometry MUST match scripts/diag/probe_tier_separability.py extract() and
scripts/diag/build_tier_cache.py: 40"x10" at the photo's PPI, resized to 256x64, RGB/255.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TIERS = ("bare", "multiplex", "comm")
PATCH_IN_W, PATCH_IN_H = 40.0, 10.0      # inches
PATCH_W, PATCH_H = 256, 64               # classifier input (w, h)

# PRODUCTION 4-class ('none' veto) checkpoint (promoted 2026-07-30, user-approved,
# midspan_tier_classifier v1.0.0 — resolves through the registry symlink).
DEFAULT_TIER_WEIGHTS = "models/production/midspan_tier_classifier/production/model.pth"
# Protect-bare asymmetry: bare needs no confidence, multiplex/comm need >=0.7 —
# a bare->comm false positive on the 62%-dominant bare class costs more chains than
# comm recall gains fix (e2e-optimal op-point is asymmetric).
DEFAULT_TIER_GATES: Tuple[float, float, float] = (0.0, 0.7, 0.7)


def _ppi_for_midspan_photo(photo: str) -> Optional[float]:
    """PPI for the patch size: stored PPI, else ruler-fit local scale at mid-ruler
    (same fallback construction as build_tier_cache / extract_ruler_line_strip)."""
    import numpy as np
    from src.wire_tracing_e2e import (_ruler_anchor_pts_for_photo, ppi_for_photo,
                                      ruler_fit_for_photo)
    from src.ruler_height_model import height_in_at
    ppi = ppi_for_photo(photo, "midspan")
    if ppi:
        return float(ppi)
    pts = _ruler_anchor_pts_for_photo(photo)
    fit = ruler_fit_for_photo(photo, "midspan")
    if len(pts) < 2 or fit is None:
        return None
    py_mid = float(np.mean([p[2] for p in pts]))
    h1, h2 = height_in_at(fit, py_mid - 0.3), height_in_at(fit, py_mid + 0.3)
    if h1 is None or h2 is None or abs(h2 - h1) < 1e-6:
        return None
    import cv2
    img = cv2.imread(photo)
    if img is None:
        return None
    return (img.shape[0] / 100.0) / (abs(h2 - h1) / 0.6)


class MidspanTierClassifier:
    """Attaches ``tier3`` to detected midspan points (in place). tier None = no signal
    (below gate / 'none' veto / patch out of bounds / no PPI)."""

    def __init__(self, weights: str = DEFAULT_TIER_WEIGHTS, device: str = "cpu",
                 gates: Sequence[float] = DEFAULT_TIER_GATES):
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        assert ck["arch"] == "resnet18", f"unexpected arch in {weights}"
        self.n_out = len(ck.get("tiers", TIERS))         # 4 incl 'none' for the veto model
        net = resnet18()
        net.fc = nn.Linear(net.fc.in_features, self.n_out)
        net.load_state_dict(ck["state_dict"])
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.gates = tuple(gates)
        self.weights_path = str(weights)

    def classify_points(self, photo: str, points: List[Dict]) -> None:
        """Set ``tier3`` on every point ({x,y} in photo percent). No-op on empty input."""
        if not points:
            return
        import cv2
        import numpy as np
        import torch
        for p in points:
            p.setdefault("tier3", None)
        ppi = _ppi_for_midspan_photo(photo)
        if not ppi:
            return
        img = cv2.imread(photo)
        if img is None:
            return
        H, W = img.shape[:2]
        half_w, half_h = PATCH_IN_W / 2.0 * ppi, PATCH_IN_H / 2.0 * ppi
        idxs, batch = [], []
        for i, p in enumerate(points):
            x_px, y_px = p["x"] / 100.0 * W, p["y"] / 100.0 * H
            x0, x1 = int(round(x_px - half_w)), int(round(x_px + half_w))
            y0, y1 = int(round(y_px - half_h)), int(round(y_px + half_h))
            if x0 < 0 or y0 < 0 or x1 > W or y1 > H or x1 - x0 < 32 or y1 - y0 < 8:
                continue                                 # border: no tier signal
            patch = cv2.resize(img[y0:y1, x0:x1], (PATCH_W, PATCH_H),
                               interpolation=cv2.INTER_AREA)
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            idxs.append(i)
            batch.append(torch.from_numpy(patch.transpose(2, 0, 1)))
        if not batch:
            return
        with torch.no_grad():
            probs = torch.softmax(self.net(torch.stack(batch).to(self.device)), dim=1)
        probs = probs.cpu().numpy()
        for i, pr in zip(idxs, probs):
            k = int(np.argmax(pr))
            # 4-class veto: argmax 'none' (index 3) -> no tier signal for this peak
            if k < len(TIERS) and pr[k] >= self.gates[k]:
                points[i]["tier3"] = TIERS[k]
