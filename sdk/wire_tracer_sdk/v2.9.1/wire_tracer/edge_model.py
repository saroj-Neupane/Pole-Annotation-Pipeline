"""
Learned per-edge cost — pure-numpy port of src/wire_tracing_match.NumpyEdgeCostModel.

This is the V2 matcher's core change. The hand-tuned geometric edge cost
(w_y·dy + w_x·dx + class + deadend) is replaced by a small MLP evaluated on the shared
21-feature edge vector (see constants.EDGE_FEATURE_NAMES / matcher.compute_edge_features):

    cost = 1 - sigmoid(MLP(standardized features[cols]))

The model is frozen from a trained sklearn MLPClassifier into plain numpy arrays
(weights/edge_matcher_unified_v2.json), so it loads with NO sklearn/torch — the SDK stays
numpy/onnxruntime/opencv/Pillow only. The A<->B couplings (tier / chain / class) stay additive
on top of this base cost (matcher.py); only the geometric+intrinsic term is replaced.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NumpyEdgeCostModel:
    """cost = 1 - sigmoid(MLP(standardized features[cols])). ReLU hidden layers (sklearn default)."""

    def __init__(self, Ws, bs, mean, std, cols, feature_names=None):
        self.Ws = [np.asarray(w, np.float64) for w in Ws]
        self.bs = [np.asarray(b, np.float64) for b in bs]
        self.mean = np.asarray(mean, np.float64)
        self.std = np.asarray(std, np.float64)
        self.cols = list(cols)
        self.feature_names = feature_names

    def edge_costs(self, feats: np.ndarray) -> np.ndarray:
        """feats: (N, F) full edge-feature rows. Returns (N,) cost in [0, 1]."""
        a = (np.asarray(feats, np.float64)[:, self.cols] - self.mean) / self.std
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            a = np.maximum(0.0, a @ W + b)        # ReLU hidden layers
        z = a @ self.Ws[-1] + self.bs[-1]         # logit (binary output)
        p = 1.0 / (1.0 + np.exp(-z))
        return (1.0 - p).reshape(-1)

    @classmethod
    def load(cls, path) -> "NumpyEdgeCostModel":
        d = json.loads(Path(path).read_text())
        n = d["n_layers"]
        return cls([d[f"W{i}"] for i in range(n)], [d[f"b{i}"] for i in range(n)],
                   d["mean"], d["std"], d["cols"], d.get("feature_names"))
