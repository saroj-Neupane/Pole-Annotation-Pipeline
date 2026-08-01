"""
Pure-numpy replacements for the two scipy calls the wire-tracer needs, so the SDK
runtime stays numpy/onnxruntime/opencv/Pillow only (no scipy).

  * linear_sum_assignment  — rectangular Hungarian (Jonker-Volgenant-free; classic
    O(n^3) Kuhn-Munkres on a padded square matrix). Matches
    scipy.optimize.linear_sum_assignment's RETURNED ASSIGNMENT for the matcher's
    cost matrices (finite costs + a large _BIG sentinel for forbidden cells).
  * find_peaks             — 1-D peak finder with height/distance/prominence gating,
    matching scipy.signal.find_peaks for the parameters the strip extractor uses
    (height: float, distance: int, prominence: float). Parity-checked in
    tools/parity_check.py.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Hungarian assignment (rectangular)
# --------------------------------------------------------------------------- #
def linear_sum_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the rectangular linear sum assignment problem (minimize total cost).

    Returns (row_ind, col_ind) like scipy.optimize.linear_sum_assignment: row_ind is
    sorted ascending, col_ind[k] is the column assigned to row_ind[k]. min(R, C)
    assignments are returned.

    Implementation: O(n^3) Kuhn-Munkres on a square padding of the matrix. The
    matcher's cost matrices use a large finite sentinel (_BIG=1e6) for forbidden
    cells rather than +inf, which this handles directly. (scipy returns the same
    optimal assignment for these well-posed matrices; ties can differ but the
    matcher's costs are effectively non-degenerate.)
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost must be 2-D")
    R, C = cost.shape
    if R == 0 or C == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    transposed = False
    if R > C:
        cost = cost.T
        R, C = C, R
        transposed = True

    # Pad to square (C >= R). Padded rows carry zero cost so they never block.
    n = C
    big = float(cost.max()) + 1.0 if cost.size else 1.0
    m = np.full((n, n), 0.0, dtype=float)
    m[:R, :] = cost

    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)   # p[j] = row matched to column j (1-based; 0 = none)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = m[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    # p[j] = row assigned to column j (1-based). Collect real (row < R) assignments.
    row_for_col = {j - 1: p[j] - 1 for j in range(1, n + 1)}
    pairs = []
    for col, row in row_for_col.items():
        if 0 <= row < R and 0 <= col < C:
            pairs.append((row, col))
    pairs.sort()
    if transposed:
        pairs = sorted((c, r) for r, c in pairs)
    rows = np.array([a for a, _ in pairs], dtype=int)
    cols = np.array([b for _, b in pairs], dtype=int)
    return rows, cols


# --------------------------------------------------------------------------- #
# 1-D peak finder (subset of scipy.signal.find_peaks)
# --------------------------------------------------------------------------- #
def _local_maxima(x: np.ndarray) -> np.ndarray:
    """Indices of local maxima, using the midpoint of flat-top plateaus (scipy rule)."""
    n = x.size
    peaks: List[int] = []
    i = 1
    i_max = n - 1
    while i < i_max:
        if x[i - 1] < x[i]:
            i_ahead = i + 1
            while i_ahead < i_max and x[i_ahead] == x[i]:
                i_ahead += 1
            if x[i_ahead] < x[i]:
                left = i
                right = i_ahead - 1
                peaks.append((left + right) // 2)
                i = i_ahead
                continue
        i += 1
    return np.asarray(peaks, dtype=int)


def _prominences(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Topographic prominence of each peak (scipy.signal.peak_prominences, wlen=None)."""
    proms = np.zeros(peaks.shape[0], dtype=float)
    for k, peak in enumerate(peaks):
        height = x[peak]
        # left: walk until a sample >= height or the array edge; track the min along the way
        i = peak - 1
        left_min = height
        while i >= 0 and x[i] <= height:
            if x[i] < left_min:
                left_min = x[i]
            i -= 1
        # right
        i = peak + 1
        right_min = height
        n = x.size
        while i < n and x[i] <= height:
            if x[i] < right_min:
                right_min = x[i]
            i += 1
        proms[k] = height - max(left_min, right_min)
    return proms


def _select_by_distance(peaks: np.ndarray, priority: np.ndarray, distance: float) -> np.ndarray:
    """scipy's distance filter: greedily keep highest-priority peaks, removing any
    within `distance` of a kept peak. Returns a boolean keep-mask over `peaks`."""
    n = peaks.shape[0]
    keep = np.ones(n, dtype=bool)
    # iterate peaks in order of decreasing priority (scipy iterates ascending then reverses)
    order = np.argsort(priority, kind="stable")  # ascending
    for idx in order[::-1]:
        if not keep[idx]:
            continue
        k = idx - 1
        while k >= 0 and peaks[idx] - peaks[k] < distance:
            keep[k] = False
            k -= 1
        k = idx + 1
        while k < n and peaks[k] - peaks[idx] < distance:
            keep[k] = False
            k += 1
    return keep


def find_peaks(
    x: np.ndarray,
    height: float | None = None,
    distance: float | None = None,
    prominence: float | None = None,
) -> Tuple[np.ndarray, dict]:
    """Pure-numpy subset of scipy.signal.find_peaks.

    Supports the three gates the strip extractor uses (height: scalar min, distance:
    min samples between peaks, prominence: scalar min). Order of application matches
    scipy: local maxima -> height -> distance -> prominence. Returns (peak_indices,
    properties_dict) where properties contains 'prominences' when prominence is given.
    """
    x = np.asarray(x, dtype=float)
    peaks = _local_maxima(x)

    if height is not None and peaks.size:
        peaks = peaks[x[peaks] >= height]

    if distance is not None and peaks.size:
        keep = _select_by_distance(peaks, x[peaks], distance)
        peaks = peaks[keep]

    props: dict = {}
    if prominence is not None and peaks.size:
        proms = _prominences(x, peaks)
        mask = proms >= prominence
        peaks = peaks[mask]
        props["prominences"] = proms[mask]

    return peaks, props
