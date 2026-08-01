"""Per-photo ``percentY -> height`` model from Katapult ruler calibration.

Vendored from ``sdk/wire_tracer_sdk/v3/wire_tracer/ruler_height_model.py`` (the
canonical SDK copy); kept byte-compatible in behaviour so feet computed here match
the desktop tracer and Katapult's own ``_measured_height``.

Katapult derives a wire's height from its pixel position against the photo's
5-point ruler calibration (the ``2.5 / 6.5 / 10.5 / 14.5 / 16.5 ft`` anchors).
That calibration is a camera projection, so the true ``percentY -> height`` curve
is **projective** (a rational ``inches = (a + b*x)/(1 + c*x)``), not a line —
reverse-engineering it from the anchors reproduces Katapult's own height to within
~half an inch across a whole job, where a straight line is off by feet.

In this training repo the 5 ruler anchors are persisted per photo in
``data/data_{pole,midspan}/Labels/<stem>_location.txt`` (the ``2.5,…``..``16.5,…``
lines, ``height_ft, percentX, percentY``); :func:`src.wire_tracing_e2e.ruler_fit_for_photo`
parses them and fits through here. With no resolvable scale the fit returns
``None`` and the caller must fall back to a percent band.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# percentY is 0-100; fit the projective model on py/100 so it stays well-scaled.
_PY_SCALE = 100.0
# A projective fit needs >= this many distinct anchors (3 frees its 3 dof; 4+ is
# the real 5-anchor ruler and over-determines it, which is what we want).
_MIN_PROJECTIVE_ANCHORS = 3


def fit_height_line(
    points: Sequence[Tuple[Optional[float], Optional[float]]],
) -> Optional[Tuple[float, float]]:
    """Least-squares ``(slope, intercept)`` for ``height_in = slope*percentY + b``.

    ``points`` are ``(percent_y, height_in)`` pairs; entries with a missing
    coordinate are ignored. Returns ``None`` when fewer than two points have a
    height at two distinct ``percent_y`` values (no resolvable scale).
    """
    xs: List[float] = []
    ys: List[float] = []
    for py, h in points:
        if py is None or h is None:
            continue
        xs.append(float(py))
        ys.append(float(h))
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var = sum((x - mean_x) ** 2 for x in xs)
    if var <= 1e-9:
        return None  # all markers at the same height-line position
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    slope = cov / var
    intercept = mean_y - slope * mean_x
    return slope, intercept


@dataclass(frozen=True)
class HeightFit:
    """A fitted ``percentY -> height(inches)`` model for one photo.

    ``kind`` selects the form (``"projective"`` from ruler calibration, exact;
    ``"linear"`` from existing marker heights, a fallback). ``coef`` is its
    coefficients:

    * projective — ``(a, b, c)`` with ``inches = (a + b*x) / (1 + c*x)`` over the
      scaled input ``x = percentY / 100`` (the camera homography of the ruler);
    * linear — ``(slope, intercept)`` with ``inches = slope*percentY + intercept``.
    """

    kind: str
    coef: Tuple[float, ...]


def _fit_projective_inches(
    anchors: Sequence[Tuple[Optional[float], Optional[float]]],
) -> Optional[HeightFit]:
    """Fit ``inches = (a + b*x)/(1 + c*x)`` (x = percentY/100) over ruler anchors.

    ``anchors`` are ``(percent_y, inches)`` pairs. Linearizes to
    ``inches = a + b*x - c*(x*inches)`` and least-squares solves for ``(a, b, c)``;
    needs :data:`_MIN_PROJECTIVE_ANCHORS` distinct ``percent_y`` values. Returns
    ``None`` when under-determined or the solve is degenerate.
    """
    xs: List[float] = []
    ys: List[float] = []
    for py, h in anchors:
        if py is None or h is None:
            continue
        try:
            xs.append(float(py) / _PY_SCALE)
            ys.append(float(h))
        except (TypeError, ValueError):
            continue
    if len({round(x, 6) for x in xs}) < _MIN_PROJECTIVE_ANCHORS:
        return None
    import numpy as np

    x = np.asarray(xs, dtype=float)
    h = np.asarray(ys, dtype=float)
    a_mat = np.column_stack([np.ones_like(x), x, -x * h])
    try:
        coef, *_ = np.linalg.lstsq(a_mat, h, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c = (float(coef[0]), float(coef[1]), float(coef[2]))
    if not all(np.isfinite(v) for v in (a, b, c)):
        return None
    return HeightFit(kind="projective", coef=(a, b, c))


def fit_photo_height(
    anchors: Sequence[Tuple[Optional[float], Optional[float]]],
    markers: Sequence[Tuple[Optional[float], Optional[float]]] = (),
) -> Optional[HeightFit]:
    """Best ``percentY -> inches`` model for one photo, anchors preferred.

    Tries, in order: the **projective** ruler model from calibration ``anchors``
    (exact); a straight line through the anchors (2 of them); a straight line
    through existing ``markers`` that already carry a height. Returns ``None`` when
    nothing yields a vertical scale. ``anchors``/``markers`` are
    ``(percent_y, inches)`` pairs.
    """
    proj = _fit_projective_inches(anchors)
    if proj is not None:
        return proj
    line = fit_height_line(anchors)
    if line is not None:
        return HeightFit(kind="linear", coef=line)
    line = fit_height_line(markers)
    if line is not None:
        return HeightFit(kind="linear", coef=line)
    return None


def _eval_inches(fit: Optional[HeightFit], percent_y: float) -> Optional[float]:
    """Raw (unrounded) inches at ``percent_y``, or None (degenerate / <= 0).

    Returns ``None`` for a missing fit, a non-physical (<= 0) height, or a
    projective evaluation past its vertical asymptote (denominator <= 0).
    """
    if fit is None:
        return None
    if fit.kind == "projective":
        a, b, c = fit.coef
        x = float(percent_y) / _PY_SCALE
        denom = 1.0 + c * x
        if denom <= 1e-9:
            return None
        val = (a + b * x) / denom
    else:
        slope, intercept = fit.coef
        val = slope * float(percent_y) + intercept
    if val <= 0:
        return None
    return float(val)


def height_at(fit: Optional[HeightFit], percent_y: float) -> Optional[int]:
    """Height (whole inches) at ``percent_y`` for a :class:`HeightFit`, or None."""
    val = _eval_inches(fit, percent_y)
    return int(round(val)) if val is not None else None


def height_ft_at(fit: Optional[HeightFit], percent_y: float) -> Optional[float]:
    """Height in **feet** (unrounded) at ``percent_y``, or None.

    Works in feet with sub-inch precision (unlike :func:`height_at`, no rounding) —
    what the chord-sag gate and the inch-based pole dedup want.
    """
    val = _eval_inches(fit, percent_y)
    return val / 12.0 if val is not None else None


def height_in_at(fit: Optional[HeightFit], percent_y: float) -> Optional[float]:
    """Height in **inches** (unrounded) at ``percent_y``, or None."""
    return _eval_inches(fit, percent_y)


def percent_y_at_height(fit: Optional[HeightFit], inches: float) -> Optional[float]:
    """Inverse model: the ``percent_y`` (0-100) where the fit reads ``inches``, or None.

    ``inches=0`` gives the projected GROUND LINE. Projective form h=(a+b·x)/(1+c·x)
    inverts to x=(h−a)/(b−h·c); linear inverts directly. Returns None for a missing
    fit or a degenerate denominator.
    """
    if fit is None:
        return None
    h = float(inches)
    if fit.kind == "projective":
        a, b, c = fit.coef
        denom = b - h * c
        if abs(denom) <= 1e-9:
            return None
        return (h - a) / denom * _PY_SCALE
    slope, intercept = fit.coef
    if abs(slope) <= 1e-12:
        return None
    return (h - intercept) / slope


__all__ = [
    "HeightFit",
    "fit_height_line",
    "fit_photo_height",
    "height_at",
    "height_ft_at",
    "height_in_at",
    "percent_y_at_height",
]
