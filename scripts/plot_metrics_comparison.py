#!/usr/bin/env python3
"""
Plot bar graphs comparing metrics across versions for each production model.

- YOLO models: mAP@0.5
- ruler_marking_detection: PCK@1"
- All other HRNet keypoint models: PCK@3"

Reads models/registry.json and metadata.json per version.
Saves results/metrics_comparison.png (16 subplots, 4x4 grid).
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "models" / "registry.json"
PRODUCTION_DIR = PROJECT_ROOT / "models" / "production"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "metrics_comparison.png"

# 16 models in display order (4x4 grid)
MODEL_ORDER = [
    "pole_detection",
    "ruler_detection",
    "equipment_detection",
    "attachment_detection",
    "ruler_marking_detection",
    "pole_top_detection",
    "riser_keypoint_detection",
    "transformer_keypoint_detection",
    "street_light_keypoint_detection",
    "secondary_drip_loop_keypoint_detection",
    "comm_keypoint_detection",
    "down_guy_keypoint_detection",
    "guy_keypoint_detection",
    "neutral_keypoint_detection",
    "primary_keypoint_detection",
    "secondary_keypoint_detection",
]

# ruler_marking uses PCK@1"; all other HRNet use PCK@3"
RULER_MARKING_MODEL = "ruler_marking_detection"


def _short_name(model_name: str) -> str:
    """Short display name for model."""
    return model_name.replace("_detection", "").replace("_keypoint", "").replace("_", " ")


def load_registry() -> dict:
    """Load model registry."""
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def load_metadata(model_name: str, version: str) -> Optional[dict]:
    """Load metadata.json for a model version."""
    meta_path = PRODUCTION_DIR / model_name / f"v{version}" / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r") as f:
        return json.load(f)


def get_version_metrics(model_name: str, model_type: str, version: str, ver_data: dict) -> Optional[float]:
    """
    Get the appropriate metric for this model version.
    YOLO: mAP50 from registry.
    ruler_marking: PCK@1" from metadata.
    Other HRNet: PCK@3" from metadata.
    """
    if model_type == "yolo":
        m50 = ver_data.get("mAP50")
        return float(m50) if m50 is not None else None

    # HRNet: load metadata for PCK
    meta = load_metadata(model_name, version)
    if not meta:
        # Fallback to registry pck_1inch if metadata missing
        pck = ver_data.get("pck_1inch")
        return float(pck) * 100 if pck is not None else None

    val = meta.get("metrics", {}).get("validation", {})
    if model_name == RULER_MARKING_MODEL:
        pck = val.get("pck_1inch")
    else:
        pck = val.get("pck_3inch")
    return float(pck) * 100 if pck is not None else None


def collect_model_series(registry: dict) -> List[Tuple[str, str, List[str], List[float]]]:
    """
    For each model, collect (name, metric_label, versions, values).
    Returns list of 16 tuples.
    """
    models = registry.get("models", {})
    series = []

    for model_name in MODEL_ORDER:
        if model_name not in models:
            series.append((_short_name(model_name), "N/A", [], []))
            continue

        model_data = models[model_name]
        model_type = model_data.get("type", "")
        versions_dict = model_data.get("versions", {})

        # Sort versions (e.g. 1.0.0, 1.0.1, 1.0.2)
        def version_key(v):
            try:
                return tuple(int(x) for x in v.split("."))
            except (ValueError, AttributeError):
                return (0, 0, 0)

        sorted_versions = sorted(versions_dict.keys(), key=version_key)
        values = []
        for ver in sorted_versions:
            ver_data = versions_dict[ver]
            val = get_version_metrics(model_name, model_type, ver, ver_data)
            if val is not None:
                values.append(val)
            else:
                values.append(np.nan)  # Missing data

        if model_type == "yolo":
            metric_label = "mAP@0.5"
            # YOLO values are 0-1, convert to % for consistency in display
            values_pct = [v * 100 if not np.isnan(v) else np.nan for v in values]
        elif model_name == RULER_MARKING_MODEL:
            metric_label = "PCK@1\""
            values_pct = values
        else:
            metric_label = "PCK@3\""
            values_pct = values

        # Only include versions that have at least one valid value
        if any(not np.isnan(v) for v in values_pct):
            series.append((_short_name(model_name), metric_label, sorted_versions, values_pct))
        else:
            series.append((_short_name(model_name), metric_label, [], []))

    return series


def plot_single_model(ax, name: str, metric_label: str, versions: list[str], values: list[float]) -> None:
    """Plot one model's version progression as bars."""
    if not versions:
        ax.text(0.5, 0.5, f"{name}\nNo data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_xticks([])
        return

    x = np.arange(len(versions))
    vals = [v if not np.isnan(v) else 0 for v in values]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(versions)))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"v{v}" for v in versions], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(metric_label, fontsize=8)
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}", ha="center", va="bottom", fontsize=6)


def main() -> None:
    registry = load_registry()
    series = collect_model_series(registry)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16), facecolor="white")
    axes = axes.flatten()

    for idx, (name, metric_label, versions, values) in enumerate(series):
        if idx < len(axes):
            plot_single_model(axes[idx], name, metric_label, versions, values)

    plt.suptitle("Production Model Metrics: Version Progress", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
