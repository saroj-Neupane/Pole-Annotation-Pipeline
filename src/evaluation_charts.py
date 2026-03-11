"""
Centralized evaluation chart generation. Produces 2x2 evaluation plots matching
the reference style: (a) bbox metrics, (b) valid detections pie, (c) success pie, (d) PCK bar chart.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from .config import CHART_COLORS, CHART_COLORS_LIST, ATTACHMENT_EVALUATION_CONFIG


def generate_detection_evaluation_chart(
    results: Dict[str, Any],
    title: str,
    detection_label: str,
    keypoint_label: str,
    valid_context: str = "Inside BB",
    success_context: str = "≤1\"",
    success_threshold_inch: float = 1.0,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> "Figure":
    """
    Generate 2x2 evaluation chart matching reference style.

    Args:
        results: Dict with keys:
            - detection: map_0_5, map_0_5_to_0_95, mean_iou, valid_rate_percent, detected_count, gt_count
            - keypoint: pck_3_inch, pck_2_inch, pck_1_inch, pck_0_5_inch, mean_error_inches, median_error_inches?, std_error_inches?
        title: Main title (e.g. "Evaluation on Pole Detection and Pole Top Detection (239 images)")
        detection_label: e.g. "Pole Detection" or "Ruler Detection"
        keypoint_label: e.g. "Pole Top Detection" or "Ruler Marking Detection"
        valid_context: e.g. "Pole Top Inside BB" or "All Markings Inside BB"
        success_context: e.g. "Error ≤3\"" or "All Markings ≤1\""
        success_threshold_inch: Threshold for success pie (3, 1, etc.)
        output_path: Where to save PNG
        show: Whether to call plt.show()

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    det = results.get('detection', results)
    kp = results.get('keypoint', {})

    # Normalize keys for different JSON structures
    map_50 = det.get('map_0_5')
    mean_iou = det.get('mean_iou')
    valid_pct = det.get('valid_rate_percent', 0)
    # F1: use stored value, or compute from detected_count / gt_count / pred_count
    f1 = det.get('f1')
    if f1 is None:
        detected = det.get('detected_count', 0)
        gt = det.get('gt_count', 1)
        pred = det.get('pred_count', detected)  # fallback for old JSONs
        precision = detected / max(1, pred)
        recall = detected / max(1, gt)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)

    # Per-keypoint PCK for bar chart (standard metric)
    pck_3 = kp.get('pck_3_inch', 0)
    pck_2 = kp.get('pck_2_inch', 0)
    pck_1 = kp.get('pck_1_inch', 0)
    pck_05 = kp.get('pck_0_5_inch', 0)

    # Per-instance success rate for pie chart (all keypoints within threshold)
    inst_pck_3 = kp.get('instance_pck_3_inch', pck_3)
    inst_pck_1 = kp.get('instance_pck_1_inch', pck_1)
    success_pct = inst_pck_1 if success_threshold_inch == 1.0 else inst_pck_3
    failed_pct = 100.0 - success_pct if success_pct is not None else 0

    mean_err = kp.get('mean_error_inches')
    median_err = kp.get('median_error_inches', mean_err)
    std_err = kp.get('std_error_inches', 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.patch.set_facecolor('white')

    # (a) Bounding Box Metrics - bar chart
    ax = axes[0, 0]
    metrics_names = ['mAP@0.5', 'F1', 'Mean IoU']
    values = [map_50 or 0, f1 or 0, mean_iou or 0]
    colors = [CHART_COLORS['blue'], CHART_COLORS['magenta'], CHART_COLORS['orange']]
    bars = ax.bar(metrics_names, values, color=colors, edgecolor='black', linewidth=1)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f'{v:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.15)  # Headroom for score labels (max 1.0 + padding)
    ax.set_title(f'(a) {detection_label} - Bounding Box Metrics')
    ax.set_facecolor('white')
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.grid(axis='y', alpha=0.3)

    # (b) Valid Detections - pie chart
    ax = axes[0, 1]
    valid_pct = valid_pct or 0
    invalid_pct = 100.0 - valid_pct
    sizes = [valid_pct, invalid_pct]
    colors_pie = [CHART_COLORS['green'], CHART_COLORS['red']]
    labels = [f'Valid\n({valid_context})', 'Invalid']
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie,
                                       startangle=90, textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    ax.set_title(f'(b) Valid {detection_label.split()[0]} Detections')
    ax.set_facecolor('white')

    # (c) Successful Keypoint Detections - pie chart
    ax = axes[1, 0]
    success_pct = success_pct or 0
    failed_pct = 100.0 - success_pct
    sizes = [success_pct, failed_pct]
    colors_pie = [CHART_COLORS['green'], CHART_COLORS['red']]
    labels = [f'Successful\n({success_context})', 'Failed']
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie,
                                       startangle=90, textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    ax.set_title(f'(c) Successful {keypoint_label.split()[0]} Detections (Per-Instance)')
    ax.set_facecolor('white')

    # (d) PCK Metrics - bar chart with stats box
    ax = axes[1, 1]
    pck_names = ['≤3"', '≤2"', '≤1"', '≤0.5"']
    pck_values = [pck_3 or 0, pck_2 or 0, pck_1 or 0, pck_05 or 0]
    pck_colors = [CHART_COLORS['blue'], CHART_COLORS['magenta'], CHART_COLORS['green'], CHART_COLORS['orange']]
    bars = ax.bar(pck_names, pck_values, color=pck_colors, edgecolor='black', linewidth=1)
    for bar, v in zip(bars, pck_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('PCK (%)')
    ax.set_ylim(0, 115)  # Headroom for score labels (max 100% + padding)
    ax.set_title(f'(d) {keypoint_label} - PCK (vertical, Per-Keypoint)')
    ax.set_facecolor('white')
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.grid(axis='y', alpha=0.3)

    # Stats box (vertical error in inches)
    mean_str = f'{mean_err:.3f}"' if mean_err is not None else 'N/A'
    median_str = f'{median_err:.3f}"' if median_err is not None else 'N/A'
    std_str = f'{std_err:.3f}"' if std_err is not None else 'N/A'
    stats_text = f"Vertical error:\nMean: {mean_str}\nMedian: {median_str}\nStd: {std_str}"
    props = dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1)
    ax.text(0.98, 0.15, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            horizontalalignment='right', bbox=props)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Chart saved to: {output_path}")

    if show:
        plt.show()

    return fig


def results_to_chart_input(calibration_results: Dict, dataset_name: str) -> Dict[str, Dict]:
    """
    Convert calibration run_full_evaluation output to chart input format.

    For pole_detection: detection=pole_detection, keypoint=pole_top_detection
    For ruler_detection: detection=ruler_detection, keypoint=ruler_marking_detection
    """
    if dataset_name == 'pole_detection':
        det = calibration_results.get('pole_detection', {})
        kp = calibration_results.get('pole_top_detection', {})
        return {
            'detection': det,
            'keypoint': _add_pck_and_stats(kp),
        }
    elif dataset_name == 'ruler_detection':
        det = calibration_results.get('ruler_detection', {})
        kp = calibration_results.get('ruler_marking_detection', {})
        return {
            'detection': det,
            'keypoint': _add_pck_and_stats(kp),
        }
    return {}


def _add_pck_and_stats(kp: Dict) -> Dict:
    """Ensure keypoint dict has pck_3, pck_2, pck_0_5 and error stats."""
    import numpy as np
    out = dict(kp)
    if 'pck_3_inch' not in out:
        out.setdefault('pck_3_inch', out.get('pck_1_inch', 0))
    if 'pck_2_inch' not in out:
        out.setdefault('pck_2_inch', out.get('pck_1_inch', 0))
    if 'pck_0_5_inch' not in out:
        out.setdefault('pck_0_5_inch', 0)
    mean_in = out.get('mean_error_inches')
    out.setdefault('median_error_inches', mean_in)
    out.setdefault('std_error_inches', 0)
    return out


def generate_all_calibration_charts(results_dir: Path) -> None:
    """
    Generate charts for pole_detection and ruler_detection from JSON in results_dir.
    Saves pole_detection.png and ruler_detection.png in results_dir.
    """
    import matplotlib.pyplot as plt

    # Support both naming conventions
    pole_path = results_dir / 'pole_detection.json'
    if not pole_path.exists():
        pole_path = results_dir / 'pole_detection_evaluation_results.json'
    ruler_path = results_dir / 'ruler_detection.json'
    if not ruler_path.exists():
        ruler_path = results_dir / 'ruler_detection_evaluation_results.json'

    if pole_path.exists():
        with open(pole_path) as f:
            pole = json.load(f)
        n = pole.get('images_processed', 0)
        generate_detection_evaluation_chart(
            results_to_chart_input(pole, 'pole_detection'),
            title=f'Evaluation on Pole Detection and Pole Top Detection ({n} images)',
            detection_label='Pole Detection',
            keypoint_label='Pole Top Detection',
            valid_context='Pole Top Inside BB',
            success_context='Error ≤3"',
            success_threshold_inch=3.0,
            output_path=results_dir / 'pole_detection.png',
            show=False,
        )
        plt.close()

    if ruler_path.exists():
        with open(ruler_path) as f:
            ruler = json.load(f)
        n = ruler.get('images_processed', 0)
        generate_detection_evaluation_chart(
            results_to_chart_input(ruler, 'ruler_detection'),
            title=f'Evaluation on Ruler and Ruler Marking Detection ({n} images)',
            detection_label='Ruler Detection',
            keypoint_label='Ruler Marking Detection',
            valid_context='All Markings Inside BB',
            success_context='All Markings ≤1"',
            success_threshold_inch=1.0,
            output_path=results_dir / 'ruler_detection.png',
            show=False,
        )
        plt.close()


def _generate_detection_2x2_chart(
    results: Dict[str, Any],
    title: str,
    output_path: Optional[Path] = None,
) -> "Figure":
    """
    Generate 2x2 detection chart: F1-Confidence, PR curve, confusion matrix, F1 bar.
    Shared by attachment and equipment detection evaluation.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    px = results.get("px")
    f1_curve = results.get("f1_curve")
    prec_values = results.get("prec_values")
    names_ordered = results.get("names_ordered") or []
    confusion_matrix = results.get("confusion_matrix")
    optimal_per_class = results.get("optimal_per_class") or {}

    fig = plt.figure(figsize=(14, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    gs = GridSpec(2, 2, figure=fig, left=0.08, right=0.92, bottom=0.12, top=0.92,
                  hspace=0.32, wspace=0.28)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # (a) F1-Confidence curve
    ax = fig.add_subplot(gs[0, 0])
    if px and f1_curve:
        px_arr = np.array(px)
        f1_arr = np.array(f1_curve)
        for i in range(f1_arr.shape[0]):
            label = names_ordered[i] if i < len(names_ordered) else f"class_{i}"
            ax.plot(px_arr, f1_arr[i], linewidth=1.5, label=label)
        f1_mean = f1_arr.mean(0)
        opt_idx = int(np.argmax(f1_mean))
        ax.plot(
            px_arr,
            f1_mean,
            linewidth=2.5,
            color=CHART_COLORS['blue'],
            label=f"all {f1_mean[opt_idx]:.2f} @ {px_arr[opt_idx]:.2f}",
        )
    ax.set_xlabel('Confidence')
    ax.set_ylabel('F1')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title('(a) F1-Confidence Curve')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_facecolor('white')

    # (b) PR curve
    ax = fig.add_subplot(gs[0, 1])
    if px and prec_values:
        px_arr = np.array(px)  # recall (x-axis for PR)
        prec_arr = np.array(prec_values)
        for i in range(prec_arr.shape[0]):
            label = names_ordered[i] if i < len(names_ordered) else f"class_{i}"
            ax.plot(px_arr, prec_arr[i], linewidth=1.5, label=label)
        ax.plot(px_arr, prec_arr.mean(0), linewidth=2.5, color=CHART_COLORS['blue'], label='all')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title('(b) Precision-Recall Curve')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_facecolor('white')

    # (c) Confusion matrix
    ax = fig.add_subplot(gs[1, 0])
    if confusion_matrix:
        cm = np.array(confusion_matrix)
        n = cm.shape[0]
        labels = names_ordered + ["background"] if n > len(names_ordered) else names_ordered
        labels = labels[:n]
        # Normalize by column (true class)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm / (cm.sum(0, keepdims=True) + 1e-9)
            cm_norm = np.nan_to_num(cm_norm)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        for i in range(min(cm_norm.shape[0], 10)):
            for j in range(min(cm_norm.shape[1], 10)):
                val = cm_norm[i, j]
                if val > 0.005:
                    ax.text(j, i, f'{val:.2f}' if val < 1 else f'{int(cm[i, j])}',
                            ha='center', va='center', fontsize=8,
                            color='white' if val > 0.45 else 'black')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        plt.colorbar(im, cax=cax)
    else:
        ax.text(0.5, 0.5, 'No confusion matrix', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('(c) Confusion Matrix (at optimal threshold)')
    ax.set_facecolor('white')

    # (d) F1 score bar chart per class
    ax = fig.add_subplot(gs[1, 1])
    if optimal_per_class:
        classes = list(optimal_per_class.keys())
        f1_vals = [optimal_per_class[c].get('f1', 0) for c in classes]
        colors = (CHART_COLORS_LIST + [CHART_COLORS['blue'], CHART_COLORS['orange']])[:len(classes)]
        bars = ax.bar(classes, f1_vals, color=colors, edgecolor='black', linewidth=1)
        for bar, v in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1')
    ax.set_ylim(0, 1.15)
    ax.set_title('(d) F1 Score (per class at optimal threshold)')
    ax.set_facecolor('white')
    ax.tick_params(axis='x', labelrotation=45)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.grid(axis='y', alpha=0.3)
    # Span full width to align with PR curve above (same column)
    if optimal_per_class:
        n = len(optimal_per_class)
        ax.set_xlim(-0.5, n - 0.5)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Chart saved to: {output_path}")

    return fig


def generate_attachment_detection_chart(
    results: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> "Figure":
    """Generate 2x2 chart for attachment_detection (comm + down_guy)."""
    split = results.get("evaluation_split", "val")
    title = f"Attachment Detection Evaluation ({split})"
    return _generate_detection_2x2_chart(results, title, output_path)


def generate_equipment_detection_chart(
    results: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> "Figure":
    """Generate 2x2 chart for equipment_detection (riser, transformer, street_light, secondary_drip_loop)."""
    split = results.get("evaluation_split", "val")
    title = f"Equipment Detection Evaluation ({split})"
    return _generate_detection_2x2_chart(results, title, output_path)


def _class_name_to_title(name: str) -> str:
    """Convert class_name to title case (e.g. down_guy -> Down Guy)."""
    return ' '.join(w.capitalize() for w in name.split('_'))


def generate_all_attachment_charts(results_dir: Path) -> None:
    """Generate charts for attachment_detection and all E2E per-class (comm, down_guy, primary, etc.)."""
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)

    # Detection-only: attachment_detection.json -> attachment_detection.png (2x2 chart)
    attach_path = results_dir / 'attachment_detection.json'
    if attach_path.exists():
        with open(attach_path) as f:
            attach = json.load(f)
        generate_attachment_detection_chart(attach, output_path=results_dir / 'attachment_detection.png')
        plt.close()

    # E2E per-class: all entries in ATTACHMENT_EVALUATION_CONFIG
    for name, cfg in ATTACHMENT_EVALUATION_CONFIG.items():
        class_title = _class_name_to_title(cfg['class_name'])
        det_label = f"{class_title} Detection"
        kp_label = f"{class_title} Keypoint"
        path = results_dir / f'{name}.json'
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            n = d.get('gt_instance_count', 0)
            generate_detection_evaluation_chart(
                {'detection': d.get('detection', {}), 'keypoint': d.get('keypoint', {})},
                title=f'Evaluation on {det_label} and {kp_label} ({n} GT instances)',
                detection_label=det_label,
                keypoint_label=kp_label,
                valid_context='All KPs in BB',
                success_context='All KPs ≤3"',
                success_threshold_inch=3.0,
                output_path=results_dir / f'{name}.png',
                show=False,
            )
            plt.close()


def generate_all_equipment_charts(results_dir: Path) -> None:
    """Generate charts for equipment_detection, streetlight, transformer, riser, secondary_drip_loop."""
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)

    # Detection-only: equipment_detection.json -> equipment_detection.png (2x2 chart)
    equip_path = results_dir / 'equipment_detection.json'
    if equip_path.exists():
        with open(equip_path) as f:
            equip = json.load(f)
        generate_equipment_detection_chart(equip, output_path=results_dir / 'equipment_detection.png')
        plt.close()

    # E2E per-class
    configs = [
        ('streetlight_detection', 'Streetlight Detection', 'Streetlight Keypoint'),
        ('secondary_drip_loop_detection', 'Secondary Drip Loop Detection', 'Secondary Drip Loop Keypoint'),
        ('transformer_detection', 'Transformer Detection', 'Transformer Keypoint'),
        ('riser_detection', 'Riser Detection', 'Riser Keypoint'),
    ]
    for name, det_label, kp_label in configs:
        path = results_dir / f'{name}.json'
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            n = d.get('gt_instance_count', 0)
            generate_detection_evaluation_chart(
                {'detection': d.get('detection', {}), 'keypoint': d.get('keypoint', {})},
                title=f'Evaluation on {det_label} and {kp_label} ({n} GT instances)',
                detection_label=det_label,
                keypoint_label=kp_label,
                valid_context='All KPs in BB',
                success_context='All KPs ≤3"',
                success_threshold_inch=3.0,
                output_path=results_dir / f'{name}.png',
                show=False,
            )
            plt.close()
