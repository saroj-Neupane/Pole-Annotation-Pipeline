"""
Data processing utilities for dataset preparation.
"""

import re
import numpy as np
import cv2
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Callable, Any, Iterable
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


def _parallel_map(items: Iterable, fn: Callable, workers: int, desc: str = None, verbose: bool = False) -> List[Any]:
    """Process items in parallel with fn(item). Returns list of results."""
    items = list(items)
    if workers <= 1 or len(items) == 0:
        return [fn(x) for x in tqdm(items, desc=desc, disable=not verbose)]
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_idx = {ex.submit(fn, x): i for i, x in enumerate(items)}
        for future in tqdm(as_completed(future_to_idx), total=len(items), desc=desc, disable=not verbose):
            results[future_to_idx[future]] = future.result()
    return results


from .config import (
    DATASETS_DIR,
    FROZEN_MANIFEST_FILENAME,
    SPLIT_MANIFEST_PATH,
    SPLIT_MANIFEST_RANDOM_STATE,
    MIDSPAN_WIRE_STRIP_DETECTION,
    MIDSPAN_WIRE_EXCLUDED_JOB_PREFIXES,
    EQUIPMENT_CLASSES,
    EQUIPMENT_CLASS_NAMES,
    ATTACHMENT_CLASSES,
    ATTACHMENT_CLASS_NAMES,
    UNIFIED_POLE_DETECTION,
    UNIFIED_POLE_DETECTION_CLASSES,
    UNIFIED_POLE_DETECTION_CLASS_NAMES,
    UNIFIED_POLE_DETECTION_NUM_KEYPOINTS,
    UNIFIED_POLE_DETECTION_KEYPOINT_NAMES,
    UNIFIED_POLE_DETECTION_BBOX_HEIGHT_FEET,
    UNIFIED_POLE_DETECTION_BBOX_WIDTH_FEET,
    unified_joint_class,
    RISER_NUM_KEYPOINTS,
    RISER_KEYPOINT_NAMES,
    TRANSFORMER_NUM_KEYPOINTS,
    TRANSFORMER_KEYPOINT_NAMES,
    STREET_LIGHT_NUM_KEYPOINTS,
    STREET_LIGHT_KEYPOINT_NAMES,
    SECONDARY_DRIP_LOOP_NUM_KEYPOINTS,
    SECONDARY_DRIP_LOOP_KEYPOINT_NAMES,
    RISER_BBOX_HEIGHT_FEET,
    RISER_BBOX_WIDTH_FEET,
    KEYPOINT_NAMES,
)
from . import photo_id_layout as _pil


def _loc_path(labels_dir: Path, stem: str):
    """Thin alias to the shared gate ``photo_id_layout.loc_path`` (kept for the prep call sites)."""
    return _pil.loc_path(labels_dir, stem)


def load_frozen_manifest(dataset_dir: Path, strict: bool = False) -> Optional[Dict]:
    """
    Load frozen manifest for a dataset.

    Args:
        dataset_dir: Path to dataset directory
        strict: If True, raise FileNotFoundError when manifest doesn't exist.
                If False, return None when not found.

    Returns:
        Manifest dictionary, or None if not found (when strict=False)
    """
    manifest_path = dataset_dir / FROZEN_MANIFEST_FILENAME
    if not manifest_path.exists():
        if strict:
            raise FileNotFoundError(
                f"Frozen manifest not found: {manifest_path}\n"
                f"Please run: python scripts/freeze_validation_test_sets.py --dataset {dataset_dir.name}"
            )
        return None
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except Exception as e:
        if strict:
            raise
        print(f"❌ Error loading manifest: {e}")
        return None


# -----------------------------------------------------------------------------
# Master split manifest (single source of truth for train/val/test across datasets)
# -----------------------------------------------------------------------------


def load_split_manifest(path: Path = SPLIT_MANIFEST_PATH) -> Optional[Dict]:
    """Load master split manifest. Returns None if not found."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_pole_split_map(manifest: Dict) -> Dict[str, str]:
    """Return {stem: 'train'|'val'|'test'} for pole photos."""
    m = manifest.get("pole", {})
    out = {}
    for split in ["train", "val", "test"]:
        for stem in m.get(split, []):
            out[stem] = split
    return out


def get_midspan_split_map(manifest: Dict) -> Dict[str, str]:
    """Return {stem: 'train'|'val'|'test'} for midspan photos."""
    m = manifest.get("midspan", {})
    out = {}
    for split in ["train", "val", "test"]:
        for stem in m.get(split, []):
            out[stem] = split
    return out


def create_split_manifest_v2(
    photos_dir: Path,
    labels_dir: Path,
    job_sites_path: Path,
    output_path: Path,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    include_mi_clean: bool = True,
    pins: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Dict:
    """
    LEAK-FREE split manifest for the unified pole dataset: whole geo-SITES (clusters of
    jobs sharing physical poles, scripts/diag/audit_split_leakage.py) are assigned to
    train/val/test — no site straddles splits, so FR-revision re-photographs of the same
    pole can never land in both train and val/test (the master split_manifest.json is
    photo-level random and has 100% site overlap).

    Greedy stratified assignment: sites are placed largest-rare-class-share first into
    the split with the largest relative deficit, jointly balancing the 17 unified class
    instance counts, macro-region (MI/MN/NE/CO) representation and photo-count ratios;
    a repair pass then enforces per-class minimum instances in val/test where the class
    has enough total instances. Deterministic (no RNG). Schema is a superset of the
    master manifest ('pole' stem lists work with get_pole_split_map); 'midspan' is empty
    (scope = unified pole only) and 'sites'/'heldout_span_jobs' record the site->split
    assignment for the held-out-site e2e eval.

    Clean-MI photos follow their SITE (not forced to train) — MI generalization is
    measured. Pass include_mi_clean=False for a non-MI-only manifest.

    pins: {site_id: split} hard assignments the optimizer must honor (e.g. pin the
    span-bearing MNRV site to 'test' so the held-out-site e2e eval has spans to score).
    """
    from datetime import datetime

    sites_meta = json.loads(Path(job_sites_path).read_text())
    job_to_site = sites_meta["jobs"]
    jobs_by_len = sorted(job_to_site, key=len, reverse=True)

    photo_files, uni_cache, mi_clean_stems, _ = collect_unified_pole_eligible(
        photos_dir, labels_dir, include_mi_clean=include_mi_clean, verbose=verbose,
    )

    class_names = list(UNIFIED_POLE_DETECTION_CLASS_NAMES)

    def _stem_job(stem: str) -> Optional[str]:
        for j in jobs_by_len:
            if stem.startswith(j + "_"):
                return j
        # photo_id layout: a bare-pid stem (data/Photos/<pid>.jpg) -> job via the label store
        rec = _pil.label_for(stem)
        if rec and rec.get("job") in job_to_site:
            return rec["job"]
        return None

    # ---- aggregate per site: stems, class instance counts, macro-region counts ----
    site_stems: Dict[str, List[str]] = {}
    site_classes: Dict[str, Dict[str, int]] = {}
    site_regions: Dict[str, Dict[str, int]] = {}
    unmapped = []
    for f in photo_files:
        job = _stem_job(f.stem)
        if job is None:
            unmapped.append(f.stem)
            continue
        sid = job_to_site[job]
        site_stems.setdefault(sid, []).append(f.stem)
        cc = site_classes.setdefault(sid, {})
        for t in uni_cache.get(f.stem, []):
            cc[t['class_name']] = cc.get(t['class_name'], 0) + 1
        region = re.match(r'[A-Za-z]{2}', job).group(0).upper() if re.match(r'[A-Za-z]{2}', job) else 'XX'
        rr = site_regions.setdefault(sid, {})
        rr[region] = rr.get(region, 0) + 1
    if unmapped and verbose:
        print(f'  ! {len(unmapped)} eligible stems map to no known job (left out of manifest -> default train)')

    total_photos = sum(len(v) for v in site_stems.values())
    total_class = {c: sum(cc.get(c, 0) for cc in site_classes.values()) for c in class_names}
    total_region = {}
    for rr in site_regions.values():
        for r, n in rr.items():
            total_region[r] = total_region.get(r, 0) + n

    splits = ('train', 'val', 'test')
    ratio_of = dict(zip(splits, ratios))
    target_photos = {s: r * total_photos for s, r in zip(splits, ratios)}
    # strict size cap: val/test may never exceed 1.5x their photo target (a site
    # bigger than the cap simply cannot be a val/test site)
    cap = {s: (1.5 * target_photos[s] if s != 'train' else float('inf')) for s in splits}

    # classes with a handful of total instances (e.g. 'primary' after the MI drop)
    # can't be balanced — exclude them from the cost so they don't dominate it
    bal_classes = [c for c in class_names if total_class[c] >= 20]

    def _new_counters():
        return ({s: 0.0 for s in splits},
                {s: {c: 0 for c in class_names} for s in splits},
                {s: {r: 0 for r in total_region} for s in splits})

    # Hinge-based cost: what actually hurts honest measurement is UNDERSHOOT — a class
    # with (near-)zero val/test instances is unmeasurable, a class with <50% of its
    # instances in train is untrainable. Overshoot is harmless for concentrated classes
    # (a class living at one site can't be ratio-balanced; a symmetric squared cost would
    # rather starve val of it than overshoot, which is exactly wrong here). Small
    # symmetric terms keep photos/regions near the ratios and break ties.
    # The site structure can't satisfy everything (MN arms live in 2 sites, one
    # train-forced by size; post is 90% one site; CO has 2 sites total) — so the
    # hinges PRIORITIZE: train coverage >= TEST coverage (headline metrics) > val
    # coverage (checkpoint selection / conf tuning only).
    MIN_EVAL_SHARE, MIN_TRAIN_SHARE, MIN_REGION_EVAL = 0.08, 0.50, 0.05
    W_EVAL = {'val': 1.5, 'test': 4.0}
    W_REVAL = {'val': 1.0, 'test': 2.0}
    W_TRAIN, W_TIE, W_REGION, W_PHOTO = 4.0, 0.15, 0.5, 3.0

    def _cost(cur_photos, cur_class, cur_region) -> float:
        cost = 0.0
        for c in bal_classes:
            tot = total_class[c]
            for s in ('val', 'test'):
                share = cur_class[s][c] / tot
                cost += W_EVAL[s] * max(0.0, MIN_EVAL_SHARE - share)
                cost += W_TIE * (share - ratio_of[s]) ** 2
            tr_share = cur_class['train'][c] / tot
            cost += W_TRAIN * max(0.0, MIN_TRAIN_SHARE - tr_share)
        for reg, tot in total_region.items():
            # every region measurable in val AND test (NEOM = the OOD-hard family;
            # a val without it makes checkpoint selection blind to it)
            for s in ('val', 'test'):
                cost += W_REVAL[s] * max(0.0, MIN_REGION_EVAL - cur_region[s][reg] / tot)
            cost += W_REVAL['test'] * max(0.0, MIN_TRAIN_SHARE - cur_region['train'][reg] / tot)
        for s in splits:
            r = ratio_of[s]
            for reg, tot in total_region.items():
                cost += W_REGION * (cur_region[s][reg] / tot - r) ** 2
            cost += W_PHOTO * (cur_photos[s] / total_photos - r) ** 2
        return cost

    def _apply(sid, s, cur_photos, cur_class, cur_region, sign=1):
        cur_photos[s] += sign * len(site_stems[sid])
        for c, v in site_classes[sid].items():
            cur_class[s][c] += sign * v
        for r, v in site_regions[sid].items():
            cur_region[s][r] += sign * v

    # ---- greedy seed: big sites first into the most photo-deficient feasible split ----
    pins = {sid: s for sid, s in (pins or {}).items() if sid in site_stems}
    cur_photos, cur_class, cur_region = _new_counters()
    assign: Dict[str, str] = {}
    for sid, s in pins.items():
        assign[sid] = s
        _apply(sid, s, cur_photos, cur_class, cur_region)
    for sid in sorted(site_stems, key=lambda x: (-len(site_stems[x]), x)):
        if sid in pins:
            continue
        n = len(site_stems[sid])
        feasible = [s for s in splits if cur_photos[s] + n <= cap[s]] or ['train']
        best = max(feasible, key=lambda s: (target_photos[s] - cur_photos[s]) / max(target_photos[s], 1e-9))
        assign[sid] = best
        _apply(sid, best, cur_photos, cur_class, cur_region)

    # ---- deterministic local search: single-site moves + pairwise swaps ----
    def _try_move(sid, dst):
        src = assign[sid]
        if sid in pins or dst == src or cur_photos[dst] + len(site_stems[sid]) > cap[dst]:
            return False
        before = _cost(cur_photos, cur_class, cur_region)
        _apply(sid, src, cur_photos, cur_class, cur_region, -1)
        _apply(sid, dst, cur_photos, cur_class, cur_region, +1)
        if _cost(cur_photos, cur_class, cur_region) < before - 1e-12:
            assign[sid] = dst
            return True
        _apply(sid, dst, cur_photos, cur_class, cur_region, -1)
        _apply(sid, src, cur_photos, cur_class, cur_region, +1)
        return False

    def _try_swap(sa, sb):
        a, b = assign[sa], assign[sb]
        if a == b or sa in pins or sb in pins:
            return False
        na, nb = len(site_stems[sa]), len(site_stems[sb])
        if cur_photos[b] - nb + na > cap[b] or cur_photos[a] - na + nb > cap[a]:
            return False
        before = _cost(cur_photos, cur_class, cur_region)
        for sid, src, dst in ((sa, a, b), (sb, b, a)):
            _apply(sid, src, cur_photos, cur_class, cur_region, -1)
            _apply(sid, dst, cur_photos, cur_class, cur_region, +1)
        if _cost(cur_photos, cur_class, cur_region) < before - 1e-12:
            assign[sa], assign[sb] = b, a
            return True
        for sid, src, dst in ((sa, b, a), (sb, a, b)):
            _apply(sid, src, cur_photos, cur_class, cur_region, -1)
            _apply(sid, dst, cur_photos, cur_class, cur_region, +1)
        return False

    sids = sorted(site_stems)
    for _pass in range(50):
        improved = False
        for sid in sids:
            for dst in splits:
                if _try_move(sid, dst):
                    improved = True
        for i, sa in enumerate(sids):
            for sb in sids[i + 1:]:
                if _try_swap(sa, sb):
                    improved = True
        if not improved:
            break

    # ---- repair pass: per-class minimum instances in val and test ----
    for s in ('val', 'test'):
        for c in class_names:
            if total_class[c] == 0:
                continue
            min_needed = min(8, max(1, total_class[c] // 20))
            if cur_class[s][c] >= min_needed:
                continue
            # smallest train site carrying this class
            candidates = sorted(
                (sid for sid in assign if assign[sid] == 'train' and sid not in pins
                 and site_classes[sid].get(c, 0) > 0),
                key=lambda sid: len(site_stems[sid]))
            for sid in candidates:
                if cur_class[s][c] >= min_needed:
                    break
                n = len(site_stems[sid])
                # don't gut train of a rare class entirely, and respect the photo cap
                if cur_class['train'][c] - site_classes[sid].get(c, 0) < min_needed:
                    continue
                if cur_photos[s] + n > 1.5 * target_photos[s]:
                    continue
                assign[sid] = s
                cur_photos['train'] -= n; cur_photos[s] += n
                for cc, v in site_classes[sid].items():
                    cur_class['train'][cc] -= v; cur_class[s][cc] += v
                for r, v in site_regions[sid].items():
                    cur_region['train'][r] -= v; cur_region[s][r] += v
            if verbose and cur_class[s][c] < min_needed:
                print(f'  ! {s}: class {c} below minimum ({cur_class[s][c]}/{min_needed}) '
                      f'— not enough disjoint sites carry it')

    stems = {s: [] for s in splits}
    for sid, s in sorted(assign.items()):
        stems[s].extend(sorted(site_stems[sid]))
    for st in sorted(unmapped):
        stems['train'].append(st)

    # span-job lists for the held-out-site e2e eval (normalized 'COAR-FR01 - 3'->'COAR-FR01');
    # an alias is listed only if ALL its raw jobs share one split (else 'mixed', excluded).
    alias_split: Dict[str, str] = {}
    for job, sid in job_to_site.items():
        alias = sites_meta.get("job_aliases", {}).get(job, job)
        s = assign.get(sid)
        if s is None:
            continue
        if alias in alias_split and alias_split[alias] != s:
            alias_split[alias] = 'mixed'
        else:
            alias_split.setdefault(alias, s)
    heldout_span_jobs = {s: sorted(a for a, sp in alias_split.items() if sp == s) for s in splits}

    manifest = {
        "version": 2,
        "type": "unified_pole_site_split",
        "created": datetime.now().isoformat(),
        "job_sites_path": str(job_sites_path),
        "radius_m": sites_meta.get("radius_m"),
        "ratios": list(ratios),
        "include_mi_clean": include_mi_clean,
        "pole": {s: stems[s] for s in splits},
        "midspan": {s: [] for s in splits},
        "sites": {sid: {"split": s, "jobs": sites_meta["sites"][sid]["jobs"]}
                  for sid, s in sorted(assign.items())},
        "heldout_span_jobs": heldout_span_jobs,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))

    if verbose:
        print(f'\nv2 site-split manifest -> {output_path}')
        print(f'  photos: ' + ' / '.join(f'{s} {len(stems[s])}' for s in splits)
              + f'  (targets {", ".join(f"{target_photos[s]:.0f}" for s in splits)})')
        print(f'  sites:  ' + ' / '.join(
            f'{s} {sum(1 for v in assign.values() if v == s)}' for s in splits))
        mi_per = {s: sum(1 for st in stems[s] if st.upper().startswith('MI')) for s in splits}
        print(f'  clean-MI photos: ' + ' / '.join(f'{s} {mi_per[s]}' for s in splits))
        print(f'  per-class instances (train/val/test):')
        for c in class_names:
            print(f'    {c:>15}: {cur_class["train"][c]:>6} / {cur_class["val"][c]:>5} / {cur_class["test"][c]:>5}')
        print(f'  per-region photos (train/val/test):')
        for r in sorted(total_region):
            print(f'    {r}: {cur_region["train"][r]} / {cur_region["val"][r]} / {cur_region["test"][r]}')
    return manifest


def extract_midspan_job_id_from_filename(filename: str) -> Optional[str]:
    """Extract job ID from midspan photo filename (e.g. MNMW029_(001)-to-(002)_...)."""
    name = Path(filename).name
    match = re.match(r'^([^_]+(?:\s*-\s*\d+)?)', name)
    return match.group(1).strip() if match else None


def midspan_job_excluded_for_wire(
    filename: str,
    excluded_prefixes: Optional[Tuple[str, ...]] = None,
) -> bool:
    """True if photo belongs to a job excluded from midspan wire training (default: MI*).

    Legacy disk-stem check only; under the photo_id layout stems are UUIDs, so
    callers must ALSO check membership in mi_photo_ids(labels_dir).
    """
    prefixes = excluded_prefixes if excluded_prefixes is not None else MIDSPAN_WIRE_EXCLUDED_JOB_PREFIXES
    job_id = extract_midspan_job_id_from_filename(filename)
    if not job_id:
        return False
    return any(job_id.startswith(prefix) for prefix in prefixes)


_MI_LIKE_JOBS_CACHE: Dict[str, set] = {}


def mi_like_jobs(jobs_dir: Optional[Path] = None) -> set:
    """Job names in the MI (bare-wire) annotation regime, detected by CONTENT not name.

    A job is MI-like when its dominant pole-tag company is UtilityCo-MI
    (count of "UtilityCo-MI" company attrs > UtilityCo-MN/UtilityCo-NE counts — a stray single
    CE trace in an UtilityCo-MN job doesn't flag it), or — for unbranded jobs — it carries
    main photos but zero insulator_spec markers (bare-wire: no spool/three-bolt/pin
    hardware annotated anywhere, the defining MI signature).
    """
    from src.config import WIRE_TRACING_JOB_SOURCE_DIR
    jdir = Path(jobs_dir) if jobs_dir is not None else Path(WIRE_TRACING_JOB_SOURCE_DIR)
    key = str(jdir)
    if key in _MI_LIKE_JOBS_CACHE:
        return _MI_LIKE_JOBS_CACHE[key]
    out: set = set()
    for f in sorted(jdir.glob('*.json')):
        try:
            text = f.read_text(errors='ignore')
        except OSError:
            continue
        ce = len(re.findall(r'"company":\s*"UtilityCo-MI"', text))
        other = len(re.findall(r'"company":\s*"(?:UtilityCo-MN|UtilityCo-MN|UtilityCo-NE)"', text))
        bare_wire = text.count('"insulator_spec"') == 0 and text.count('"association": "main"') > 0
        if ce > other or bare_wire:
            out.add(f.stem)
    _MI_LIKE_JOBS_CACHE[key] = out
    return out


def mi_dirty_midspan_pids(jobs_dir: Optional[Path] = None) -> set:
    """MI-like midspan photo_ids carrying a Primary-traced wire marker.

    The MI (bare-wire) regime collapses multiple primaries into one marker, so a
    primary-bearing MI midspan photo teaches missing peaks — excluded. Primary-FREE
    MI midspan photos have trustworthy wire heights and are kept for training.
    """
    from src.config import WIRE_TRACING_JOB_SOURCE_DIR
    jdir = Path(jobs_dir) if jobs_dir is not None else Path(WIRE_TRACING_JOB_SOURCE_DIR)
    dirty: set = set()
    for job in mi_like_jobs(jdir):
        f = jdir / f'{job}.json'
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        traces = (d.get('traces', {}) or {}).get('trace_data', {}) or {}
        jphotos = d.get('photos', {}) or {}
        for cd in (d.get('connections', {}) or {}).values():
            for sv in (cd.get('sections', {}) or {}).values():
                for pid, m in ((sv or {}).get('photos', {}) or {}).items():
                    assoc = m.get('association') if isinstance(m, dict) else m
                    if assoc != 'main':
                        continue
                    ws = ((jphotos.get(pid, {}) or {}).get('photofirst_data') or {}).get('wire') or {}
                    if any((traces.get(w.get('_trace'), {}) or {}).get('cable_type') == 'Primary'
                           for w in ws.values()):
                        dirty.add(pid)
    return dirty


def mi_photo_ids(labels_dir: Path, role: Optional[str] = None) -> set:
    """photo_ids belonging to MI-like jobs in the label store (optionally one role)."""
    mi = mi_like_jobs()
    out: set = set()
    for jf in Path(labels_dir).glob('*.json'):
        if jf.stem not in mi:
            continue
        try:
            photos = json.loads(jf.read_text()).get('photos') or {}
        except (OSError, json.JSONDecodeError):
            continue
        for pid, entry in photos.items():
            if role is None or entry.get('role') == role:
                out.add(pid)
    return out


def get_manifest_test_stems(manifest: Dict, domain: str = "pole") -> set:
    """Return set of test stems for E2E evaluation (never-seen data)."""
    return set(manifest.get(domain, {}).get("test", []))


def get_manifest_val_stems(manifest: Dict, domain: str = "pole") -> set:
    """Return set of val stems for inference demos (random / end-to-end prediction)."""
    return set(manifest.get(domain, {}).get("val", []))


def _resolve_stems_to_images(stems, images_dir: Path) -> List[Path]:
    """Map split-manifest DISK stems to image Paths: the photo_id layout (data/Photos/<pid>.jpg via
    the disk-stem->pid reverse index) when enabled, else the legacy pole Photos dir glob."""
    if _pil.ENABLED:
        out = []
        for st in sorted(stems):
            # canonical pid-keyed manifest: the stem IS the pid (data/Photos/<pid>.jpg)
            p = Path(_pil.photo_path(st))
            if not p.exists():
                # legacy disk-stem manifest: translate disk-stem -> pid
                pid = _pil.pid_for_disk_stem(st)
                p = Path(_pil.photo_path(pid)) if pid else None
            if p is not None and p.exists():
                out.append(p)
        return out
    return [p for p in sorted(images_dir.glob("*.jpg")) if p.stem in stems]


def get_e2e_test_images(domain: str = "equipment") -> List[Path]:
    """
    Canonical helper for E2E test images (manifest-filtered, never-seen during training).
    If E2E_USE_TEST_SPLIT_ONLY, returns only test-split images; raises if manifest or
    test stems are missing. Use this everywhere for consistent unseen evaluation.
    """
    from .config import (
        E2E_USE_TEST_SPLIT_ONLY,
        EQUIPMENT_E2E_IMAGES_DIR,
        ATTACHMENT_E2E_IMAGES_DIR,
        EQUIPMENT_DATASET_DIR,
        ATTACHMENT_DATASET_DIR,
    )

    images_dir = Path(EQUIPMENT_E2E_IMAGES_DIR) if domain == "equipment" else Path(ATTACHMENT_E2E_IMAGES_DIR)
    # Image source: photo_id layout (data/Photos/<pid>.jpg, mapped from the disk stem via the
    # reverse index) when enabled, else the legacy pole Photos dir. The split manifest keys on disk
    # stems, so under the layout we resolve each test disk-stem to its pid path.
    if not E2E_USE_TEST_SPLIT_ONLY:
        if _pil.ENABLED:
            return [Path(p) for _pid, p in _pil.iter_photos("pole")]
        return sorted(images_dir.glob("*.jpg"))

    manifest = load_split_manifest()
    if manifest:
        test_stems = get_manifest_test_stems(manifest, "pole")
    else:
        test_dir = (EQUIPMENT_DATASET_DIR if domain == "equipment" else ATTACHMENT_DATASET_DIR) / "images" / "test"
        test_stems = {p.stem for p in test_dir.glob("*.jpg")} if test_dir.exists() else set()

    if not test_stems:
        raise RuntimeError(
            f"E2E test-only mode is enabled but no test stems were found for '{domain}'. "
            "Ensure split manifest exists (preferred) or dataset test split is prepared."
        )
    filtered = _resolve_stems_to_images(test_stems, images_dir)
    if not filtered:
        raise RuntimeError(
            f"E2E test-only mode is enabled for '{domain}', but no files matched test stems "
            f"(source: {'photo_id layout' if _pil.ENABLED else images_dir}). "
            "Check image source dir and split manifest consistency."
        )
    return filtered


def get_e2e_val_images(domain: str = "equipment") -> List[Path]:
    """
    Return val-split images for inference demos (random prediction, end-to-end).
    Uses manifest val stems. Raises if manifest or val stems missing.
    """
    from .config import EQUIPMENT_E2E_IMAGES_DIR, ATTACHMENT_E2E_IMAGES_DIR

    images_dir = Path(EQUIPMENT_E2E_IMAGES_DIR) if domain == "equipment" else Path(ATTACHMENT_E2E_IMAGES_DIR)

    manifest = load_split_manifest()
    if not manifest:
        raise RuntimeError("Split manifest not found. Generate manifest for val-based inference demos.")
    val_stems = get_manifest_val_stems(manifest, "pole")
    if not val_stems:
        raise RuntimeError("No pole val stems in split manifest.")
    filtered = _resolve_stems_to_images(val_stems, images_dir)
    if not filtered:
        raise RuntimeError(
            f"No images matched manifest val stems "
            f"(source: {'photo_id layout' if _pil.ENABLED else images_dir}). Check manifest and image dir."
        )
    return filtered


def parse_label_file(label_path: Path) -> Tuple[Optional[List[float]], Optional[List[float]], Dict[str, Tuple[float, float]], Optional[float]]:
    """
    Parse label file to extract pole bbox, ruler bbox, keypoints, and PPI.
    
    Args:
        label_path: Path to the label file
        
    Returns:
        Tuple of (pole_bbox, ruler_bbox, keypoints_dict, ppi)
        - pole_bbox: [left, right, top, bottom] in percent coordinates, or None
        - ruler_bbox: [left, right, top, bottom] in percent coordinates, or None
        - keypoints: Dict mapping height strings to (x, y) tuples in percent coordinates
        - ppi: Pixels per inch value, or None
    """
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    keypoints = {}
    pole_bbox = None
    ruler_bbox = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('#') and ',' in line and 'Left,Right,Top,Bottom' not in line:
            try:
                bbox_data = line.replace('#', '').strip().split(',')
                if len(bbox_data) >= 7:
                    # Check previous lines for type
                    for j in range(i-1, -1, -1):
                        prev_line = lines[j].strip()
                        if prev_line and not prev_line.startswith('#'):
                            break
                        if 'Pole bounding box' in prev_line:
                            pole_bbox = [float(x) for x in bbox_data[:4]]  # left, right, top, bottom
                            break
                        elif 'Ruler bounding box' in prev_line:
                            ruler_bbox = [float(x) for x in bbox_data[:4]]
                            break
            except Exception:
                pass
        elif not line.startswith('#') and ',' in line:
            parts = line.strip().split(',')
            if len(parts) == 3 and parts[0] and parts[1] and parts[2]:
                try:
                    height = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    keypoints[height] = (x, y)
                except Exception:
                    pass
    
    # Parse PPI from location file
    ppi = None
    for line in lines:
        line = line.strip()
        if line.startswith('# PPI='):
            try:
                ppi_str = line.split('=')[1].strip()
                ppi = float(ppi_str)
            except Exception:
                pass
    
    return pole_bbox, ruler_bbox, keypoints, ppi


def keypoint_in_bbox(kp_x_percent: float, kp_y_percent: float, 
                     bbox_left: float, bbox_right: float, 
                     bbox_top: float, bbox_bottom: float) -> bool:
    """
    Check if keypoint (in percent coordinates) is within bbox (in percent coordinates).
    
    Args:
        kp_x_percent: Keypoint x coordinate in percent
        kp_y_percent: Keypoint y coordinate in percent
        bbox_left: Bounding box left edge in percent
        bbox_right: Bounding box right edge in percent
        bbox_top: Bounding box top edge in percent
        bbox_bottom: Bounding box bottom edge in percent
        
    Returns:
        True if keypoint is within bbox, False otherwise
    """
    return (bbox_left <= kp_x_percent <= bbox_right and 
            bbox_top <= kp_y_percent <= bbox_bottom)


def check_dataset_complete(dataset_dir: Path, photo_files: Optional[List[Path]] = None,
                          train_files: Optional[List[Path]] = None,
                          val_files: Optional[List[Path]] = None,
                          test_files: Optional[List[Path]] = None,
                          manifest: Optional[Dict] = None,
                          domain: str = "pole") -> bool:
    """
    Check if all expected files exist in the dataset.
    
    This function supports three calling patterns:
    1. With photo_files, train_files, val_files, test_files: Checks exact counts
    2. With manifest: Verifies all manifest stems exist in dataset (catches new train samples)
    3. Without files/manifest: Checks that train/val/test splits have matching images/labels
    
    Args:
        dataset_dir: Path to dataset directory
        photo_files: Optional list of all photo files (for exact count checking)
        train_files: Optional list of training files
        val_files: Optional list of validation files
        test_files: Optional list of test files
        manifest: Optional split manifest; when provided, verifies all stems exist in dataset
        domain: 'pole' or 'midspan' when manifest is provided
        
    Returns:
        True if dataset is complete, False otherwise
    """
    # Check if dataset directory exists
    if not dataset_dir.exists():
        return False

    # If manifest provided, verify all stems exist (supports incremental manifest updates)
    if manifest is not None:
        m = manifest.get(domain, {})
        for split in ["train", "val", "test"]:
            stems = set(m.get(split, []))
            img_dir = dataset_dir / "images" / split
            lbl_dir = dataset_dir / "labels" / split
            for stem in stems:
                if not (img_dir / f"{stem}.jpg").exists() or not (lbl_dir / f"{stem}.txt").exists():
                    return False
        return True
    
    # Check images and labels for train, val, and test
    train_images = len(list((dataset_dir / "images" / "train").glob("*.jpg")))
    train_labels = len(list((dataset_dir / "labels" / "train").glob("*.txt")))
    val_images = len(list((dataset_dir / "images" / "val").glob("*.jpg")))
    val_labels = len(list((dataset_dir / "labels" / "val").glob("*.txt")))
    test_images = len(list((dataset_dir / "images" / "test").glob("*.jpg")))
    test_labels = len(list((dataset_dir / "labels" / "test").glob("*.txt")))
    
    # If file lists provided, check exact counts
    if train_files is not None and val_files is not None and test_files is not None:
        train_count = len(train_files)
        val_count = len(val_files)
        test_count = len(test_files)
        
        return (train_images >= train_count * 0.5 and train_labels >= train_count * 0.5 and
                val_images >= val_count * 0.5 and val_labels >= val_count * 0.5 and
                test_images >= test_count * 0.5 and test_labels >= test_count * 0.5)
    else:
        # Dataset is complete if we have a reasonable number of files in each split
        # (at least 10 files per split, or if we have files, check that images and labels match)
        has_train = train_images > 0 and train_labels > 0 and train_images == train_labels
        has_val = val_images > 0 and val_labels > 0 and val_images == val_labels
        has_test = test_images > 0 and test_labels > 0 and test_images == test_labels
        
        # Consider complete if we have files in all three splits with matching images/labels
        return has_train and has_val and has_test


def is_photo_labeled(label_path: Path) -> bool:
    """
    Check if a photo has been labeled (has equipment or attachment markers).
    
    A photo is considered labeled if the location file contains equipment
    (riser, transformer, street_light) or attachments (comm, down_guy).
    Photos with only pole_top and height measurements are considered unlabeled.
    
    Args:
        label_path: Path to *_location.txt file
        
    Returns:
        True if photo has equipment or attachment labels, False otherwise
    """
    with open(label_path, 'r') as f:
        content = f.read()
    
    # Check for equipment markers
    equipment_markers = ['riser', 'transformer', 'street_light', 'secondary_drip_loop']
    for marker in equipment_markers:
        if f'\n{marker}' in content or content.startswith(marker):
            return True
    
    # Check for attachment markers
    attachment_markers = ['comm', 'down_guy', 'primary', 'secondary', 'neutral', 'guy']
    for marker in attachment_markers:
        if f'\n{marker}' in content or content.startswith(marker):
            return True
    
    return False


def _pct_to_pixels(pct_x: float, pct_y: float, img_width: int, img_height: int) -> Tuple[float, float]:
    """Convert percentage coordinates (0-100) to pixel coordinates."""
    return (pct_x / 100.0) * img_width, (pct_y / 100.0) * img_height


def _load_bbox_from_location_file(location_path: Path, img_width: int, img_height: int,
                                   section_marker: str) -> Optional[Tuple[int, int, int, int]]:
    """Load bbox from location file for a given section marker (e.g., 'Pole bounding box')."""
    if not location_path.exists():
        return None

    with open(location_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if section_marker in line and 'percentage coordinates' in line.lower():
            if i + 1 < len(lines):
                data_line = lines[i + 1].strip()
                if data_line.startswith('#'):
                    try:
                        bbox_data = data_line.replace('#', '').strip().split(',')
                        if len(bbox_data) >= 4:
                            left_pct, right_pct = float(bbox_data[0]), float(bbox_data[1])
                            top_pct, bottom_pct = float(bbox_data[2]), float(bbox_data[3])
                            x1 = int(_pct_to_pixels(left_pct, 0, img_width, img_height)[0])
                            y1 = int(_pct_to_pixels(0, top_pct, img_width, img_height)[1])
                            x2 = int(_pct_to_pixels(right_pct, 0, img_width, img_height)[0])
                            y2 = int(_pct_to_pixels(0, bottom_pct, img_width, img_height)[1])
                            return (x1, y1, x2, y2)
                    except (ValueError, IndexError):
                        continue
    return None


def load_yolo_label(label_path: Path) -> Optional[List[float]]:
    """Load YOLO format label file."""
    if not label_path.exists():
        return None
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
    if not lines:
        return None
    parts = lines[0].split()
    if len(parts) < 5:
        return None
    return [float(x) for x in parts[:5]]


def load_ruler_marking_keypoints(label_path: Path, crop_width: float, crop_height: float) -> Optional[Dict[float, Dict[str, float]]]:
    """Load ruler marking keypoints from YOLO pose format."""
    if not label_path.exists():
        return None
    
    keypoint_heights = [2.5, 6.5, 10.5, 14.5, 16.5]
    
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
    
    if not lines:
        return None
    
    parts = lines[0].split()
    if len(parts) < 11:
        return None
    
    keypoints = {}
    num_keypoints = (len(parts) - 5) // 3
    
    for i in range(min(num_keypoints, len(keypoint_heights))):
        kp_idx = 5 + i * 3
        if kp_idx + 2 < len(parts):
            kp_x_norm = float(parts[kp_idx])
            kp_y_norm = float(parts[kp_idx + 1])
            kp_v = float(parts[kp_idx + 2])
            
            if kp_v > 0:
                height = keypoint_heights[i]
                keypoints[height] = {
                    'x': kp_x_norm * crop_width,
                    'y': kp_y_norm * crop_height
                }
    
    return keypoints if keypoints else None


def load_pole_top_keypoint(label_path: Path, crop_width: float, crop_height: float) -> Optional[Dict[str, float]]:
    """Load pole top keypoint from YOLO pose format."""
    if not label_path.exists():
        return None
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
    if not lines:
        return None
    parts = lines[0].split()
    if len(parts) >= 8:
        kp_x_norm = float(parts[5])
        kp_y_norm = float(parts[6])
        vis = float(parts[7])
        if vis > 0:
            return {
                'x': kp_x_norm * crop_width,
                'y': kp_y_norm * crop_height
            }
    return None


def load_pole_top_from_location_file(location_path: Path, img_width: int, img_height: int) -> Optional[Dict[str, float]]:
    """Load pole top keypoint from location file (percentage coordinates -> global pixel coordinates)."""
    if not location_path.exists():
        return None
    with open(location_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or ',' not in line:
                continue
            parts = line.split(',')
            if len(parts) >= 3 and parts[0].strip() == 'pole_top':
                try:
                    x_global, y_global = _pct_to_pixels(float(parts[1]), float(parts[2]), img_width, img_height)
                    return {'x': x_global, 'y': y_global}
                except (ValueError, IndexError):
                    continue
    return None


def load_ruler_marking_keypoints_from_location_file(location_path: Path, img_width: int, img_height: int) -> Optional[Dict[float, Dict[str, float]]]:
    """Load ruler marking keypoints from location file (percentage coordinates -> global pixel coordinates)."""
    if not location_path.exists():
        return None

    expected_heights = [2.5, 6.5, 10.5, 14.5, 16.5]
    keypoints = {}

    with open(location_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or ',' not in line:
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    height = float(parts[0].strip())
                    if height in expected_heights:
                        x_global, y_global = _pct_to_pixels(float(parts[1]), float(parts[2]), img_width, img_height)
                        keypoints[height] = {'x': x_global, 'y': y_global}
                except (ValueError, IndexError):
                    continue

    return keypoints if keypoints else None


def load_pole_bbox_from_location_file(location_path: Path, img_width: int, img_height: int) -> Optional[Tuple[int, int, int, int]]:
    """Load pole bounding box from location file (percentage coordinates -> global pixel coordinates)."""
    return _load_bbox_from_location_file(location_path, img_width, img_height, 'Pole bounding box')


def load_ruler_bbox_from_location_file(location_path: Path, img_width: int, img_height: int) -> Optional[Tuple[int, int, int, int]]:
    """Load ruler bounding box from location file (percentage coordinates -> global pixel coordinates)."""
    return _load_bbox_from_location_file(location_path, img_width, img_height, 'Ruler bounding box')


def load_pole_top_ppi(label_path: Path) -> Optional[float]:
    """Load PPI from pole top label file comment."""
    if not label_path.exists():
        return None
    with open(label_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith('# PPI='):
                try:
                    return float(line.split('=')[1])
                except:
                    return None
    return None


def load_ppi_from_label(label_path: Path) -> Optional[float]:
    """Load PPI from label file comment. Alias for load_pole_top_ppi."""
    return load_pole_top_ppi(label_path)


def calculate_ppi_from_keypoints(keypoints: Dict[float, Dict[str, float]]) -> Optional[float]:
    """
    Calculate Pixels-per-Inch (PPI) from ruler marking keypoints using linear regression.
    
    This function implements Algorithm 3 from the research paper:
    - Fits linear regression on the five ruler marking keypoints (2.5, 6.5, 10.5, 14.5, 16.5 feet)
    - Extracts the slope (pixels per foot) from the linear model
    - Converts to pixels per inch by dividing by 12
    
    Args:
        keypoints: Dictionary mapping height (float) to {'x': x_coord, 'y': y_coord}
                  Expected keys: 2.5, 6.5, 10.5, 14.5, 16.5 (feet)
    
    Returns:
        PPI value (pixels per inch) if calculation succeeds, None otherwise
    """
    expected_heights = [2.5, 6.5, 10.5, 14.5, 16.5]
    
    # Extract heights and Y-coordinates for available keypoints
    heights = []
    y_coords = []
    
    for height in expected_heights:
        if height in keypoints and 'y' in keypoints[height]:
            heights.append(height)
            y_coords.append(keypoints[height]['y'])
    
    # Need at least 2 keypoints for linear regression
    if len(heights) < 2:
        return None
    
    try:
        # Fit linear regression: y = a * h + b, where a is pixels per foot
        y_coeffs = np.polyfit(heights, y_coords, 1)
        slope = y_coeffs[0]  # pixels per foot
        
        # Convert to pixels per inch: PPI = slope / 12
        ppi = slope / 12.0
        
        return ppi if ppi > 0 else None
    except Exception:
        return None


def load_ground_truth_keypoints(image_path: Path, keypoint_names: List[str]) -> List[Dict]:
    """
    Load ground truth ruler marking keypoints from training dataset label file (YOLO format).
    
    Args:
        image_path: Path to ruler crop image
        keypoint_names: List of keypoint names (e.g., ['2.5', '6.5', '10.5', '14.5', '16.5'])
        
    Returns:
        List of keypoint dictionaries with 'name', 'x', 'y', 'conf', 'ppi' keys
    """
    import cv2
    
    # Load from training dataset label file (YOLO format with normalized coordinates)
    label_file_path = DATASETS_DIR / 'ruler_marking_detection' / 'labels' / 'val' / f"{image_path.stem}.txt"
    
    if not label_file_path.exists():
        return []

    # Load the ruler crop image to get dimensions
    orig_image = cv2.imread(str(image_path))
    if orig_image is None:
        return []
    orig_h, orig_w = orig_image.shape[:2]

    # Load keypoints from YOLO format label file
    gt_points = []
    
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the label line (non-comment line)
    label_line = None
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            label_line = line
            break
    
    if not label_line:
        return []
    
    # Parse YOLO format: class x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
    parts = label_line.split()
    num_keypoints = len(keypoint_names)
    if len(parts) >= 5 + num_keypoints * 3:
        # Keypoints start at index 5 (after class + bbox)
        for i in range(num_keypoints):
            kp_idx = 5 + i * 3
            if kp_idx + 2 < len(parts):
                try:
                    # YOLO format: normalized coordinates (0-1) relative to image
                    x_norm = float(parts[kp_idx])
                    y_norm = float(parts[kp_idx + 1])
                    visibility = float(parts[kp_idx + 2])
                    
                    # Only include visible keypoints
                    if visibility >= 2:  # Visible keypoint
                        # Convert normalized coordinates to pixel coordinates
                        x_px = x_norm * orig_w
                        y_px = y_norm * orig_h
                        
                        gt_points.append({
                            'name': keypoint_names[i],
                            'x': x_px,
                            'y': y_px,
                            'conf': 1.0,  # GT has full confidence
                            'ppi': 0.0  # PPI not available from YOLO labels
                        })
                except (ValueError, IndexError):
                    continue
    
    return gt_points


def load_ground_truth_pole_top(image_path: Path) -> Optional[Dict]:
    """Load ground truth pole top keypoint from Pole Top Detection dataset label file."""
    import cv2

    orig_image = cv2.imread(str(image_path))
    if orig_image is None:
        return None
    orig_h, orig_w = orig_image.shape[:2]

    # Determine split from path
    if '/test/' in str(image_path):
        split = 'test'
    elif '/train/' in str(image_path):
        split = 'train'
    else:
        split = 'val'

    label_path = DATASETS_DIR / 'pole_top_detection' / 'labels' / split / f"{image_path.stem}.txt"
    if not label_path.exists():
        return None

    with open(label_path, 'r') as f:
        ppi = 0.0
        for line in f:
            line = line.strip()
            if line.startswith('# PPI='):
                try:
                    ppi = float(line.split('=')[1])
                except:
                    pass
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 8:
                try:
                    if int(parts[7]) >= 2:  # Visible
                        x_full = float(parts[5]) * orig_w
                        y_full = float(parts[6]) * orig_h
                        return {'name': 'pole_top', 'x': x_full, 'y': y_full, 'conf': 1.0, 'ppi': ppi}
                except (ValueError, IndexError):
                    continue

    return None

# ============================================================================
# Equipment Detection Data Utilities
# ============================================================================


def parse_equipment_from_label_file(label_path: Path) -> List[Dict]:
    """
    Parse equipment bounding boxes from a location label file.

    Equipment bbox lines have format:
        riser1_bbox,Left,Right,Top,Bottom  (percentage coordinates 0-100)

    Args:
        label_path: Path to the *_location.txt file

    Returns:
        List of dicts with keys: 'class_id', 'class_name', 'left', 'right', 'top', 'bottom'
        (all coordinates in percentage 0-100)
    """
    if not label_path.exists():
        return []

    equipment = []

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(',')
            if len(parts) < 5:
                continue

            name = parts[0].strip()
            if not name.endswith('_bbox'):
                continue

            try:
                left = float(parts[1])
                right = float(parts[2])
                top = float(parts[3])
                bottom = float(parts[4])
            except (ValueError, IndexError):
                continue

            # Determine class from name prefix
            if name.startswith('riser'):
                class_name = 'riser'
            elif name.startswith('transformer'):
                class_name = 'transformer'
            elif name.startswith('street_light'):
                class_name = 'street_light'
            elif name.startswith('secondary_drip_loop'):
                class_name = 'secondary_drip_loop'
            else:
                continue

            equipment.append({
                'class_id': EQUIPMENT_CLASSES[class_name],
                'class_name': class_name,
                'left': left,
                'right': right,
                'top': top,
                'bottom': bottom,
            })

    return equipment


def equipment_bbox_to_yolo(left: float, right: float, top: float, bottom: float) -> Tuple[float, float, float, float]:
    """
    Convert equipment bbox from percentage (0-100) left/right/top/bottom
    to YOLO normalized (0-1) x_center/y_center/width/height.

    Returns:
        (x_center, y_center, width, height) all in 0-1 range
    """
    x_center = (left + right) / 200.0
    y_center = (top + bottom) / 200.0
    width = (right - left) / 100.0
    height = (bottom - top) / 100.0
    return x_center, y_center, width, height


def parse_attachments_from_label_file(label_path: Path) -> List[Dict]:
    """
    Parse attachment bounding boxes (comm, down_guy) from a location label file.

    Attachment bbox lines have format:
        comm1_bbox,Left,Right,Top,Bottom  (percentage coordinates 0-100)
        down_guy1_bbox,Left,Right,Top,Bottom

    Returns:
        List of dicts with keys: 'class_id', 'class_name', 'left', 'right', 'top', 'bottom'
    """
    if not label_path.exists():
        return []

    attachments = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 5:
                continue
            name = parts[0].strip()
            if not name.endswith('_bbox'):
                continue
            try:
                left, right, top, bottom = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except (ValueError, IndexError):
                continue
            if name.startswith('comm'):
                class_name = 'comm'
            elif name.startswith('down_guy'):
                class_name = 'down_guy'
            elif name.startswith('primary'):
                class_name = 'primary'
            elif name.startswith('secondary_drip'):
                continue  # equipment, not attachment
            elif name.startswith('secondary'):
                class_name = 'secondary'
            elif name.startswith('open_secondary') or name.startswith('neutral'):
                class_name = 'neutral'
            elif name.startswith('power_guy') or (name.startswith('guy') and not name.startswith('guying')):
                class_name = 'guy'
            else:
                continue
            attachments.append({
                'class_id': ATTACHMENT_CLASSES[class_name],
                'class_name': class_name,
                'left': left,
                'right': right,
                'top': top,
                'bottom': bottom,
            })
    return attachments


def parse_wires_from_label_file(label_path: Path) -> List[Dict]:
    """
    Parse midspan wire bounding boxes from a location label file.

    Wire bbox lines: wire1_bbox,Left,Right,Top,Bottom (percentage 0-100).
    Requires # PPI= in file (caller should filter).

    Returns:
        List of dicts with keys: class_id (0), class_name ('wire'), left, right, top, bottom
    """
    if not label_path.exists():
        return []

    wires = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 5:
                continue
            name = parts[0].strip()
            if not re.match(r'^wire\d+_bbox$', name):
                continue
            try:
                left, right, top, bottom = (
                    float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                )
            except (ValueError, IndexError):
                continue
            wires.append({
                'class_id': 0,
                'class_name': 'wire',
                'left': left,
                'right': right,
                'top': top,
                'bottom': bottom,
            })
    return wires


def parse_wire_y_percent_from_label_file(label_path: Path) -> List[float]:
    """
    Parse midspan wire height markers (wireN,PercentX,PercentY) from a location file.

    Deduplicates wires at the same PercentY (within 0.001%) and returns sorted heights.
    """
    if not label_path.exists():
        return []

    ys: List[float] = []
    seen: set = set()
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            if not re.match(r'^wire\d+$', name):
                continue
            try:
                y_pct = float(parts[2])
            except (ValueError, IndexError):
                continue
            key = round(y_pct, 3)
            if key in seen:
                continue
            seen.add(key)
            ys.append(y_pct)
    return sorted(ys)


def extract_ruler_column_strip(
    img: np.ndarray,
    ruler_bbox_pct: List[float],
    width_expand: float = 1.0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop a full-height vertical strip using ruler bbox left/right (percent coords).

    width_expand: multiplicative widening of the ruler x-range about its centre (1.0 =
        legacy ruler-width crop). >1 includes image columns on either side of the ruler so
        a wire occluded AT the ruler column (e.g. by the calibration stick or pole) can
        still register where it is visible; after the resize to a fixed strip width the
        central-band profile then averages a wider physical region. MUST match between
        dataset prep and inference (wire_tracing_e2e.detect_midspan_points_strip).

    Returns:
        strip image and pixel crop box (x1, y1, x2, y2) in full-image coordinates.
    """
    img_h, img_w = img.shape[:2]
    left_pct, right_pct = ruler_bbox_pct[0], ruler_bbox_pct[1]
    x1f = left_pct / 100.0 * img_w
    x2f = right_pct / 100.0 * img_w
    if width_expand != 1.0:
        cx = 0.5 * (x1f + x2f)
        half = 0.5 * (x2f - x1f) * width_expand
        x1f, x2f = cx - half, cx + half
    x1 = int(x1f)
    x2 = int(x2f)
    x1 = max(0, min(x1, img_w - 1))
    x2 = max(x1 + 1, min(x2, img_w))
    strip = img[0:img_h, x1:x2]
    return strip, (x1, 0, x2, img_h)


def parse_ruler_anchor_points(label_path: Path) -> List[Tuple[float, float, float]]:
    """Ruler tick anchors WITH x from a ``*_location.txt``: ``(height_ft, percent_x, percent_y)``.

    Anchor lines are ``height_ft, percentX, percentY``; only the real RULER_ANCHOR_FEET
    ticks (2.5/6.5/10.5/14.5/16.5) are kept — the ``0.0`` ground row and the legacy
    ``17.0`` top row are ignored (projection-only policy).
    """
    from src.height_calculations import RULER_ANCHOR_FEET
    pts: List[Tuple[float, float, float]] = []
    label_path = Path(label_path)
    if not label_path.exists():
        return pts
    for line in label_path.read_text().splitlines():
        parts = line.split(',')
        if len(parts) != 3:
            continue
        try:
            ft, px, py = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        if ft in RULER_ANCHOR_FEET:
            pts.append((ft, px, py))
    return pts


def extract_ruler_line_strip(
    img: np.ndarray,
    anchor_points: List[Tuple[float, float, float]],
    fit,
    width_ft: float = 3.0,
) -> Optional[Tuple[np.ndarray, Dict]]:
    """Rectified strip along the RULER AXIS: ground line (0 ft) to the photo top.

    The strip axis is the least-squares straight line x = m·y + c (pixels) through the
    ruler tick anchors, so a tilted ruler yields a tilted (sheared) crop that is
    rectified into a straight image. Width = ``width_ft`` converted to pixels via the
    projective height model's LOCAL vertical scale at the mid-ruler height (px are
    assumed square). Bottom edge = projected 0.0 ft ground line (clamped to the photo);
    top edge = image row 0. Off-image columns are zero-padded (black), never replicated.

    Returns (strip, meta) or None when the line/fit is unusable. meta records the
    affine (m, c), ground_y_px, width_px and in-strip normalization so labels and a
    matching inference crop can reproduce the mapping exactly.
    """
    from src.ruler_height_model import height_in_at, percent_y_at_height
    if fit is None or len(anchor_points) < 2:
        return None
    img_h, img_w = img.shape[:2]
    ys = np.array([p[2] / 100.0 * img_h for p in anchor_points], dtype=float)
    xs = np.array([p[1] / 100.0 * img_w for p in anchor_points], dtype=float)
    if len({round(v, 3) for v in ys}) < 2:
        return None
    m, c = np.polyfit(ys, xs, 1)                       # x = m·y + c (pixels)

    ground_pct = percent_y_at_height(fit, 0.0)
    if ground_pct is None or ground_pct * img_h / 100.0 <= ys.max():
        ground_y = img_h                               # degenerate inverse -> full height
    else:
        ground_y = min(img_h, int(round(ground_pct / 100.0 * img_h)))
    if ground_y < 8:
        return None

    # local vertical scale (inches per percentY) at mid-ruler -> pixels for width_ft
    py_mid = float(np.mean([p[2] for p in anchor_points]))
    h1 = height_in_at(fit, py_mid - 0.3)
    h2 = height_in_at(fit, py_mid + 0.3)
    if h1 is None or h2 is None:
        return None
    in_per_pct = abs(h2 - h1) / 0.6
    if in_per_pct <= 1e-6:
        return None
    px_per_inch = (img_h / 100.0) / in_per_pct
    width_px = max(8, int(round(width_ft * 12.0 * px_per_inch)))

    # dst (xd, yd) -> src (xd + m·yd + c - width/2, yd): pure shear, WARP_INVERSE_MAP
    M = np.array([[1.0, m, c - width_px / 2.0],
                  [0.0, 1.0, 0.0]], dtype=np.float64)
    strip = cv2.warpAffine(
        img, M, (width_px, ground_y),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    meta = {
        'line_m': float(m), 'line_c': float(c),
        'ground_y_px': int(ground_y), 'width_px': int(width_px),
        'px_per_inch_mid': float(px_per_inch), 'width_ft': float(width_ft),
        'full_h': int(img_h), 'full_w': int(img_w),
    }
    return strip, meta


def _label_has_ppi(label_path: Path) -> bool:
    """Return True if location file contains a valid # PPI= value."""
    if not label_path.exists():
        return False
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# PPI='):
                try:
                    return float(line.split('=')[1]) > 0
                except (ValueError, IndexError):
                    return False
    return False


def parse_attachments_with_keypoints(label_path: Path) -> List[Dict]:
    """
    Parse attachment bboxes and center keypoints from a location file.

    Returns list of dicts with: class_id, class_name, left, right, top, bottom,
    center (px%, py%), hardware (token|None), cable_type (raw str|None), arm_k (int|None).
    Center comes from comm1, down_guy1 lines (Measurement,PercentX,PercentY).

    `<prefix>_ct,<raw cable_type>` and `<prefix>_arm,<K>` are backward-compatible
    additions (missing -> cable_type/arm_k None) consumed by unified_pole_detection.
    """
    if not label_path.exists():
        return []

    bboxes = {}
    centers = {}
    hardware = {}
    cable_types = {}
    arm_counts = {}
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            name = parts[0].strip()
            if name.endswith('_hw') and len(parts) >= 2:
                hardware[name[:-3]] = parts[1].strip()
                continue
            if name.endswith('_ct') and len(parts) >= 2:
                cable_types[name[:-3]] = parts[1].strip()
                continue
            if name.endswith('_arm') and len(parts) >= 2:
                try:
                    arm_counts[name[:-4]] = int(float(parts[1].strip()))
                except (ValueError, IndexError):
                    pass
                continue
            if name.endswith('_bbox') and len(parts) >= 5:
                try:
                    prefix = name[:-5]
                    bboxes[prefix] = {
                        'left': float(parts[1]), 'right': float(parts[2]),
                        'top': float(parts[3]), 'bottom': float(parts[4]),
                    }
                except (ValueError, IndexError):
                    pass
            elif (name.startswith('comm') or name.startswith('down_guy') or name.startswith('primary')
                  or name.startswith('secondary') or name.startswith('open_secondary') or name.startswith('neutral')
                  or name.startswith('power_guy') or name.startswith('guy')) \
                 and not name.endswith('_bbox') and not name.startswith('secondary_drip') and not name.startswith('guying') and len(parts) >= 3:
                try:
                    centers[name] = (float(parts[1]), float(parts[2]))
                except (ValueError, IndexError):
                    pass

    results = []
    for prefix, bbox in bboxes.items():
        if prefix.startswith('comm'):
            class_name = 'comm'
        elif prefix.startswith('down_guy'):
            class_name = 'down_guy'
        elif prefix.startswith('primary'):
            class_name = 'primary'
        elif prefix.startswith('secondary_drip'):
            continue
        elif prefix.startswith('secondary'):
            class_name = 'secondary'
        elif prefix.startswith('open_secondary') or prefix.startswith('neutral'):
            class_name = 'neutral'
        elif prefix.startswith('power_guy') or (prefix.startswith('guy') and not prefix.startswith('guying')):
            class_name = 'guy'
        else:
            continue
        center = centers.get(prefix)
        results.append({
            'class_id': ATTACHMENT_CLASSES[class_name],
            'class_name': class_name,
            'left': bbox['left'], 'right': bbox['right'],
            'top': bbox['top'], 'bottom': bbox['bottom'],
            'center': center,
            'hardware': hardware.get(prefix),
            'cable_type': cable_types.get(prefix),
            'arm_k': arm_counts.get(prefix),
        })
    return results


# Per-type keypoint counts and names for separate HRNet models
# (Imported from config.py)

def riser_attachment_bbox(
    attachment_pct: Tuple[float, float],
    ppi: float,
    img_w: int,
    img_h: int,
    height_feet: float = RISER_BBOX_HEIGHT_FEET,
    width_feet: float = RISER_BBOX_WIDTH_FEET,
) -> Dict[str, float]:
    """Compute riser bbox (H x W) centered on attachment point. No padding.

    Args:
        attachment_pct: (percent_x, percent_y) in 0-100 range.
        ppi: Pixels per inch for the image.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        height_feet: Box height in feet (default from config).
        width_feet: Box width in feet (default from config).

    Returns:
        Dict with 'left', 'right', 'top', 'bottom' in percentage coordinates (0-100).
    """
    box_w_px = width_feet * 12.0 * ppi
    box_h_px = height_feet * 12.0 * ppi

    cx_px = attachment_pct[0] / 100.0 * img_w
    cy_px = attachment_pct[1] / 100.0 * img_h

    top_px = cy_px - box_h_px / 2
    bottom_px = cy_px + box_h_px / 2

    left = max(0.0, (cx_px - box_w_px / 2) / img_w * 100.0)
    right = min(100.0, (cx_px + box_w_px / 2) / img_w * 100.0)
    top = max(0.0, top_px / img_h * 100.0)
    bottom = min(100.0, bottom_px / img_h * 100.0)

    return {'left': left, 'right': right, 'top': top, 'bottom': bottom}


def parse_equipment_with_keypoints(label_path: Path) -> List[Dict]:
    """
    Parse equipment bboxes AND their associated keypoints from a location file.

    Returns a list of equipment instances, each with bbox and up to 2 keypoints
    (top/primary, bottom/secondary) in percentage coordinates (0-100).

    Keypoint mapping:
      - riser:        kp0 = riser point,        kp1 = None, kp2 = None (1 keypoint)
      - transformer:  kp0 = top_bolt,            kp1 = bottom, kp2 = None (2 keypoints)
      - street_light: kp0 = upper bracket,       kp1 = lower bracket, kp2 = drip_loop (3 keypoints)
    """
    if not label_path.exists():
        return []

    # First pass: collect all equipment keypoints and bboxes
    bboxes = {}       # e.g. 'riser1' -> {left, right, top, bottom}
    keypoints = {}    # e.g. 'riser1' -> (px, py)  or  'transformer1_top' -> (px, py)

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            name = parts[0].strip()

            if name.endswith('_bbox') and len(parts) >= 5:
                try:
                    prefix = name[:-5]  # strip '_bbox'
                    bboxes[prefix] = {
                        'left': float(parts[1]), 'right': float(parts[2]),
                        'top': float(parts[3]), 'bottom': float(parts[4]),
                    }
                except (ValueError, IndexError):
                    continue
            elif len(parts) >= 3:
                if name.startswith(('riser', 'transformer', 'street_light', 'secondary_drip_loop')):
                    try:
                        keypoints[name] = (float(parts[1]), float(parts[2]))
                    except (ValueError, IndexError):
                        continue

    # Second pass: pair bboxes with their keypoints
    results = []
    for prefix, bbox in bboxes.items():
        # Determine class
        if prefix.startswith('riser'):
            class_name = 'riser'
            # Single keypoint: the riser attachment point
            kp0 = keypoints.get(prefix)  # e.g. 'riser1'
            kp1 = None
        elif prefix.startswith('transformer'):
            class_name = 'transformer'
            kp0 = keypoints.get(f'{prefix}_top')
            kp1 = keypoints.get(f'{prefix}_bottom')
        elif prefix.startswith('street_light'):
            class_name = 'street_light'
            kp0 = keypoints.get(f'{prefix}_upper')
            kp1 = keypoints.get(f'{prefix}_lower')
            kp2 = keypoints.get(f'{prefix}_drip_loop')
        elif prefix.startswith('secondary_drip_loop'):
            class_name = 'secondary_drip_loop'
            kp0 = keypoints.get(prefix)
            kp1 = None
            kp2 = None
        else:
            continue

        # Must have at least one keypoint (street_light needs upper or lower; drip_loop is optional)
        if kp0 is None and kp1 is None:
            continue

        result = {
            'class_id': EQUIPMENT_CLASSES[class_name],
            'class_name': class_name,
            'bbox': bbox,
            'kp0': kp0,
            'kp1': kp1,
        }
        if class_name == 'street_light':
            result['kp2'] = kp2
        results.append(result)

    return results


def _compute_pole_upper70_2x5_crop(
    img: np.ndarray,
    pole_bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[np.ndarray, int, int, int, int, int, int]]:
    """
    Crop to pole bbox, upper 70%, with horizontal expansion for 2:5 aspect ratio.
    Returns (crop, x1_new, y1, x2_new, crop_y2, crop_w_actual, crop_h_actual) or None.
    """
    x1, y1, x2, y2 = pole_bbox
    crop_h_full = y2 - y1
    crop_h = int(crop_h_full * 0.7)
    if crop_h < 10 or (x2 - x1) < 10:
        return None
    target_width = int(crop_h * (2 / 5))
    center_x = (x1 + x2) / 2
    x1_new = max(0, int(center_x - target_width / 2))
    x2_new = min(img_w, int(center_x + target_width / 2))
    if x2_new - x1_new < 10:
        return None
    crop = img[y1 : y1 + crop_h, x1_new:x2_new]
    crop_h_actual, crop_w_actual = crop.shape[:2]
    crop_y2 = y1 + crop_h
    return crop, x1_new, y1, x2_new, crop_y2, crop_w_actual, crop_h_actual


def prepare_equipment_detection_dataset(photos_dir: Path, labels_dir: Path, dataset_dir: Path, verbose: bool = False, workers: int = 1, max_neg_ratio: float = 0.2) -> None:
    """
    Prepare equipment detection dataset (Riser, Transformer, Street Light) for YOLO training.

    Crops each image to the pole bounding box, then takes the upper 70% of that
    crop (where equipment typically appears). Expands the bbox width so the crop
    has 2:5 aspect ratio (no padding—pure crop from source). Only equipment fully contained
    within this region are included. Negative examples (labeled images with no
    equipment in the crop) are included with empty label files, capped at
    max_neg_ratio of positive examples per split. Unlabeled photos
    (no equipment or attachment markers) are skipped.

    Args:
        photos_dir: Path to directory with photos (*.jpg)
        labels_dir: Path to directory with *_location.txt label files
        dataset_dir: Output directory for prepared dataset

    Note:
        Delete the existing dataset directory before re-running if you change
        the preparation logic (e.g. crop strategy).
    """
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    photos_dir = Path(photos_dir)
    labels_dir = Path(labels_dir)
    dataset_dir = Path(dataset_dir)
    manifest = load_split_manifest()

    if check_dataset_complete(dataset_dir, manifest=manifest, domain="pole") if manifest else check_dataset_complete(dataset_dir):
        if verbose:
            print(f"✓ Equipment detection dataset already prepared at {dataset_dir}")
            for split in ['train', 'val', 'test']:
                n_img = len(list((dataset_dir / "images" / split).glob("*.jpg")))
                n_lbl = len(list((dataset_dir / "labels" / split).glob("*.txt")))
                print(f"  {split}: {n_img} images, {n_lbl} labels")
        return

    # Find all photos with pole bbox (required for cropping)
    from PIL import Image
    photo_files = []
    equipment_cache = {}
    photos_skipped_no_pole = 0
    photos_skipped_unlabeled = 0
    photos_with_equipment = 0

    for photo_path in sorted(photos_dir.glob("*.jpg")):
        label_path = _loc_path(labels_dir, photo_path.stem)
        if label_path is None or not label_path.exists():
            continue
        
        # Skip unlabeled photos (only have pole_top/height markers, no equipment/attachments)
        if not is_photo_labeled(label_path):
            photos_skipped_unlabeled += 1
            continue
        try:
            with Image.open(photo_path) as im:
                img_w, img_h = im.size
        except Exception:
            continue
        pole_bbox = load_pole_bbox_from_location_file(label_path, img_w, img_h)
        if pole_bbox is None:
            photos_skipped_no_pole += 1
            continue
        equipment = parse_equipment_from_label_file(label_path)
        equipment_cache[photo_path.stem] = equipment
        if equipment:
            photos_with_equipment += 1
        photo_files.append(photo_path)

    if verbose:
        print(f"Found {len(photo_files)} pole photos with pole bbox")
        print(f"  Skipped {photos_skipped_no_pole} (no pole bbox)")
        print(f"  Skipped {photos_skipped_unlabeled} (unlabeled - no equipment/attachment markers)")
        print(f"  {photos_with_equipment} with equipment")
    class_counts = {name: 0 for name in EQUIPMENT_CLASS_NAMES}
    for photo_path in photo_files:
        for eq in equipment_cache.get(photo_path.stem, []):
            class_counts[eq['class_name']] += 1
    if verbose:
        print(f"Class distribution: {class_counts}")
        print(f"Total equipment instances: {sum(class_counts.values())}")

    # Create directory structure
    for split in ['train', 'val', 'test']:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    if manifest:
        pole_split_map = get_pole_split_map(manifest)
        split_map = {f: pole_split_map.get(f.stem, 'train') for f in photo_files}
        if verbose:
            train_count = sum(1 for s in split_map.values() if s == 'train')
            val_count = sum(1 for s in split_map.values() if s == 'val')
            test_count = sum(1 for s in split_map.values() if s == 'test')
            print(f"Split (manifest): {train_count} train / {val_count} val / {test_count} test")
    else:
        train_files, temp_files = train_test_split(photo_files, test_size=0.2, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        split_map = {f: 'train' for f in train_files}
        split_map.update({f: 'val' for f in val_files})
        split_map.update({f: 'test' for f in test_files})
        if verbose:
            print(f"Split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")
    if verbose:
        print(f"Strategy: Crop to pole bbox, upper 70%, expand width for 2:5 ratio, include negatives")

    # Pre-build riser keypoint lookup and ppi cache (parse once per photo)
    riser_kp_lookup = {}
    ppi_cache = {}
    for photo_path in photo_files:
        label_path = _loc_path(labels_dir, photo_path.stem)
        eq_with_kp = parse_equipment_with_keypoints(label_path)
        ppi_cache[photo_path.stem] = load_ppi_from_label(label_path)
        riser_idx = 0
        for eq in eq_with_kp:
            if eq['class_name'] == 'riser' and eq['kp0'] is not None:
                riser_kp_lookup[(photo_path.stem, riser_idx)] = eq['kp0']
                riser_idx += 1

    # Generate YOLO dataset
    processed = 0
    skipped = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    class_counts = {name: 0 for name in EQUIPMENT_CLASS_NAMES}
    riser_bbox_replaced = 0

    def _process_one_eq(photo_path):
        label_path = _loc_path(labels_dir, photo_path.stem)
        img = cv2.imread(str(photo_path))
        if img is None:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0}, {n: 0 for n in EQUIPMENT_CLASS_NAMES}, 0)

        img_h, img_w = img.shape[:2]
        pole_bbox = load_pole_bbox_from_location_file(label_path, img_w, img_h)
        if pole_bbox is None:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0}, {n: 0 for n in EQUIPMENT_CLASS_NAMES}, 0)

        crop_result = _compute_pole_upper70_2x5_crop(img, pole_bbox, img_w, img_h)
        if crop_result is None:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0}, {n: 0 for n in EQUIPMENT_CLASS_NAMES}, 0)

        crop, x1_new, y1, x2_new, crop_y2, crop_w_actual, crop_h_actual = crop_result
        ppi = ppi_cache.get(photo_path.stem)
        split = split_map[photo_path]
        img_dst = dataset_dir / "images" / split / photo_path.name
        lbl_dst = dataset_dir / "labels" / split / f"{photo_path.stem}.txt"

        equipment = equipment_cache.get(photo_path.stem, [])
        lines = []
        riser_idx = 0
        _class_counts = {n: 0 for n in EQUIPMENT_CLASS_NAMES}
        _riser_replaced = 0

        for eq in equipment:
            l_px = eq['left'] / 100.0 * img_w
            r_px = eq['right'] / 100.0 * img_w
            t_px = eq['top'] / 100.0 * img_h
            b_px = eq['bottom'] / 100.0 * img_h
            if l_px >= x1_new and r_px <= x2_new and t_px >= y1 and b_px <= crop_y2:
                if eq['class_name'] == 'riser':
                    kp = riser_kp_lookup.get((photo_path.stem, riser_idx))
                    if kp is not None and ppi is not None:
                        new_bbox = riser_attachment_bbox(kp, ppi, img_w, img_h)
                        l_px_new = new_bbox['left'] / 100.0 * img_w
                        r_px_new = new_bbox['right'] / 100.0 * img_w
                        t_px_new = new_bbox['top'] / 100.0 * img_h
                        b_px_new = new_bbox['bottom'] / 100.0 * img_h
                        if l_px_new >= x1_new and r_px_new <= x2_new and t_px_new >= y1 and b_px_new <= crop_y2:
                            l_px, r_px, t_px, b_px = l_px_new, r_px_new, t_px_new, b_px_new
                            _riser_replaced += 1
                    riser_idx += 1
                left_crop = (l_px - x1_new) / crop_w_actual * 100.0
                right_crop = (r_px - x1_new) / crop_w_actual * 100.0
                top_crop = (t_px - y1) / crop_h_actual * 100.0
                bottom_crop = (b_px - y1) / crop_h_actual * 100.0
                cx, cy, w, h = equipment_bbox_to_yolo(left_crop, right_crop, top_crop, bottom_crop)
                lines.append(f"{eq['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                _class_counts[eq['class_name']] += 1
            elif eq['class_name'] == 'riser':
                riser_idx += 1

        _split = {'train': 0, 'val': 0, 'test': 0}
        _split[split] = 1

        if not lines:
            cv2.imwrite(str(img_dst), crop)
            lbl_dst.write_text("")
            return (1, 0, _split, _class_counts, _riser_replaced)

        cv2.imwrite(str(img_dst), crop)
        lbl_dst.write_text('\n'.join(lines) + '\n')
        return (1, 0, _split, _class_counts, _riser_replaced)

    results = _parallel_map(photo_files, _process_one_eq, workers, desc="Preparing equipment dataset", verbose=verbose)
    for _processed, _skipped, _split, _class, _riser in results:
        processed += _processed
        skipped += _skipped
        for k, v in _split.items():
            split_counts[k] += v
        for k in class_counts:
            class_counts[k] += _class.get(k, 0)
        riser_bbox_replaced += _riser

    # Downsample negative examples (empty label files) per split
    import random as _random
    _random.seed(42)
    neg_removed = 0
    for split in ['train', 'val', 'test']:
        lbl_dir = dataset_dir / "labels" / split
        img_dir = dataset_dir / "images" / split
        all_labels = list(lbl_dir.glob("*.txt"))
        positives = [p for p in all_labels if p.stat().st_size > 0]
        negatives = [p for p in all_labels if p.stat().st_size == 0]
        max_neg = int(len(positives) * max_neg_ratio)
        if len(negatives) > max_neg:
            _random.shuffle(negatives)
            to_remove = negatives[max_neg:]
            for lbl_path in to_remove:
                img_path = img_dir / f"{lbl_path.stem}.jpg"
                lbl_path.unlink(missing_ok=True)
                img_path.unlink(missing_ok=True)
            neg_removed += len(to_remove)
            split_counts[split] -= len(to_remove)
            if verbose:
                print(f"  {split}: kept {max_neg}/{len(negatives)} negatives ({len(to_remove)} removed)")

    # Write data.yaml
    yaml_content = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(EQUIPMENT_CLASS_NAMES)}\n"
        f"names: {EQUIPMENT_CLASS_NAMES}\n"
    )
    with open(dataset_dir / "data.yaml", 'w') as f:
        f.write(yaml_content)

    if verbose:
        print(f"\n✓ Equipment dataset prepared: {processed} images ({skipped} skipped, {neg_removed} negatives removed)")
        print(f"  Riser bboxes replaced with 5'x2' attachment box: {riser_bbox_replaced}")
        for split, count in split_counts.items():
            print(f"  {split}: {count} images")
        print(f"  Class distribution in crops: {class_counts}")
        print(f"  data.yaml: {dataset_dir / 'data.yaml'}")
        print(f"  Classes: {EQUIPMENT_CLASS_NAMES}")


def unified_class_for_attachment(att: Dict) -> Optional[str]:
    """Map one parsed attachment to a unified_pole_detection class name, or None to skip.

    Mirrors the wire_tracer/eval convention so GT labels match the encoder:
      - 'down_guy' prefix -> 'down_guy'
      - 'guy' prefix with no cable_type -> 'guy'
      - otherwise delegate to config.unified_joint_class(hw, cable_type, is_arm, K).
    """
    class_name = att.get('class_name')
    cable_type = att.get('cable_type')
    arm_k = att.get('arm_k')
    if class_name == 'down_guy':
        return 'down_guy'
    if class_name == 'guy' and cable_type is None:
        return 'guy'
    return unified_joint_class(
        hw_token=att.get('hardware'),
        cable_type=cable_type,
        is_arm=(arm_k is not None),
        arm_k=arm_k,
    )


def build_unified_pole_pose_targets(
    label_path: Path,
    img_w: int,
    img_h: int,
    ppi: float,
) -> Optional[List[Dict]]:
    """Build unified_pole_detection pose targets (17 joint classes, 1 keypoint).

    Each attachment is encoded to a joint class via unified_class_for_attachment
    (hardware token + cable_type + arm wire-count), the same 1ft×2ft (H×W) box +
    attachment keypoint as wire_detection. Returns None to SKIP the photo if any
    mapped attachment lacks a keypoint (mirrors wire_detection); [] if no attachment
    maps to a class.
    """
    from .bounding_boxes import calculate_attachment_bounding_box

    if not ppi or ppi <= 0:
        return None

    targets: List[Dict] = []
    saw_mapped = False
    for att in parse_attachments_with_keypoints(label_path):
        mapped = unified_class_for_attachment(att)
        if mapped is None:
            continue
        saw_mapped = True
        center = att.get('center')
        if center is None:
            return None
        kp_x, kp_y = center
        bbox = calculate_attachment_bounding_box(
            {'percentX': kp_x, 'percentY': kp_y},
            ppi,
            img_w,
            img_h,
            height_feet=UNIFIED_POLE_DETECTION_BBOX_HEIGHT_FEET,
            width_feet=UNIFIED_POLE_DETECTION_BBOX_WIDTH_FEET,
        )
        if bbox is None:
            return None
        targets.append({
            'class_id': UNIFIED_POLE_DETECTION_CLASSES[mapped],
            'class_name': mapped,
            'left': bbox['left'],
            'right': bbox['right'],
            'top': bbox['top'],
            'bottom': bbox['bottom'],
            'kp_x': kp_x,
            'kp_y': kp_y,
        })
    return targets if saw_mapped else []


def collect_unified_pole_eligible(
    photos_dir: Path,
    labels_dir: Path,
    include_mi_clean: bool = False,
    verbose: bool = False,
) -> Tuple[List[Path], Dict[str, List[Dict]], set, Dict[str, int]]:
    """
    Shared eligibility scan for the unified pole dataset (single source of truth for
    prepare_unified_pole_detection_dataset AND create_split_manifest_v2).

    Returns (photo_files, uni_cache {stem: targets}, mi_clean_stems, skip_stats).
    Eligibility: labeled location file + pole bbox + PPI + every mapped attachment has
    a keypoint; non-MI by default; include_mi_clean adds primary/crossarm-free MI photos.
    """
    POWER_ARM_CLASSES = {'pin', 'post', 'davit', 'deadend', 'primary',
                         'arm2', 'arm3', 'arm4plus'}
    from PIL import Image

    photos_dir = Path(photos_dir)
    labels_dir = Path(labels_dir)
    photo_files: List[Path] = []
    uni_cache: Dict[str, List[Dict]] = {}
    mi_clean_stems: set = set()
    stats = {'mi': 0, 'mi_dirty': 0, 'no_pole': 0, 'unlabeled': 0,
             'no_ppi': 0, 'missing_kp': 0, 'with_targets': 0}

    # Under the photo_id layout stems are UUIDs, so the legacy MI stem-prefix test never
    # fires — resolve MI membership by job CONTENT (UtilityCo-MI / bare-wire), not name.
    mi_pids = mi_photo_ids(labels_dir, role='pole')

    for photo_path in sorted(photos_dir.glob('*.jpg')):
        # NON-MI filter: skip MI-regime jobs (lack hardware + collapsed-crossarm primaries).
        # include_mi_clean keeps the subset that is primary/crossarm-free.
        is_mi = photo_path.stem.upper().startswith('MI') or photo_path.stem in mi_pids
        if is_mi and not include_mi_clean:
            stats['mi'] += 1
            continue
        label_path = _loc_path(labels_dir, photo_path.stem)
        if label_path is None or not label_path.exists():
            continue
        if not is_photo_labeled(label_path):
            stats['unlabeled'] += 1
            continue
        try:
            with Image.open(photo_path) as im:
                img_w, img_h = im.size
        except Exception:
            continue
        pole_bbox = load_pole_bbox_from_location_file(label_path, img_w, img_h)
        if pole_bbox is None:
            stats['no_pole'] += 1
            continue

        ppi = load_ppi_from_label(label_path)
        if not ppi or ppi <= 0:
            stats['no_ppi'] += 1
            continue

        targets = build_unified_pole_pose_targets(label_path, img_w, img_h, ppi)
        if targets is None:
            stats['missing_kp'] += 1
            continue

        if is_mi:
            # clean-MI gate: keep only primary-free AND crossarm-free photos (the
            # cable-type-only attachments encode identically to non-MI).
            if any(t['class_name'] in POWER_ARM_CLASSES for t in targets):
                stats['mi_dirty'] += 1
                continue
            mi_clean_stems.add(photo_path.stem)

        uni_cache[photo_path.stem] = targets
        if targets:
            stats['with_targets'] += 1
        photo_files.append(photo_path)

    if verbose:
        label = ('non-MI + clean-MI' if include_mi_clean else 'non-MI')
        print(f'Found {len(photo_files)} {label} pole photos with pole bbox + PPI + keypoints')
        if include_mi_clean:
            print(f'  Included {len(mi_clean_stems)} clean MI photos (primary/crossarm-free)')
            print(f'  Skipped {stats["mi_dirty"]} MI photos (carried a primary/crossarm class)')
        print(f'  Skipped {stats["mi"]} (MI job)')
        print(f'  Skipped {stats["no_pole"]} (no pole bbox)')
        print(f'  Skipped {stats["unlabeled"]} (unlabeled)')
        print(f'  Skipped {stats["no_ppi"]} (no PPI)')
        print(f'  Skipped {stats["missing_kp"]} (mapped attachment missing keypoint)')
        print(f'  {stats["with_targets"]} with at least one unified joint class in label file')

    return photo_files, uni_cache, mi_clean_stems, stats


def prepare_unified_pole_detection_dataset(
    photos_dir: Path,
    labels_dir: Path,
    dataset_dir: Path,
    verbose: bool = False,
    workers: int = 1,
    max_neg_ratio: float = 0.2,
    include_mi_clean: bool = False,
    split_manifest_path: Optional[Path] = None,
    mi_train_only: bool = True,
    arm_oversample: int = 1,
) -> None:
    """
    Prepare unified_pole_detection for YOLO pose training (17 joint classes, 1 keypoint).

    Same pole upper-70% 2:5 crop and 1ft×2ft as the legacy wire_detection prep, and
    (H×W) box + attachment keypoint, but CLASSES = the unified joint taxonomy
    (config.UNIFIED_POLE_DETECTION_CLASS_NAMES) encoded per-attachment from
    hardware token + cable_type + arm wire-count via unified_class_for_attachment.
    Labels: YOLO pose — class cx cy w h kp_x kp_y v (1 keypoint, v=2 visible).

    NON-MI ONLY (default): photos whose location-file stem starts with 'MI'
    (case-insensitive) are skipped — MI-regime jobs lack insulator hardware and carry
    misleading multi-primary (collapsed-crossarm) annotations.

    include_mi_clean=True: ADDITIONALLY include MI photos that are *primary-free AND
    crossarm-free* — i.e. none of their attachments encode to a power/crossarm class
    (pin/post/davit/deadend/primary/arm2/arm3/arm4plus). Such MI photos carry only the
    cable-type-determined classes (secondary/open_secondary/neutral/catv/telco/fiber/
    guy/down_guy/unspecified), which the encoder maps IDENTICALLY to non-MI (hardware is
    redundant there), so they are clean additive data for the weak non-power classes.
    These MI photos go to the TRAIN split ONLY — val/test stay non-MI (frozen manifest)
    so per-pole fidelity eval remains comparable to the deployed non-MI model.

    split_manifest_path: alternate manifest (e.g. datasets/split_manifest_v2.json, the
    site-grouped leak-free split). Default None = master split_manifest.json.
    mi_train_only: force clean-MI photos into train (legacy). The v2 split passes False
    so MI photos follow their site and MI generalization is measured on val/test.
    arm_oversample: duplicate arm-bearing (arm2/arm3/arm4plus) TRAIN photos N-1 extra
    times ('_armdup' suffix) — codifies the unified_pole_mi_armboost recipe.

    Skips a photo when PPI is missing or any mapped attachment lacks a keypoint.
    """
    photos_dir = Path(photos_dir)
    labels_dir = Path(labels_dir)
    dataset_dir = Path(dataset_dir or DATASET_DIRS[UNIFIED_POLE_DETECTION])
    manifest = load_split_manifest(split_manifest_path or SPLIT_MANIFEST_PATH)

    if check_dataset_complete(dataset_dir, manifest=manifest, domain="pole") if manifest else check_dataset_complete(dataset_dir):
        if verbose:
            print(f'✓ Unified pole detection dataset already prepared at {dataset_dir}')
            for split in ['train', 'val', 'test']:
                n_img = len(list((dataset_dir / 'images' / split).glob('*.jpg')))
                n_lbl = len(list((dataset_dir / 'labels' / split).glob('*.txt')))
                print(f'  {split}: {n_img} images, {n_lbl} labels')
        return

    photo_files, uni_cache, mi_clean_stems, _stats = collect_unified_pole_eligible(
        photos_dir, labels_dir, include_mi_clean=include_mi_clean, verbose=verbose,
    )
    if verbose and include_mi_clean and mi_train_only:
        print('  (clean-MI photos -> TRAIN only)')

    for split in ['train', 'val', 'test']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    if manifest:
        pole_split_map = get_pole_split_map(manifest)
        split_map = {f: pole_split_map.get(f.stem, 'train') for f in photo_files}
        if verbose:
            train_count = sum(1 for s in split_map.values() if s == 'train')
            val_count = sum(1 for s in split_map.values() if s == 'val')
            test_count = sum(1 for s in split_map.values() if s == 'test')
            print(f'Split (manifest): {train_count} train / {val_count} val / {test_count} test')
    else:
        train_files, temp_files = train_test_split(photo_files, test_size=0.2, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        split_map = {f: 'train' for f in train_files}
        split_map.update({f: 'val' for f in val_files})
        split_map.update({f: 'test' for f in test_files})
        if verbose:
            print(f'Split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test')

    # clean-MI photos are TRAIN-only: val/test stay non-MI (frozen) so per-pole
    # fidelity eval is directly comparable to the deployed non-MI model.
    # (mi_train_only=False — v2 site split — lets MI photos follow their site instead.)
    if mi_clean_stems and mi_train_only:
        forced = sum(1 for f in photo_files
                     if f.stem in mi_clean_stems and split_map.get(f) != 'train')
        for f in photo_files:
            if f.stem in mi_clean_stems:
                split_map[f] = 'train'
        if verbose and forced:
            print(f'  Forced {forced} clean-MI photos out of val/test into train')

    if verbose:
        print(
            'Strategy: pole upper-70% 2:5 crop; 1ft×2ft box + attachment keypoint; '
            'YOLO pose (17 unified joint classes from hardware + cable_type + arm count)'
        )

    processed = 0
    skipped = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    class_counts = {name: 0 for name in UNIFIED_POLE_DETECTION_CLASS_NAMES}

    def _process_one_unified(photo_path):
        label_path = _loc_path(labels_dir, photo_path.stem)
        img = cv2.imread(str(photo_path))
        if img is None:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0}, {n: 0 for n in UNIFIED_POLE_DETECTION_CLASS_NAMES})
        img_h, img_w = img.shape[:2]
        pole_bbox = load_pole_bbox_from_location_file(label_path, img_w, img_h)
        if pole_bbox is None:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0}, {n: 0 for n in UNIFIED_POLE_DETECTION_CLASS_NAMES})
        crop_result = _compute_pole_upper70_2x5_crop(img, pole_bbox, img_w, img_h)
        if crop_result is None:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0}, {n: 0 for n in UNIFIED_POLE_DETECTION_CLASS_NAMES})
        crop, x1_new, y1, x2_new, crop_y2, crop_w_actual, crop_h_actual = crop_result
        targets = uni_cache.get(photo_path.stem, [])
        lines = []
        _class = {n: 0 for n in UNIFIED_POLE_DETECTION_CLASS_NAMES}
        for att in targets:
            l_px = att['left'] / 100.0 * img_w
            r_px = att['right'] / 100.0 * img_w
            t_px = att['top'] / 100.0 * img_h
            b_px = att['bottom'] / 100.0 * img_h
            if l_px >= x1_new and r_px <= x2_new and t_px >= y1 and b_px <= crop_y2:
                left_crop = (l_px - x1_new) / crop_w_actual * 100.0
                right_crop = (r_px - x1_new) / crop_w_actual * 100.0
                top_crop = (t_px - y1) / crop_h_actual * 100.0
                bottom_crop = (b_px - y1) / crop_h_actual * 100.0
                cx, cy, bw, bh = equipment_bbox_to_yolo(left_crop, right_crop, top_crop, bottom_crop)
                kp_x_crop = (att['kp_x'] / 100.0 * img_w - x1_new) / crop_w_actual
                kp_y_crop = (att['kp_y'] / 100.0 * img_h - y1) / crop_h_actual
                kp_x_crop = min(max(kp_x_crop, 0.0), 1.0)
                kp_y_crop = min(max(kp_y_crop, 0.0), 1.0)
                lines.append(
                    f"{att['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
                    f"{kp_x_crop:.6f} {kp_y_crop:.6f} 2"
                )
                _class[att['class_name']] += 1
        split = split_map[photo_path]
        img_dst = dataset_dir / 'images' / split / photo_path.name
        lbl_dst = dataset_dir / 'labels' / split / f'{photo_path.stem}.txt'
        _split = {'train': 0, 'val': 0, 'test': 0}
        _split[split] = 1
        if not lines:
            cv2.imwrite(str(img_dst), crop)
            lbl_dst.write_text('')
            return (1, 0, _split, _class)
        cv2.imwrite(str(img_dst), crop)
        lbl_dst.write_text('\n'.join(lines) + '\n')
        return (1, 0, _split, _class)

    results = _parallel_map(
        photo_files, _process_one_unified, workers, desc='Preparing unified pole detection dataset', verbose=verbose
    )
    for _processed, _skipped, _split, _class in results:
        processed += _processed
        skipped += _skipped
        for k, v in _split.items():
            split_counts[k] += v
        for k in class_counts:
            class_counts[k] += _class.get(k, 0)

    import random as _random
    _random.seed(42)
    neg_removed = 0
    for split in ['train', 'val', 'test']:
        lbl_dir = dataset_dir / 'labels' / split
        img_dir = dataset_dir / 'images' / split
        all_labels = list(lbl_dir.glob('*.txt'))
        positives = [p for p in all_labels if p.stat().st_size > 0]
        negatives = [p for p in all_labels if p.stat().st_size == 0]
        max_neg = int(len(positives) * max_neg_ratio)
        if len(negatives) > max_neg:
            _random.shuffle(negatives)
            to_remove = negatives[max_neg:]
            for lbl_path in to_remove:
                img_path = img_dir / f'{lbl_path.stem}.jpg'
                lbl_path.unlink(missing_ok=True)
                img_path.unlink(missing_ok=True)
            neg_removed += len(to_remove)
            split_counts[split] -= len(to_remove)
            if verbose:
                print(f'  {split}: kept {max_neg}/{len(negatives)} negatives ({len(to_remove)} removed)')

    # arm oversampling (armboost recipe): duplicate TRAIN photos carrying any
    # arm2/arm3/arm4plus instance with an '_armdup' suffix (N-1 extra copies) to
    # counter crossarm dilution. TRAIN only — val/test are never duplicated.
    if arm_oversample > 1:
        import shutil
        arm_ids = {str(UNIFIED_POLE_DETECTION_CLASS_NAMES.index(c))
                   for c in ('arm2', 'arm3', 'arm4plus')}
        lbl_dir = dataset_dir / 'labels' / 'train'
        img_dir = dataset_dir / 'images' / 'train'
        n_dup = 0
        for lbl_path in sorted(lbl_dir.glob('*.txt')):
            if lbl_path.stem.endswith('_armdup') or '_armdup' in lbl_path.stem:
                continue
            classes = {line.split()[0] for line in lbl_path.read_text().splitlines() if line.strip()}
            if not (classes & arm_ids):
                continue
            img_path = img_dir / f'{lbl_path.stem}.jpg'
            if not img_path.exists():
                continue
            for k in range(1, arm_oversample):
                suffix = '_armdup' if k == 1 else f'_armdup{k}'
                shutil.copy2(img_path, img_dir / f'{lbl_path.stem}{suffix}.jpg')
                shutil.copy2(lbl_path, lbl_dir / f'{lbl_path.stem}{suffix}.txt')
                n_dup += 1
        split_counts['train'] += n_dup
        if verbose:
            print(f'  arm oversample x{arm_oversample}: {n_dup} duplicate train images added')

    yaml_content = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(UNIFIED_POLE_DETECTION_CLASS_NAMES)}\n"
        f"names: {UNIFIED_POLE_DETECTION_CLASS_NAMES}\n"
        f"kpt_shape: [{UNIFIED_POLE_DETECTION_NUM_KEYPOINTS}, 3]\n"
        f"flip_idx: [0]\n"
    )
    (dataset_dir / 'data.yaml').write_text(yaml_content)

    if verbose:
        print(f'\n✓ Unified pole detection dataset prepared: {processed} images ({skipped} skipped, {neg_removed} negatives removed)')
        for split, count in split_counts.items():
            print(f'  {split}: {count} images')
        print(f'  Class distribution in crops: {class_counts}')
        print(f'  data.yaml: {dataset_dir / "data.yaml"}')
        print(f'  Classes ({len(UNIFIED_POLE_DETECTION_CLASS_NAMES)}): {UNIFIED_POLE_DETECTION_CLASS_NAMES}')
        print(f'  Keypoints: {UNIFIED_POLE_DETECTION_KEYPOINT_NAMES} (YOLO pose, v=2)')
        print(f'  Bbox: {UNIFIED_POLE_DETECTION_BBOX_HEIGHT_FEET}ft×{UNIFIED_POLE_DETECTION_BBOX_WIDTH_FEET}ft (H×W)')


def prepare_keypoint_detection_dataset(photos_dir: Path, labels_dir: Path, eq_type: str, dataset_dir: Path, verbose: bool = False, workers: int = 1) -> None:
    """
    Prepare keypoint detection dataset for HRNet training.

    Args:
        photos_dir: Path to directory with photos (*.jpg)
        labels_dir: Path to directory with *_location.txt label files
        eq_type: Equipment type ('riser', 'transformer', or 'street_light')
        dataset_dir: Output directory for prepared dataset
    """
    import cv2
    from collections import Counter
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    from PIL import Image

    photos_dir = Path(photos_dir)
    labels_dir = Path(labels_dir)
    dataset_dir = Path(dataset_dir)

    # Configuration for equipment types
    config_map = {
        'riser': {
            'num_keypoints': RISER_NUM_KEYPOINTS,
            'keypoint_names': RISER_KEYPOINT_NAMES,
        },
        'transformer': {
            'num_keypoints': TRANSFORMER_NUM_KEYPOINTS,
            'keypoint_names': TRANSFORMER_KEYPOINT_NAMES,
        },
        'street_light': {
            'num_keypoints': STREET_LIGHT_NUM_KEYPOINTS,
            'keypoint_names': STREET_LIGHT_KEYPOINT_NAMES,
        },
        'secondary_drip_loop': {
            'num_keypoints': SECONDARY_DRIP_LOOP_NUM_KEYPOINTS,
            'keypoint_names': SECONDARY_DRIP_LOOP_KEYPOINT_NAMES,
        },
    }

    cfg = config_map.get(eq_type)
    if not cfg:
        raise ValueError(f"Unknown equipment type: {eq_type}")

    num_kp = cfg['num_keypoints']

    if check_dataset_complete(dataset_dir):
        if verbose:
            print(f"✓ {eq_type.upper()} keypoint dataset already prepared at {dataset_dir}")
            for split in ['train', 'val', 'test']:
                n = len(list((dataset_dir / "images" / split).glob("*.jpg")))
                print(f"  {split}: {n} crops")
        return

    # Collect all equipment instances
    instances = []
    for photo_path in sorted(photos_dir.glob("*.jpg")):
        label_path = _loc_path(labels_dir, photo_path.stem)
        if label_path is None:
            continue
        for eq in parse_equipment_with_keypoints(label_path):
            if eq['class_name'] == eq_type:
                instances.append((photo_path, label_path, eq))

    if not instances:
        if verbose:
            print(f"No {eq_type} instances found")
        return

    if verbose:
        photos = len(set(p for p, _, _ in instances))
        print(f"{eq_type.upper()} — {len(instances)} instances across {photos} photos ({num_kp} keypoint(s))")

    # Create directory structure
    for split in ['train', 'val', 'test']:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Split by photo; use manifest for consistency
    unique_photos = sorted(set(p for p, _, _ in instances))
    ppi_cache = {p.stem: load_ppi_from_label(_loc_path(labels_dir, p.stem)) for p in unique_photos}
    manifest = load_split_manifest()
    if manifest:
        pole_split_map = get_pole_split_map(manifest)
        split_map = {p: pole_split_map.get(p.stem, 'train') for p in unique_photos}
    else:
        train_photos, temp_photos = train_test_split(unique_photos, test_size=0.2, random_state=42)
        val_photos, test_photos = train_test_split(temp_photos, test_size=0.5, random_state=42)
        split_map = {p: s for ps, s in [(train_photos, 'train'), (val_photos, 'val'), (test_photos, 'test')] for p in ps}

    if verbose:
        train_count = sum(1 for s in split_map.values() if s == 'train')
        val_count = sum(1 for s in split_map.values() if s == 'val')
        test_count = sum(1 for s in split_map.values() if s == 'test')
        print(f"Split: {train_count} train / {val_count} val / {test_count} test photos")

    # Generate crops
    processed = 0
    skipped = 0
    split_counts = Counter()
    photo_instance_idx = Counter()
    instances_with_idx = []
    for photo_path, label_path, eq in instances:
        photo_instance_idx[photo_path] += 1
        instances_with_idx.append((photo_path, label_path, eq, photo_instance_idx[photo_path]))

    def _process_one_eq_kp(item):
        photo_path, label_path, eq, idx = item
        img = cv2.imread(str(photo_path))
        if img is None:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0})
        h, w = img.shape[:2]
        bbox = eq['bbox']
        if eq_type == 'riser' and eq['kp0'] is not None:
            ppi = ppi_cache.get(photo_path.stem)
            if ppi is not None:
                bbox = riser_attachment_bbox(eq['kp0'], ppi, w, h)
        x1 = max(0, int(bbox['left'] / 100 * w))
        x2 = min(w, int(bbox['right'] / 100 * w))
        y1 = max(0, int(bbox['top'] / 100 * h))
        y2 = min(h, int(bbox['bottom'] / 100 * h))
        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < 10 or crop_h < 10:
            return (0, 1, {'train': 0, 'val': 0, 'test': 0})
        crop = img[y1:y2, x1:x2]
        kp_sources = [eq.get('kp0')]
        for i in range(1, num_kp):
            kp_sources.append(eq.get(f'kp{i}'))
        kp_data = []
        for kp in kp_sources:
            if kp is not None:
                kp_x = min(max((kp[0] / 100 * w - x1) / crop_w, 0.0), 0.999999)
                kp_y = min(max((kp[1] / 100 * h - y1) / crop_h, 0.0), 0.999999)
                kp_data.append((kp_x, kp_y, 2))
            else:
                kp_data.append((0.0, 0.0, 0))
        split = split_map[photo_path]
        stem = f"{photo_path.stem}_{eq_type}{idx}"
        img_dst = dataset_dir / "images" / split / f"{stem}.jpg"
        lbl_dst = dataset_dir / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(img_dst), crop)
        label_line = "0 0.5 0.5 1.0 1.0"
        for kx, ky, kv in kp_data:
            label_line += f" {kx:.6f} {ky:.6f} {kv}"
        ppi = ppi_cache.get(photo_path.stem)
        ppi_comment = f"# PPI={ppi}\n" if ppi else ""
        with open(lbl_dst, 'w') as f:
            f.write(ppi_comment + label_line + '\n')
        _split = {'train': 0, 'val': 0, 'test': 0}
        _split[split] = 1
        return (1, 0, _split)

    results = _parallel_map(instances_with_idx, _process_one_eq_kp, workers, desc=f"Cropping {eq_type}", verbose=verbose)
    for _processed, _skipped, _split in results:
        processed += _processed
        skipped += _skipped
        for k, v in _split.items():
            split_counts[k] += v

    # Write data.yaml
    yaml_content = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names: ['{eq_type}']\n"
        f"kpt_shape: [{num_kp}, 3]\n"
    )
    with open(dataset_dir / "data.yaml", 'w') as f:
        f.write(yaml_content)

    if verbose:
        print(f"✓ {processed} crops ({skipped} skipped)")
        for split in ['train', 'val', 'test']:
            print(f"  {split}: {split_counts[split]} crops")
        print(f"  Keypoints: {cfg['keypoint_names']}")


def prepare_calibration_datasets(
    pole_photos_dir: Path, pole_labels_dir: Path,
    midspan_photos_dir: Path, midspan_labels_dir: Path,
    datasets_dir: Path = DATASETS_DIR,
    verbose: bool = False,
    workers: int = 1,
) -> None:
    """
    Prepare all calibration datasets (pole, ruler, ruler marking, pole top detection).

    Handles:
    - Train/val/test splitting (80/10/10)
    - YOLO format label generation
    - Filtering by keypoint visibility
    - Dataset completion checking
    """
    # Ensure all paths are Path objects
    pole_photos_dir = Path(pole_photos_dir)
    pole_labels_dir = Path(pole_labels_dir)
    midspan_photos_dir = Path(midspan_photos_dir)
    midspan_labels_dir = Path(midspan_labels_dir)
    datasets_dir = Path(datasets_dir)

    # Create dataset directories
    datasets = {
        'pole_detection': datasets_dir / 'pole_detection',
        'ruler_detection': datasets_dir / 'ruler_detection',
        'ruler_marking_detection': datasets_dir / 'ruler_marking_detection',
        'pole_top_detection': datasets_dir / 'pole_top_detection',
    }

    for dataset_dir in datasets.values():
        for split in ['train', 'val', 'test']:
            (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Collect images with labels
    def _has_label(labels_dir, stem):
        lp = _loc_path(labels_dir, stem)
        return lp is not None and lp.exists()
    # Role-separate the photos. Under the photo_id layout pole and midspan share data/Photos,
    # so a directory glob returns the FULL merged set for both -> the midspan split loop below
    # would clobber every pole photo's split to 'train' (pole pid absent from midspan_map). Use
    # the label store's role to keep pole_files / midspan_files DISJOINT.
    if _pil.ENABLED:
        pole_files = [Path(p) for _pid, p in _pil.iter_photos("pole")
                      if _has_label(pole_labels_dir, Path(p).stem)]
        midspan_files = [Path(p) for _pid, p in _pil.iter_photos("midspan")
                         if _has_label(midspan_labels_dir, Path(p).stem)]
    else:
        pole_files = [p for p in pole_photos_dir.glob('*.jpg')
                      if _has_label(pole_labels_dir, p.stem)]
        midspan_files = [p for p in midspan_photos_dir.glob('*.jpg')
                         if _has_label(midspan_labels_dir, p.stem)]

    photo_files = pole_files + midspan_files
    if not photo_files:
        raise RuntimeError(f'No images found (pole: {len(pole_files)}, midspan: {len(midspan_files)})')

    # Use master split manifest; pole and midspan have separate splits
    manifest = load_split_manifest()
    if manifest:
        pole_map = get_pole_split_map(manifest)
        midspan_map = get_midspan_split_map(manifest)
        split_map = {}
        for p in pole_files:
            split_map[p] = pole_map.get(p.stem, 'train')
        for p in midspan_files:
            split_map[p] = midspan_map.get(p.stem, 'train')
    else:
        train_files, temp_files = train_test_split(photo_files, test_size=0.2, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        split_map = {f: 'train' for f in train_files}
        split_map.update({f: 'val' for f in val_files})
        split_map.update({f: 'test' for f in test_files})

    completeness = {k: check_dataset_complete(v) for k, v in datasets.items()}
    if verbose:
        print(f'Dataset completion: {sum(completeness.values())}/{len(completeness)}')
    if all(completeness.values()):
        if verbose:
            print('✓ All datasets complete, skipping preparation')
        return

    # Process images
    processed = {k: 0 for k in datasets}
    skipped_total = 0

    def _process_one(photo_path):
        out = {'processed': processed.copy(), 'skipped': 0}
        out['processed'] = {k: 0 for k in datasets}
        if photo_path.resolve().is_relative_to(midspan_photos_dir.resolve()):
            label_path = _loc_path(midspan_labels_dir, photo_path.stem)
        else:
            label_path = _loc_path(pole_labels_dir, photo_path.stem)

        img = cv2.imread(str(photo_path))
        if img is None:
            return (out['processed'], 1)  # skipped

        h, w = img.shape[:2]
        pole_bbox, ruler_bbox, keypoints, ppi = parse_label_file(label_path)
        subdir = split_map[photo_path]

        # Pole detection
        if pole_bbox and not completeness['pole_detection']:
            pole_top_kp = keypoints.get('pole_top')
            added_pole = False
            if pole_top_kp:
                x1, x2, y1, y2 = [int(v/100*w) if i < 2 else int(v/100*h)
                                   for i, v in enumerate(pole_bbox)]
                if x1 < x2 and y1 < y2:
                    img_path = datasets['pole_detection'] / 'images' / subdir / photo_path.name
                    lbl_path = datasets['pole_detection'] / 'labels' / subdir / f'{photo_path.stem}.txt'
                    if not img_path.exists() or not lbl_path.exists():
                        shutil.copy(photo_path, img_path)
                        cx, cy = (pole_bbox[0] + pole_bbox[1]) / 200, (pole_bbox[2] + pole_bbox[3]) / 200
                        bw, bh = (pole_bbox[1] - pole_bbox[0]) / 100, (pole_bbox[3] - pole_bbox[2]) / 100
                        with open(lbl_path, 'w') as f:
                            f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
                        out['processed']['pole_detection'] = 1
                        added_pole = True
            if not added_pole:
                out['skipped'] += 1

        # Ruler and ruler marking detection
        if ruler_bbox and (not completeness['ruler_detection'] or not completeness['ruler_marking_detection']):
            x1, x2, y1, y2 = [int(v/100*w) if i < 2 else int(v/100*h)
                               for i, v in enumerate(ruler_bbox)]
            if x1 >= x2 or y1 >= y2:
                out['skipped'] += 1
            else:
                visible_kps = sum(1 for kp_name in KEYPOINT_NAMES
                                if kp_name in keypoints and
                                x1 <= keypoints[kp_name][0]/100*w <= x2 and
                                y1 <= keypoints[kp_name][1]/100*h <= y2)
                if visible_kps < 5:
                    out['skipped'] += 1
                else:
                    if not completeness['ruler_detection']:
                        img_path = datasets['ruler_detection'] / 'images' / subdir / photo_path.name
                        lbl_path = datasets['ruler_detection'] / 'labels' / subdir / f'{photo_path.stem}.txt'
                        if not img_path.exists() or not lbl_path.exists():
                            shutil.copy(photo_path, img_path)
                            cx, cy = (ruler_bbox[0] + ruler_bbox[1]) / 200, (ruler_bbox[2] + ruler_bbox[3]) / 200
                            bw, bh = (ruler_bbox[1] - ruler_bbox[0]) / 100, (ruler_bbox[3] - ruler_bbox[2]) / 100
                            with open(lbl_path, 'w') as f:
                                f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
                            out['processed']['ruler_detection'] = 1

                    if not completeness['ruler_marking_detection']:
                        crop = img[y1:y2, x1:x2]
                        crop_h, crop_w = crop.shape[:2]
                        if crop_h >= 10 and crop_w >= 10:
                            img_path = datasets['ruler_marking_detection'] / 'images' / subdir / photo_path.name
                            lbl_path = datasets['ruler_marking_detection'] / 'labels' / subdir / f'{photo_path.stem}.txt'
                            if not img_path.exists() or not lbl_path.exists():
                                cv2.imwrite(str(img_path), crop)
                                ppi_comment = f'# PPI={ppi:.6f}\n' if ppi and ppi > 0 else ''
                                label_content = ppi_comment + '0 0.5 0.5 1.0 1.0'
                                for kp_name in KEYPOINT_NAMES:
                                    if kp_name in keypoints:
                                        kx_px = keypoints[kp_name][0] / 100 * w
                                        ky_px = keypoints[kp_name][1] / 100 * h
                                        if x1 <= kx_px <= x2 and y1 <= ky_px <= y2:
                                            kx_norm = min(max((kx_px - x1) / crop_w, 0), 0.999999)
                                            ky_norm = min(max((ky_px - y1) / crop_h, 0), 0.999999)
                                            label_content += f' {kx_norm:.6f} {ky_norm:.6f} 2'
                                        else:
                                            label_content += ' 0.0 0.0 0'
                                    else:
                                        label_content += ' 0.0 0.0 0'
                                label_content += '\n'
                                with open(lbl_path, 'w') as f:
                                    f.write(label_content)
                                out['processed']['ruler_marking_detection'] = 1

        # Pole top detection
        if pole_bbox and not completeness['pole_top_detection']:
            pole_top_kp = keypoints.get('pole_top')
            x1, x2, y1, y2 = [int(v/100*w) if i < 2 else int(v/100*h)
                               for i, v in enumerate(pole_bbox)]
            if x1 < x2 and y1 < y2 and pole_top_kp:
                crop = img[y1:y2, x1:x2]
                crop_h, crop_w = crop.shape[:2]
                if crop_h >= 10 and crop_w >= 10:
                    kx_px = pole_top_kp[0] / 100 * w
                    ky_px = pole_top_kp[1] / 100 * h
                    kx_norm = min(max((kx_px - x1) / crop_w, 0), 0.999999)
                    ky_norm = min(max((ky_px - y1) / crop_h, 0), 0.999999)
                    img_path = datasets['pole_top_detection'] / 'images' / subdir / photo_path.name
                    lbl_path = datasets['pole_top_detection'] / 'labels' / subdir / f'{photo_path.stem}.txt'
                    if not img_path.exists() or not lbl_path.exists():
                        cv2.imwrite(str(img_path), crop)
                        ppi_comment = f'# PPI={ppi:.6f}\n' if ppi and ppi > 0 else ''
                        with open(lbl_path, 'w') as f:
                            f.write(f'{ppi_comment}0 0.5 0.5 1.0 1.0 {kx_norm:.6f} {ky_norm:.6f} 2\n')
                        out['processed']['pole_top_detection'] = 1

        return (out['processed'], out['skipped'])

    results = _parallel_map(photo_files, _process_one, workers, desc='Preparing datasets', verbose=verbose)
    for proc, skipped in results:
        for k in processed:
            processed[k] += proc.get(k, 0)
        skipped_total += skipped

    if verbose:
        for dataset_name, count in processed.items():
            if count > 0:
                print(f'  {dataset_name}: {count} images')
        if skipped_total > 0:
            print(f'  Skipped: {skipped_total} images')


def _midspan_wire_dataset_complete(
    dataset_dir: Path,
    photo_files: List[Path],
    split_map: Dict[Path, str],
) -> bool:
    """True if every expected wire-positive midspan image exists in the dataset."""
    if not dataset_dir.exists():
        return False
    for photo_path in photo_files:
        split = split_map.get(photo_path, 'train')
        stem = photo_path.stem
        if not (dataset_dir / 'images' / split / f'{stem}.jpg').exists():
            return False
        if not (dataset_dir / 'labels' / split / f'{stem}.txt').exists():
            return False
    return True


def _midspan_wire_strip_dataset_complete(
    dataset_dir: Path,
    photo_files: List[Path],
    split_map: Dict[Path, str],
) -> bool:
    """True if every expected strip sample exists in the dataset."""
    return _midspan_wire_dataset_complete(dataset_dir, photo_files, split_map)


def prepare_midspan_wire_strip_detection_dataset(
    photos_dir: Path,
    labels_dir: Path,
    dataset_dir: Optional[Path] = None,
    verbose: bool = False,
    workers: int = 1,
    width_expand: float = 1.0,
    strip_mode: str = 'column',
) -> None:
    """
    Prepare midspan wire strip + 1D heatmap dataset.

    strip_mode='column' (legacy): full-height vertical strip over the ruler bbox
    x-range (optionally widened by width_expand). Requires ruler bbox + PPI.

    strip_mode='ruler-line': rectified strip along the straight line joining the
    ruler tick anchors (2.5..16.5 ft), width = 3 ft via the projective height model's
    local scale, bottom = projected 0.0 ft ground line, top = photo top (see
    extract_ruler_line_strip). Requires the tick anchors + a projective fit; PPI and
    ruler bbox are NOT required. Use a separate dataset_dir — labels are normalized
    by GROUND-LINE height, not photo height, so the two modes are not miscible.

    Saves strip JPGs and label files with normalized wire y positions (0-1 along
    strip height). Positive-only: at least one wire required. Excludes MI jobs.

    width_expand: widen the ruler x-range about its centre (column mode only).
        1.0 = legacy. Use a separate dataset_dir per width; the matching inference crop is
        e2e --strip-width-expand.
    """
    if strip_mode not in ('column', 'ruler-line'):
        raise ValueError(f'strip_mode must be column|ruler-line, got {strip_mode!r}')
    photos_dir = Path(photos_dir)
    labels_dir = Path(labels_dir)
    dataset_dir = Path(dataset_dir or DATASET_DIRS[MIDSPAN_WIRE_STRIP_DETECTION])
    manifest = load_split_manifest()

    photo_files: List[Path] = []
    strip_cache: Dict[str, Dict] = {}
    skipped_no_label = 0
    skipped_no_ppi = 0
    skipped_no_ruler = 0
    skipped_no_wires = 0
    skipped_excluded_job = 0
    skipped_read_fail = 0

    # MI exclusion is CONTENT-based under the pid layout: drop only primary-bearing
    # MI midspan photos (collapsed multi-primary labels); keep primary-free MI photos.
    _mi_dirty = mi_photo_ids(labels_dir, role='midspan') & mi_dirty_midspan_pids()
    for photo_path in sorted(photos_dir.glob('*.jpg')):
        if midspan_job_excluded_for_wire(photo_path.name) or photo_path.stem in _mi_dirty:
            skipped_excluded_job += 1
            continue
        label_path = _loc_path(labels_dir, photo_path.stem)
        if label_path is None or not label_path.exists():
            skipped_no_label += 1
            continue
        anchor_pts = None
        fit = None
        if strip_mode == 'ruler-line':
            # projection-only gate: tick anchors + a height fit (no PPI/bbox needed)
            from src.ruler_height_model import fit_photo_height as _fph
            anchor_pts = parse_ruler_anchor_points(label_path)
            fit = _fph([(py, ft * 12.0) for ft, _px, py in anchor_pts])
            if fit is None or len(anchor_pts) < 2:
                skipped_no_ruler += 1
                continue
            _, ruler_bbox, _, ppi = parse_label_file(label_path)
        else:
            if not _label_has_ppi(label_path):
                skipped_no_ppi += 1
                continue
            _, ruler_bbox, _, ppi = parse_label_file(label_path)
            if not ruler_bbox:
                skipped_no_ruler += 1
                continue
        wire_ys = parse_wire_y_percent_from_label_file(label_path)
        if not wire_ys:
            skipped_no_wires += 1
            continue
        strip_cache[photo_path.stem] = {
            'label_path': label_path,
            'ruler_bbox': ruler_bbox,
            'wire_ys': wire_ys,
            'ppi': ppi,
            'anchor_pts': anchor_pts,
            'fit': fit,
        }
        photo_files.append(photo_path)

    if not photo_files:
        raise RuntimeError(
            'No midspan wire strip training images found '
            f'(skipped: {skipped_excluded_job} excluded job, {skipped_no_label} no label, '
            f'{skipped_no_ppi} no PPI, {skipped_no_ruler} no ruler, {skipped_no_wires} no wires)'
        )

    if manifest:
        midspan_map = get_midspan_split_map(manifest)
        split_map = {f: midspan_map.get(f.stem, 'train') for f in photo_files}
    else:
        train_files, temp_files = train_test_split(
            photo_files, test_size=0.2, random_state=SPLIT_MANIFEST_RANDOM_STATE
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=0.5, random_state=SPLIT_MANIFEST_RANDOM_STATE
        )
        split_map = {f: 'train' for f in train_files}
        split_map.update({f: 'val' for f in val_files})
        split_map.update({f: 'test' for f in test_files})

    if _midspan_wire_strip_dataset_complete(dataset_dir, photo_files, split_map):
        if verbose:
            print(f'✓ Midspan wire strip dataset already prepared at {dataset_dir}')
            for split in ['train', 'val', 'test']:
                n_img = len(list((dataset_dir / 'images' / split).glob('*.jpg')))
                n_lbl = len(list((dataset_dir / 'labels' / split).glob('*.txt')))
                print(f'  {split}: {n_img} images, {n_lbl} labels')
        return

    for split in ['train', 'val', 'test']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f'Midspan wire strip positives: {len(photo_files)} images')
        print(f'  Skipped {skipped_excluded_job} (MI-regime primary-bearing midspan / excluded job)')
        print(f'  Skipped {skipped_no_label} (no label), {skipped_no_ppi} (no PPI), '
              f'{skipped_no_ruler} (no ruler), {skipped_no_wires} (no wires)')
        train_count = sum(1 for s in split_map.values() if s == 'train')
        val_count = sum(1 for s in split_map.values() if s == 'val')
        test_count = sum(1 for s in split_map.values() if s == 'test')
        print(f'  Split: {train_count} train / {val_count} val / {test_count} test')
        total_wires = sum(len(strip_cache[s]['wire_ys']) for s in strip_cache)
        print(f'  Total unique wire heights: {total_wires}')

    def _process_one(photo_path: Path):
        split = split_map[photo_path]
        meta = strip_cache[photo_path.stem]
        img_path = dataset_dir / 'images' / split / photo_path.name
        lbl_path = dataset_dir / 'labels' / split / f'{photo_path.stem}.txt'
        if img_path.exists() and lbl_path.exists():
            return (split, len(meta['wire_ys']), 0)

        img = cv2.imread(str(photo_path))
        if img is None:
            return (split, 0, 0)
        img_h, img_w = img.shape[:2]
        if strip_mode == 'ruler-line':
            out = extract_ruler_line_strip(img, meta['anchor_pts'], meta['fit'])
            if out is None:
                return (split, 0, 0)
            strip, lmeta = out
            strip_h, strip_w = strip.shape[:2]
            # normalize wire y by GROUND-LINE height (crop bottom = 0 ft), drop below-ground
            gy = lmeta['ground_y_px']
            wire_y_norms = [y_pct / 100.0 * img_h / gy for y_pct in meta['wire_ys']
                            if y_pct / 100.0 * img_h < gy]
            if not wire_y_norms:
                return (split, 0, 0)
            cv2.imwrite(str(img_path), strip, [cv2.IMWRITE_JPEG_QUALITY, 95])
            header = [
                f'# PPI={meta["ppi"] if meta["ppi"] else 0.0:.6f}',
                f'# source={photo_path.name}',
                '# strip_mode=ruler-line',
                f'# line_m={lmeta["line_m"]:.8f} line_c={lmeta["line_c"]:.4f}',
                f'# ground_y_px={gy} width_px={lmeta["width_px"]} '
                f'px_per_inch_mid={lmeta["px_per_inch_mid"]:.6f} width_ft={lmeta["width_ft"]:.2f}',
                f'# full_h={img_h} full_w={img_w}',
                f'# strip_h={strip_h} strip_w={strip_w}',
                '# wire_y normalized 0-1 along strip height (ground-line bottom; one wire per line)',
            ]
            body = [f'{y:.8f}' for y in wire_y_norms]
            lbl_path.write_text('\n'.join(header + body) + '\n')
            return (split, len(wire_y_norms), 1)

        strip, crop_box = extract_ruler_column_strip(img, meta['ruler_bbox'], width_expand=width_expand)
        if strip.size == 0:
            return (split, 0, 0)

        cv2.imwrite(str(img_path), strip, [cv2.IMWRITE_JPEG_QUALITY, 95])
        x1, y1, x2, y2 = crop_box
        strip_h, strip_w = strip.shape[:2]
        ruler = meta['ruler_bbox']
        wire_y_norms = [y_pct / 100.0 for y_pct in meta['wire_ys']]
        header = [
            f'# PPI={meta["ppi"]:.6f}',
            f'# source={photo_path.name}',
            f'# ruler_left={ruler[0]:.4f} ruler_right={ruler[1]:.4f}',
            f'# crop_x1={x1} crop_x2={x2} full_h={img_h} full_w={img_w}',
            f'# strip_h={strip_h} strip_w={strip_w} width_expand={width_expand:.3f}',
            '# wire_y normalized 0-1 along strip height (one wire per line)',
        ]
        body = [f'{y:.8f}' for y in wire_y_norms]
        lbl_path.write_text('\n'.join(header + body) + '\n')
        return (split, len(wire_y_norms), 1)

    split_counts = {'train': 0, 'val': 0, 'test': 0}
    wire_count = 0
    processed = 0
    results = _parallel_map(
        photo_files, _process_one, workers, desc='Midspan wire strip dataset', verbose=verbose
    )
    for split, n_wires, did_write in results:
        wire_count += n_wires
        if did_write:
            split_counts[split] += 1
            processed += 1

    if verbose:
        print(f'\n✓ Midspan wire strip dataset prepared: {processed} strips ({wire_count} wire heights)')
        for split, count in split_counts.items():
            print(f'  {split}: {count} strips')
        print(f'  Output: {dataset_dir}')