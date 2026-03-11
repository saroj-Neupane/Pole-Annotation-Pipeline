#!/usr/bin/env python3
"""
Extract height measurement data (percentageX and percentageY) for job photo sets
from their corresponding JSON job files.

Use --pole or --midspan to process only one dataset. If neither flag is given, processes both.

This script orchestrates the extraction pipeline:
1. Parses complex JSON job files to extract measurements
2. Calculates ground and ruler coordinates using interpolation
3. Creates bounding boxes for poles, rulers, and equipment
4. Generates location files documenting all measurements

Wire/Attachment Processing:
- Direct wire markers: photofirst_data.wire
- Insulator-attached wires: photofirst_data.insulator._children.wire (e.g., COAR jobs)

Note: Excludes proposed infrastructure (traces with proposed=True field).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, DefaultDict

from PIL import Image

from src.height_calculations import (
    calculate_ground_coordinates,
    calculate_ruler_top_coordinates,
    calculate_ppi_from_measurements,
)
from src.config import DOWN_GUY_BBOX_HEIGHT_FEET, DOWN_GUY_BBOX_WIDTH_FEET
from src.bounding_boxes import (
    calculate_attachment_bounding_box,
    calculate_bounding_box,
    calculate_ruler_bounding_box,
    calculate_riser_bounding_box,
    calculate_secondary_drip_loop_bounding_box,
    calculate_transformer_bounding_box,
    calculate_street_light_bounding_box,
)


def extract_all_json_data(json_file_path: str) -> Tuple[Dict, Dict, Dict]:
    """Load JSON once and extract all needed data: locations, photo mappings, and photo data."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scid_locations = {}
    scid_photos = {}

    # Extract SCID locations and photo mappings from nodes
    nodes = data.get('nodes', {})
    for node_id, node_data in nodes.items():
        scid_data = node_data.get('attributes', {}).get('scid', {})
        scid = scid_data.get('auto_button', '')
        if not scid:
            continue

        # Get photos
        photos = node_data.get('photos', {})
        scid_photos[scid] = list(photos.keys())

    # Extract SCID locations from connections
    connections = data.get('connections', {})
    for connection_id, connection_data in connections.items():
        scid_data = connection_data.get('attributes', {}).get('scid', {})
        scid = scid_data.get('auto_button', '')
        if not scid:
            continue

        sections = connection_data.get('sections', {})
        if sections:
            first_section = list(sections.values())[0]
            latitude = first_section.get('latitude')
            longitude = first_section.get('longitude')
            if latitude is not None and longitude is not None:
                scid_locations[scid] = {'latitude': latitude, 'longitude': longitude}

    # Extract photo data
    photo_data = {}
    photos = data.get('photos', {})

    for photo_id, photo_info in photos.items():
        filename = photo_info.get('filename', '')
        if not filename or not filename.startswith('IMG_'):
            continue

        photofirst_data = photo_info.get('photofirst_data', {})
        if not photofirst_data:
            continue

        height_measurements = {}

        # Load trace_data once at the beginning (needed for equipment and wire processing)
        trace_data = (data.get('traces') or {}).get('trace_data', {}) or {}

        # Process anchor calibration measurements
        anchor_calibration = photofirst_data.get('anchor_calibration', {})
        for measurement_id, measurement_data in anchor_calibration.items():
            height = measurement_data.get('height')
            pixel_selection = measurement_data.get('pixel_selection', [])

            if height is not None and pixel_selection:
                if isinstance(height, str):
                    try:
                        height = float(height)
                    except (ValueError, TypeError):
                        continue

                pixel = pixel_selection[0]
                px, py = pixel.get('percentX'), pixel.get('percentY')
                if px is not None and py is not None:
                    height_measurements[height] = {'percentX': px, 'percentY': py}

        # Process pole top measurements
        pole_top = photofirst_data.get('pole_top', {})
        for pole_top_id, pole_top_data in pole_top.items():
            measured_height = pole_top_data.get('_measured_height')
            pixel_selection = pole_top_data.get('pixel_selection', [])

            if measured_height is not None and pixel_selection:
                pixel = pixel_selection[0]
                px, py = pixel.get('percentX'), pixel.get('percentY')
                if px is not None and py is not None:
                    height_measurements[f"pole_top_{measured_height}"] = {'percentX': px, 'percentY': py}

        # Process equipment data
        equipment = photofirst_data.get('equipment', {})
        routine_groups: Dict[str, List[Dict]] = defaultdict(list)
        secondary_drip_loop_list: List[Dict] = []

        def _street_light_role(et: str, measurement_of: str) -> Optional[str]:
            """Map equipment_type + measurement_of to street_light role. Returns 'upper', 'lower', or None."""
            mo_lower = (measurement_of or '').strip().lower()
            if mo_lower == 'top_of_bracket':
                return 'upper'
            if mo_lower == 'bottom_of_bracket':
                return 'lower'
            et_lower = (et or '').strip().lower()
            if et_lower in ('street_light', 'street_light_sg', 'lower street light bracket', 'street light'):
                return 'lower'
            if 'upper' in et_lower and 'street' in et_lower and 'light' in et_lower:
                return 'upper'
            return None

        def _is_street_light_type(et: str) -> bool:
            et_lower = (et or '').strip().lower()
            return (et_lower in ('street_light', 'street_light_sg', 'street light') or
                    ('street' in et_lower and 'light' in et_lower and ('upper' in et_lower or 'bracket' in et_lower)))

        for equipment_id, equipment_data in equipment.items():
            equipment_type = equipment_data.get('equipment_type', '')
            if not equipment_type:
                continue

            # Exclude proposed equipment (not existing infrastructure)
            trace_id = equipment_data.get('_trace')
            if trace_id and trace_id in trace_data:
                trace = trace_data.get(trace_id, {})
                if trace.get('proposed') == True:
                    continue

            pixel_selection = equipment_data.get('pixel_selection', [])
            if not pixel_selection:
                continue
            pixel = pixel_selection[0]
            if pixel.get('percentX') is None or pixel.get('percentY') is None:
                continue

            coords = {'percentX': pixel['percentX'], 'percentY': pixel['percentY']}
            routine_id = equipment_data.get('_routine_instance_id') or equipment_id

            et_lower = equipment_type.strip().lower()
            if et_lower == 'riser':
                routine_groups[routine_id].append({'category': 'riser', 'coords': coords})
            elif et_lower == 'transformer':
                measurement_of = equipment_data.get('measurement_of', '')
                role = 'top' if (measurement_of or '').strip().lower() == 'top_bolt' else 'bottom'
                routine_groups[routine_id].append({
                    'category': 'transformer',
                    'role': role,
                    'coords': coords,
                })
            elif _is_street_light_type(equipment_type):
                measurement_of = equipment_data.get('measurement_of', '')
                sl_role = _street_light_role(equipment_type, measurement_of)
                if sl_role:
                    routine_groups[routine_id].append({
                        'category': 'street_light',
                        'role': sl_role,
                        'coords': coords,
                    })
            elif et_lower in ('drip_loop', 'drip loop', 'drip_loop_sg'):
                drip_spec = (equipment_data.get('drip_loop_spec') or '').strip().lower()
                if drip_spec == 'secondary':
                    secondary_drip_loop_list.append(coords)
                elif drip_spec == 'street light':
                    # Street light drip loop: add to same routine as street_light (upper/lower)
                    routine_groups[routine_id].append({
                        'category': 'street_light',
                        'role': 'drip_loop',
                        'coords': coords,
                    })

        # Secondary drip loop: flat list, sorted top-to-bottom by percentY (like comm/down_guy)
        for idx, coords in enumerate(sorted(secondary_drip_loop_list, key=lambda c: c['percentY']), 1):
            height_measurements[f'secondary_drip_loop{idx}'] = coords

        # Assign sequential indices per category
        category_counters: DefaultDict[str, int] = defaultdict(int)
        for routine_id in sorted(routine_groups.keys()):
            entries = routine_groups[routine_id]
            if not entries:
                continue
            category = entries[0]['category']
            category_counters[category] += 1
            idx = category_counters[category]

            if category == 'riser':
                height_measurements[f'riser{idx}'] = entries[0]['coords']
            elif category == 'transformer':
                for entry in entries:
                    height_measurements[f'transformer{idx}_{entry["role"]}'] = entry['coords']
            elif category == 'street_light':
                for entry in entries:
                    height_measurements[f'street_light{idx}_{entry["role"]}'] = entry['coords']
            # secondary_drip_loop already assigned above (flat list, sorted by percentY)

        # Process wire (attachments): comm (Catv+Telco+Fiber merged), Down Guy, power (primary/secondary/neutral/open_secondary)
        # trace_data already loaded at the beginning
        wire = photofirst_data.get('wire', {}) or {}
        seen_trace_per_category: DefaultDict[str, set] = defaultdict(set)
        comm_list: List[Dict] = []
        down_guy_list: List[Dict] = []
        primary_list: List[Dict] = []
        secondary_list: List[Dict] = []
        neutral_list: List[Dict] = []  # includes open_secondary
        guy_list: List[Dict] = []     # includes power_guy

        # Power wire types; priority for mixed arm: primary > secondary > neutral (neutral includes open_secondary)

        def _attachment_category(trace: dict) -> Optional[str]:
            """Map trace to attachment category: comm (catv/telco/fiber) or down_guy.

            Patterns (from wire + trace_data):
            - comm: Catv, Telco, Fiber (cable_type or company)
            - Down Guy: _trace_type == down_guy
            - Excludes: Power/primary/secondary, proposed infrastructure (proposed=True)
            """
            tt = (trace.get('_trace_type') or '').strip().lower()
            ct = (trace.get('cable_type') or '').strip().lower()
            company = (trace.get('company') or '').strip().lower()
            
            # Exclude proposed infrastructure
            if trace.get('proposed') == True:
                return None
            
            if tt == 'down_guy':
                return 'down_guy'
            # Exclude power/primary/secondary - they sit at pole top and can be misclassified as comm
            if tt in ('primary', 'power', 'secondary', 'neutral'):
                return None
            if 'power' in ct or 'primary' in ct or 'electric' in ct or 'neutral' in ct:
                return None
            if 'power' in company:
                return None
            if 'catv' in ct or 'telco' in ct or 'telephone' in company or 'fiber' in ct:
                return 'comm'
            return None

        def _power_attachment_category(trace: dict) -> Optional[str]:
            """Map trace to power/guy category. neutral includes open_secondary; guy includes power_guy."""
            if trace.get('proposed') == True:
                return None
            ct = (trace.get('cable_type') or '').strip().lower()
            if ct == 'power guy' or ct == 'guy':
                return 'guy'
            if ct == 'primary':
                return 'primary'
            if ct == 'secondary':
                return 'secondary'
            if ct == 'open secondary' or ct == 'neutral':
                return 'neutral'
            return None

        def _resolve_arm_mixed_type(wire_types: set) -> str:
            """For mixed wire types on one arm, pick highest priority: primary > secondary > neutral > guy."""
            for cat in ('primary', 'secondary', 'neutral', 'guy'):
                if cat in wire_types:
                    return cat
            return 'neutral'  # fallback

        def _add_power_coords(cat: str, coords: Dict, seen: DefaultDict[str, set], trace_id: str) -> None:
            """Add coords to the appropriate power/guy list if not seen."""
            if trace_id in seen[cat]:
                return
            seen[cat].add(trace_id)
            if cat == 'primary':
                primary_list.append(coords)
            elif cat == 'secondary':
                secondary_list.append(coords)
            elif cat == 'neutral':
                neutral_list.append(coords)
            elif cat == 'guy':
                guy_list.append(coords)

        def _is_down_guy_guying(guying_type_val: str) -> bool:
            """Match guying_type: 'down guy', 'down_guy', or 'sidewalk brace' (sidewalk guy)."""
            g = (guying_type_val or '').strip().lower()
            return g in ('down guy', 'down_guy', 'sidewalk brace', 'sidewalk guy', 'sidewalk_guy')

        # Process wire markers (direct in photofirst_data.wire)
        for _mid, mdata in wire.items():
            ps = mdata.get('pixel_selection', [])
            trace_id = mdata.get('_trace')
            if not ps or not trace_id:
                continue
            px = ps[0].get('percentX')
            py = ps[0].get('percentY')
            if px is None or py is None:
                continue
            trace = trace_data.get(trace_id, {})
            cat = _attachment_category(trace)
            if cat:
                if trace_id in seen_trace_per_category[cat]:
                    continue
                seen_trace_per_category[cat].add(trace_id)
                coords = {'percentX': px, 'percentY': py}
                if cat == 'comm':
                    comm_list.append(coords)
                elif cat == 'down_guy':
                    down_guy_list.append(coords)
            else:
                pcat = _power_attachment_category(trace)
                if pcat:
                    coords = {'percentX': px, 'percentY': py}
                    _add_power_coords(pcat, coords, seen_trace_per_category, trace_id)

        # Process wires attached to insulators (e.g., COAR jobs)
        # In some jobs (COAR), wires sit on insulators and are stored as children
        # Insulator wires don't have their own pixel_selection - use the insulator's coordinates
        insulators = photofirst_data.get('insulator', {}) or {}
        for ins_id, ins_data in insulators.items():
            # Get insulator coordinates
            ins_ps = ins_data.get('pixel_selection', [])
            if not ins_ps:
                continue
            ins_px = ins_ps[0].get('percentX')
            ins_py = ins_ps[0].get('percentY')
            if ins_px is None or ins_py is None:
                continue
            
            # Check wires attached to this insulator
            children = ins_data.get('_children', {}) or {}
            insulator_wires = children.get('wire', {}) or {}
            
            for wire_id, wire_data in insulator_wires.items():
                trace_id = wire_data.get('_trace')
                if not trace_id:
                    continue
                trace = trace_data.get(trace_id, {})
                cat = _attachment_category(trace)
                if cat:
                    if trace_id in seen_trace_per_category[cat]:
                        continue
                    seen_trace_per_category[cat].add(trace_id)
                    coords = {'percentX': ins_px, 'percentY': ins_py}
                    if cat == 'comm':
                        comm_list.append(coords)
                    elif cat == 'down_guy':
                        down_guy_list.append(coords)
                else:
                    pcat = _power_attachment_category(trace)
                    if pcat:
                        coords = {'percentX': ins_px, 'percentY': ins_py}
                        _add_power_coords(pcat, coords, seen_trace_per_category, trace_id)

        # Process arms (crossarm, alley arm): one label per arm at arm attachment point; mixed types use priority
        arms = photofirst_data.get('arm', {}) or {}
        for arm_id, arm_data in arms.items():
            arm_ps = arm_data.get('pixel_selection', [])
            if not arm_ps:
                continue
            arm_px = arm_ps[0].get('percentX')
            arm_py = arm_ps[0].get('percentY')
            if arm_px is None or arm_py is None:
                continue
            children = arm_data.get('_children', {}) or {}
            insulators = children.get('insulator', {}) or {}
            arm_wire_types: set = set()
            for ins_id, ins_data in insulators.items():
                i_children = ins_data.get('_children', {}) or {}
                wires = i_children.get('wire', {}) or {}
                for w_id, w_data in wires.items():
                    tid = w_data.get('_trace')
                    if not tid or tid not in trace_data:
                        continue
                    pcat = _power_attachment_category(trace_data[tid])
                    if pcat:
                        arm_wire_types.add(pcat)
            if not arm_wire_types:
                continue
            resolved = _resolve_arm_mixed_type(arm_wire_types)
            coords = {'percentX': arm_px, 'percentY': arm_py}
            # Use arm_id as dedup key (one per arm)
            seen_key = f'arm_{arm_id}'
            if seen_key in seen_trace_per_category[resolved]:
                continue
            seen_trace_per_category[resolved].add(seen_key)
            if resolved == 'primary':
                primary_list.append(coords)
            elif resolved == 'secondary':
                secondary_list.append(coords)
            elif resolved == 'open_secondary':
                open_secondary_list.append(coords)
            elif resolved == 'neutral':
                neutral_list.append(coords)

        # Process guying (Down Guy) - separate from wire, in photofirst_data.guying
        guying = photofirst_data.get('guying', {}) or {}
        for _gid, gdata in guying.items():
            if not _is_down_guy_guying(gdata.get('guying_type', '')):
                continue
            ps = gdata.get('pixel_selection', [])
            trace_id = gdata.get('_trace')
            if not ps or not trace_id:
                continue
            px = ps[0].get('percentX')
            py = ps[0].get('percentY')
            if px is None or py is None:
                continue
            if trace_id in seen_trace_per_category['down_guy']:
                continue
            seen_trace_per_category['down_guy'].add(trace_id)
            down_guy_list.append({'percentX': px, 'percentY': py})

        # Sort top-to-bottom (ascending percentY) and label comm1, comm2, down_guy1, primary1, secondary1, etc.
        for idx, coords in enumerate(sorted(comm_list, key=lambda c: c['percentY']), 1):
            height_measurements[f'comm{idx}'] = coords
        for idx, coords in enumerate(sorted(down_guy_list, key=lambda c: c['percentY']), 1):
            height_measurements[f'down_guy{idx}'] = coords
        for idx, coords in enumerate(sorted(primary_list, key=lambda c: c['percentY']), 1):
            height_measurements[f'primary{idx}'] = coords
        for idx, coords in enumerate(sorted(secondary_list, key=lambda c: c['percentY']), 1):
            height_measurements[f'secondary{idx}'] = coords
        for idx, coords in enumerate(sorted(neutral_list, key=lambda c: c['percentY']), 1):
            height_measurements[f'neutral{idx}'] = coords
        for idx, coords in enumerate(sorted(guy_list, key=lambda c: c['percentY']), 1):
            height_measurements[f'guy{idx}'] = coords

        if height_measurements:
            photo_data[photo_id] = {
                'filename': filename,
                'height_measurements': height_measurements,
            }

    return scid_locations, scid_photos, photo_data


def create_location_files(job_id: str, scid_locations: Dict, scid_photos: Dict, photo_data: Dict, photos_dir: Path, labels_dir: Path) -> Dict[str, int]:
    """Create location files for each job photo.

    Only processes *_1_Main.jpg files. Matches disk photos to JSON photos with calibration data.
    
    Matching strategy:
    - If exactly 1 JSON photo has calibration data: use it (unambiguous match)
    - If multiple JSON photos have calibration data: SKIP (cannot determine correct match)
    - If no JSON photos have calibration data: SKIP (no data available)
    
    Ensures at most one label file per (job_id, scid).
    
    Returns:
        Dict with counts: {
            'processed': int,
            'skipped_multiple_cal': int,
            'skipped_no_cal': int
        }
    """
    labels_dir.mkdir(parents=True, exist_ok=True)

    photo_files = []
    if photos_dir.exists():
        job_prefix = f"{job_id}_"
        # Only _1_Main.jpg: ensures 1:1 mapping, one label per SCID
        photo_files = [
            f for f in photos_dir.iterdir()
            if f.name.startswith(job_prefix) and f.suffix == '.jpg' and f.name.endswith('_1_Main.jpg')
        ]

    if not photo_files:
        return {'processed': 0, 'skipped_multiple_cal': 0, 'skipped_no_cal': 0}

    # Track (job_id, scid) to enforce one label per SCID
    seen_scids = set()

    # Cache for image dimensions to avoid reloading same images
    img_dims_cache = {}

    # Preload all image dimensions at once (PIL reads headers only, faster than cv2.imread)
    for photo_path in photo_files:
        try:
            with Image.open(photo_path) as img:
                w, h = img.size
                img_dims_cache[photo_path.name] = (w, h)
        except Exception:
            pass

    processed_count = 0
    skipped_multiple_calibrations = 0
    skipped_no_calibration = 0

    for photo_path in photo_files:
        photo_file = photo_path.name
        # Extract SCID and index from filename: {job_id}_{scid}_{idx}_Main.jpg
        match = re.search(rf'{re.escape(job_id)}_(.+?)_(\d+)_Main\.jpg', photo_file)
        if not match:
            continue

        scid = match.group(1)
        photo_index = int(match.group(2))  # 1 for _1_Main.jpg

        # One label per SCID
        if scid in seen_scids:
            continue
        seen_scids.add(scid)

        photo_uuids = scid_photos.get(scid, [])

        # Find photos with calibration data
        candidates_with_data = [uid for uid in photo_uuids if uid in photo_data]
        
        if not candidates_with_data:
            # No photos with calibration data - skip this SCID
            skipped_no_calibration += 1
            continue
        
        if len(candidates_with_data) > 1:
            # Multiple photos with calibration data - ambiguous match, skip this SCID
            skipped_multiple_calibrations += 1
            continue
        
        # Exactly one photo with calibration data - use it (unambiguous match)
        chosen_uuid = candidates_with_data[0]
        height_measurements = photo_data[chosen_uuid]['height_measurements']
        if not height_measurements:
            continue

        # Calculate coordinates and bounding boxes
        ground_coords = calculate_ground_coordinates(height_measurements)
        ruler_top_coords = calculate_ruler_top_coordinates(height_measurements)
        bounding_box = calculate_bounding_box(height_measurements)
        ruler_bounding_box = calculate_ruler_bounding_box(height_measurements)

        # Calculate PPI and equipment bboxes if image dims available
        ppi = None
        equipment_bboxes = {}
        if photo_file in img_dims_cache:
            w, h = img_dims_cache[photo_file]
            ppi = calculate_ppi_from_measurements(height_measurements, h)

            if ppi and ppi > 0:
                # Riser bboxes
                for key, coords in height_measurements.items():
                    if isinstance(key, str) and re.match(r'^riser\d+$', key):
                        bbox = calculate_riser_bounding_box(coords, ground_coords, ppi, w, h)
                        if bbox:
                            equipment_bboxes[f'{key}_bbox'] = bbox

                # Transformer bboxes
                transformer_indices = set()
                for key in height_measurements:
                    m = re.match(r'^transformer(\d+)_', str(key))
                    if m:
                        transformer_indices.add(int(m.group(1)))
                for idx in sorted(transformer_indices):
                    top_c = height_measurements.get(f'transformer{idx}_top')
                    bot_c = height_measurements.get(f'transformer{idx}_bottom')
                    if top_c and bot_c:
                        bbox = calculate_transformer_bounding_box(top_c, bot_c, ppi, w, h)
                        if bbox:
                            equipment_bboxes[f'transformer{idx}_bbox'] = bbox

                # Street light bboxes
                sl_indices = set()
                for key in height_measurements:
                    m = re.match(r'^street_light(\d+)_', str(key))
                    if m:
                        sl_indices.add(int(m.group(1)))
                for idx in sorted(sl_indices):
                    upper_c = height_measurements.get(f'street_light{idx}_upper')
                    lower_c = height_measurements.get(f'street_light{idx}_lower')
                    drip_c = height_measurements.get(f'street_light{idx}_drip_loop')
                    if upper_c or lower_c:
                        bbox = calculate_street_light_bounding_box(upper_c, lower_c, ppi, w, h, drip_loop_coords=drip_c)
                        if bbox:
                            equipment_bboxes[f'street_light{idx}_bbox'] = bbox

                # Secondary drip loop bboxes
                sdl_keys = sorted((k for k in height_measurements if isinstance(k, str) and re.match(r'^secondary_drip_loop\d+$', k)), key=lambda k: int(re.search(r'\d+', k).group()))
                for key in sdl_keys:
                    coords = height_measurements[key]
                    bbox = calculate_secondary_drip_loop_bounding_box(coords, ppi, w, h)
                    if bbox:
                        equipment_bboxes[f'{key}_bbox'] = bbox

                # Attachment bboxes: comm/power/guy 1ft×2ft, down_guy 4ft×2ft, keypoint at center
                for prefix in ('comm', 'down_guy', 'primary', 'secondary', 'neutral', 'guy'):
                    keys = sorted(k for k in height_measurements if isinstance(k, str) and re.match(rf'^{re.escape(prefix)}\d+$', k))
                    for key in keys:
                        coords = height_measurements[key]
                        if prefix == 'down_guy':
                            bbox = calculate_attachment_bounding_box(
                                coords, ppi, w, h,
                                height_feet=DOWN_GUY_BBOX_HEIGHT_FEET,
                                width_feet=DOWN_GUY_BBOX_WIDTH_FEET,
                            )
                        else:
                            bbox = calculate_attachment_bounding_box(coords, ppi, w, h)
                        if bbox:
                            equipment_bboxes[f'{key}_bbox'] = bbox

        # Write location file
        location_file = labels_dir / f"{photo_path.stem}_location.txt"

        with open(location_file, 'w', encoding='utf-8') as f:
            f.write("# Height measurements (feet) - Percentage coordinates\n")
            f.write("# Height,PercentX,PercentY\n")

            if ground_coords:
                f.write(f"0.0,{ground_coords['percentX']},{ground_coords['percentY']}\n")
            else:
                f.write("0.0,,,\n")

            if ruler_top_coords:
                f.write(f"17.0,{ruler_top_coords['percentX']},{ruler_top_coords['percentY']}\n")
            else:
                f.write("17.0,,,\n")

            for height in [2.5, 6.5, 10.5, 14.5, 16.5]:
                if height in height_measurements:
                    m = height_measurements[height]
                    f.write(f"{height},{m['percentX']},{m['percentY']}\n")
                else:
                    f.write(f"{height},,,\n")

            # Write pole top measurements
            for key, measurement in height_measurements.items():
                if str(key).startswith('pole_top_'):
                    f.write(f"pole_top,{measurement['percentX']},{measurement['percentY']}\n")

            # Write equipment measurements
            for prefix, header in [('riser', 'Riser'), ('transformer', 'Transformer'), ('street_light', 'Street light'), ('secondary_drip_loop', 'Secondary drip loop')]:
                keys = sorted(k for k in height_measurements if isinstance(k, str) and k.startswith(prefix))
                if keys:
                    f.write(f"\n# {header} measurements (percentage coordinates)\n")
                    f.write("# Measurement,PercentX,PercentY\n")
                    for key in keys:
                        m = height_measurements[key]
                        f.write(f"{key},{m['percentX']},{m['percentY']}\n")

            # Write attachment measurements (comm, down_guy, primary, secondary, neutral, guy)
            attach_prefixes = ('comm', 'down_guy', 'primary', 'secondary', 'neutral', 'guy')
            attach_keys = sorted(k for k in height_measurements if isinstance(k, str) and any(k.startswith(p) and k[len(p):].isdigit() for p in attach_prefixes))
            if attach_keys:
                f.write("\n# Attachment measurements (comm, down_guy, primary, secondary, neutral, guy)\n")
                f.write("# Measurement,PercentX,PercentY\n")
                for key in attach_keys:
                    m = height_measurements[key]
                    f.write(f"{key},{m['percentX']},{m['percentY']}\n")

            # Write equipment bounding boxes
            if equipment_bboxes:
                f.write("\n# Equipment bounding boxes (percentage coordinates)\n")
                f.write("# Name,Left,Right,Top,Bottom\n")
                for bbox_name in sorted(equipment_bboxes.keys()):
                    bb = equipment_bboxes[bbox_name]
                    f.write(f"{bbox_name},{bb['left']:.2f},{bb['right']:.2f},{bb['top']:.2f},{bb['bottom']:.2f}\n")

            # Write bounding boxes
            if bounding_box:
                f.write(f"\n# Pole bounding box (percentage coordinates)\n")
                f.write(f"# {bounding_box['left']:.2f},{bounding_box['right']:.2f},{bounding_box['top']:.2f},{bounding_box['bottom']:.2f},{bounding_box['center_x']:.2f},{bounding_box['width']:.2f},{bounding_box['height']:.2f}\n")

            if ruler_bounding_box:
                f.write(f"\n# Ruler bounding box (percentage coordinates)\n")
                f.write(f"# {ruler_bounding_box['left']:.2f},{ruler_bounding_box['right']:.2f},{ruler_bounding_box['top']:.2f},{ruler_bounding_box['bottom']:.2f},{ruler_bounding_box['center_x']:.2f},{ruler_bounding_box['width']:.2f},{ruler_bounding_box['height']:.2f}\n")

            if ppi:
                f.write(f"\n# PPI={ppi:.6f}\n")

        processed_count += 1

    # Return statistics (quiet during processing)
    return {
        'processed': processed_count,
        'skipped_multiple_cal': skipped_multiple_calibrations,
        'skipped_no_cal': skipped_no_calibration
    }


def process_job(json_file: Path, photos_dir: Path, labels_dir: Path, quiet: bool = False) -> Dict[str, int]:
    """Process a single job JSON and create location files.
    
    Returns:
        Dict with counts for this job
    """
    job_id = json_file.stem

    if not json_file.exists():
        if not quiet:
            print(f"Error: JSON file not found: {json_file}", flush=True)
        return {'processed': 0, 'skipped_multiple_cal': 0, 'skipped_no_cal': 0}

    # Extract all data in one pass
    scid_locations, scid_photos, photo_data = extract_all_json_data(str(json_file))

    # Create location files and return statistics
    stats = create_location_files(
        job_id=job_id,
        scid_locations=scid_locations,
        scid_photos=scid_photos,
        photo_data=photo_data,
        photos_dir=photos_dir,
        labels_dir=labels_dir,
    )
    
    if not quiet:
        print(f"Created {stats['processed']} location files for {job_id}", flush=True)
    
    return stats


def _process_pole_job_worker(args: Tuple) -> Tuple[str, Dict[str, int]]:
    """Worker for parallel pole processing. Returns (job_id, stats). Top-level for pickling."""
    json_file, photos_dir, labels_dir = args
    stats = process_job(json_file, photos_dir, labels_dir, quiet=True)
    return (json_file.stem, stats)


def _process_midspan_job_worker(args: Tuple) -> Tuple[str, Dict[str, int]]:
    """Worker for parallel midspan processing. Returns (job_id, stats). Top-level for pickling."""
    job_id, data_dir, photos_dir, labels_dir = args
    json_file = Path(data_dir) / f"{job_id}.json"
    if not json_file.exists():
        return (job_id, {'processed': 0})
    photo_data = extract_midspan_photo_data(str(json_file))
    stats = create_midspan_location_files(
        job_id=job_id,
        photo_data=photo_data,
        photos_dir=Path(photos_dir),
        labels_dir=Path(labels_dir),
    )
    return (job_id, stats or {'processed': 0})


def extract_midspan_photo_data(json_file_path: str) -> Dict:
    """Extract midspan photo data including filenames and height measurements from sections.

    Returns a dict with:
    - photo_id -> height_measurements for each photo
    - '_ordered_photos': list of photo_ids sorted by date_taken
    - '_node_map': mapping of (node_1, node_2) -> list of photo_ids
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    photo_data = {}
    connections = data.get('connections', {})
    nodes = data.get('nodes', {})

    # Build node-based mapping for fallback matching
    node_map = {}

    # Build ordered list of photos with height data
    photos_with_metadata = []

    # Extract photos from sections
    for connection_id, connection_data in connections.items():
        node_1_id = connection_data.get('node_id_1', '')
        node_2_id = connection_data.get('node_id_2', '')

        # Get SCIDs for nodes
        node_1_attrs = nodes.get(node_1_id, {}).get('attributes', {}) if node_1_id in nodes else {}
        node_2_attrs = nodes.get(node_2_id, {}).get('attributes', {}) if node_2_id in nodes else {}

        node_1_scid = node_1_attrs.get('scid', {}).get('auto_button', '') if isinstance(node_1_attrs.get('scid'), dict) else ''
        node_2_scid = node_2_attrs.get('scid', {}).get('auto_button', '') if isinstance(node_2_attrs.get('scid'), dict) else ''

        sections = connection_data.get('sections', {})
        for section_id, section_data in sections.items():
            section_photos = section_data.get('photos', {})

            # Process photos in this section
            for photo_id in section_photos:
                if photo_id not in data.get('photos', {}):
                    continue

                photo_info = data['photos'][photo_id]
                photofirst_data = photo_info.get('photofirst_data', {})

                if not photofirst_data:
                    continue

                # Extract anchor calibration data (height measurements)
                anchor_calibration = photofirst_data.get('anchor_calibration', {})

                height_measurements = {}

                # Process anchor calibration measurements
                for measurement_id, measurement_data in anchor_calibration.items():
                    height = measurement_data.get('height')
                    pixel_selection = measurement_data.get('pixel_selection', [])

                    if height is not None and pixel_selection:
                        if isinstance(height, str):
                            try:
                                height = float(height)
                            except (ValueError, TypeError):
                                continue

                        pixel = pixel_selection[0]
                        px, py = pixel.get('percentX'), pixel.get('percentY')
                        if px is not None and py is not None:
                            height_measurements[height] = {'percentX': px, 'percentY': py}

                # Only store photos that have height measurements
                if height_measurements:
                    photo_data[photo_id] = {
                        'height_measurements': height_measurements,
                    }
                    # Store with metadata for sorting
                    photos_with_metadata.append({
                        'photo_id': photo_id,
                        'date_taken': photo_info.get('date_taken', 0),
                        'filename': photo_info.get('filename', ''),
                        'section_order': len(photos_with_metadata)
                    })

                    # Build node map for fallback matching
                    key1 = (node_1_scid or '', node_2_scid or '')
                    key2 = (node_2_scid or '', node_1_scid or '')

                    for key in [key1, key2]:
                        if key not in node_map:
                            node_map[key] = []
                        node_map[key].append(photo_id)

    # Sort by date_taken, fall back to section order
    photos_with_metadata.sort(key=lambda x: (x['date_taken'] if x['date_taken'] > 0 else float('inf'), x['section_order']))

    ordered_photos = [p['photo_id'] for p in photos_with_metadata]
    photo_data['_ordered_photos'] = ordered_photos
    photo_data['_node_map'] = node_map

    return photo_data


def extract_job_id_from_photo_filename(filename: str) -> Optional[str]:
    """Extract job ID from photo filename.

    Examples:
    - COAR-FR01 - 3_(135)-to-(134)_Midspan_Height_1.jpg -> COAR-FR01 - 3
    - MIFN021_(001)-to-(002)_Midspan_Height_5.jpg -> MIFN021
    """
    match = re.match(r'^([^_]+(?:\s*-\s*\d+)?)', filename)
    if match:
        return match.group(1).strip()
    return None


def create_midspan_location_files(job_id: str, photo_data: Dict, photos_dir: Path, labels_dir: Path) -> Dict[str, int]:
    """Create location files for each midspan photo using index-based matching.

    Only processes *_Midspan_Height_N.jpg files. For each node pair, files are sorted
    by the trailing number N. Photo at index i gets labels from the i-th JSON photo
    for that node pair (same logic as pole: N_th photo gets N_th label).
    
    Returns:
        Dict with counts: {'processed': int}
    """
    labels_dir.mkdir(parents=True, exist_ok=True)

    node_map = photo_data.pop('_node_map', {})

    photo_files = []
    if photos_dir.exists():
        job_prefix = f"{job_id}_"
        # Only _Midspan_Height_N.jpg (same logic as pole: match by index)
        photo_files = [
            f for f in photos_dir.iterdir()
            if f.name.startswith(job_prefix) and '_Midspan_Height_' in f.name and f.suffix == '.jpg'
        ]

    if not photo_files:
        return {'processed': 0}

    # Cache image dimensions (PIL reads headers only, faster than cv2.imread)
    img_dims_cache = {}
    for photo_path in photo_files:
        try:
            with Image.open(photo_path) as img:
                w, h = img.size
                img_dims_cache[photo_path.name] = (w, h)
        except Exception:
            pass

    # Group files by node pair
    files_by_node_pair = defaultdict(list)
    for photo_path in photo_files:
        photo_file = photo_path.name
        node_match = re.search(r'\(([^)]*)\)-to-\(([^)]*)\)', photo_file)
        if node_match:
            from_node = (node_match.group(1) or '').strip()
            to_node = (node_match.group(2) or '').strip()

            if not from_node or not to_node:
                continue

            key = tuple(sorted([from_node, to_node]))
            files_by_node_pair[key].append((photo_path, photo_file))

    # Count JSON photos per node pair
    json_by_node_pair = defaultdict(list)
    for key, photo_ids in node_map.items():
        node1, node2 = key
        if not node1 or not node2:
            continue
        normalized_key = tuple(sorted([node1, node2]))
        json_by_node_pair[normalized_key].extend(photo_ids)

    # Dedupe while preserving order (same logic as pole: index N -> use JSON photo at index N)
    for key in json_by_node_pair:
        json_by_node_pair[key] = list(dict.fromkeys(json_by_node_pair[key]))

    processed_count = 0

    # Process all photos: match by index. File at index i gets label from JSON photo at index i.
    # Files sorted by trailing number in filename (_Midspan_Height_N.jpg) for deterministic order.
    for node_pair, photo_files_for_pair in files_by_node_pair.items():
        json_photos_for_pair = json_by_node_pair.get(node_pair, [])
        if not json_photos_for_pair:
            continue

        # Sort files by trailing number (e.g. _53, _54) for deterministic order
        def sort_key(item):
            path, name = item
            m = re.search(r'_Midspan_Height_(\d+)\.jpg$', name)
            return int(m.group(1)) if m else 0

        sorted_files = sorted(photo_files_for_pair, key=sort_key)

        for file_index, (photo_path, photo_file) in enumerate(sorted_files):
            if file_index >= len(json_photos_for_pair):
                continue

            matched_photo_id = json_photos_for_pair[file_index]

            file_job_id = extract_job_id_from_photo_filename(photo_file)
            if file_job_id != job_id:
                continue

            if matched_photo_id not in photo_data:
                continue

            height_measurements = photo_data[matched_photo_id]['height_measurements']

            ground_coords = calculate_ground_coordinates(height_measurements)
            ruler_top_coords = calculate_ruler_top_coordinates(height_measurements)
            ruler_bounding_box = calculate_ruler_bounding_box(height_measurements)

            ppi = None
            if photo_file in img_dims_cache:
                w, h = img_dims_cache[photo_file]
                ppi = calculate_ppi_from_measurements(height_measurements, h)

            location_file = labels_dir / f"{photo_path.stem}_location.txt"

            with open(location_file, 'w', encoding='utf-8') as f:
                f.write("# Height measurements (feet) - Percentage coordinates\n")
                f.write("# Height,PercentX,PercentY\n")

                if ground_coords:
                    f.write(f"0.0,{ground_coords['percentX']},{ground_coords['percentY']}\n")
                else:
                    f.write("0.0,,,\n")

                if ruler_top_coords:
                    f.write(f"17.0,{ruler_top_coords['percentX']},{ruler_top_coords['percentY']}\n")
                else:
                    f.write("17.0,,,\n")

                for height in [2.5, 6.5, 10.5, 14.5, 16.5]:
                    if height in height_measurements:
                        m = height_measurements[height]
                        f.write(f"{height},{m['percentX']},{m['percentY']}\n")

                if ruler_bounding_box:
                    f.write(f"\n# Ruler bounding box (percentage coordinates)\n")
                    f.write(f"# {ruler_bounding_box['left']:.2f},{ruler_bounding_box['right']:.2f},{ruler_bounding_box['top']:.2f},{ruler_bounding_box['bottom']:.2f},{ruler_bounding_box['center_x']:.2f},{ruler_bounding_box['width']:.2f},{ruler_bounding_box['height']:.2f}\n")

                if ppi:
                    f.write(f"\n# PPI={ppi:.6f}\n")

            processed_count += 1

    return {'processed': processed_count}


def process_all_midspan_photos(
    photos_dir: Path,
    labels_dir: Path,
    data_dir: Path,
    workers: int = 1,
) -> Dict[str, int]:
    """Process all midspan photos by matching each to its correct JSON file.
    
    Args:
        workers: Number of parallel workers (1 = sequential).
    
    Returns:
        Dict with total counts across all jobs
    """
    labels_dir.mkdir(parents=True, exist_ok=True)

    if not photos_dir.exists():
        return {'processed': 0}

    all_photo_files = [f.name for f in photos_dir.iterdir() if 'Midspan' in f.name and f.suffix == '.jpg']

    if not all_photo_files:
        return {'processed': 0}

    # Group photos by job ID
    photos_by_job = {}
    for photo_file in all_photo_files:
        job_id = extract_job_id_from_photo_filename(photo_file)
        if job_id:
            if job_id not in photos_by_job:
                photos_by_job[job_id] = []
            photos_by_job[job_id].append(photo_file)

    job_args = [
        (job_id, data_dir, photos_dir, labels_dir)
        for job_id in photos_by_job
        if (data_dir / f"{job_id}.json").exists()
    ]

    total_processed = 0
    if workers <= 1:
        for job_id, data_dir_p, photos_dir_p, labels_dir_p in job_args:
            json_file = data_dir_p / f"{job_id}.json"
            photo_data = extract_midspan_photo_data(str(json_file))
            stats = create_midspan_location_files(
                job_id=job_id,
                photo_data=photo_data,
                photos_dir=photos_dir_p,
                labels_dir=labels_dir_p,
            )
            stats = stats or {'processed': 0}
            if stats['processed'] > 0:
                print(f"Created {stats['processed']} midspan location files for {job_id}", flush=True)
            total_processed += stats['processed']
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_midspan_job_worker, a): a[0] for a in job_args}
            for future in as_completed(futures):
                job_id, stats = future.result()
                if stats['processed'] > 0:
                    print(f"Created {stats['processed']} midspan location files for {job_id}", flush=True)
                total_processed += stats['processed']

    return {'processed': total_processed}


def find_job_json_files(data_dir: Path, explicit_files: Optional[List[str]] = None) -> List[Path]:
    """Determine which job JSON files to process."""
    if explicit_files:
        resolved_files: List[Path] = []
        for file_path in explicit_files:
            candidate = Path(file_path)
            if not candidate.is_absolute() and not candidate.exists():
                candidate = data_dir / candidate
            resolved_files.append(candidate)
        return resolved_files

    return sorted(data_dir.glob("*.json"))


def main():
    """Main function to extract height measurements and create location files.
    Processes both pole and midspan datasets automatically."""

    parser = argparse.ArgumentParser(description="Extract height measurements for job photos.")
    parser.add_argument("--json-files", nargs="+", help="Specific job JSON files to process.")
    parser.add_argument("--data-dir-pole", default=None, help="Directory containing pole job JSON files.")
    parser.add_argument("--photos-dir-pole", default=None, help="Directory containing pole photos.")
    parser.add_argument("--labels-dir-pole", default=None, help="Directory to store pole location files.")
    parser.add_argument("--data-dir-midspan", default=None, help="Directory containing midspan job JSON files.")
    parser.add_argument("--photos-dir-midspan", default=None, help="Directory containing midspan photos.")
    parser.add_argument("--labels-dir-midspan", default=None, help="Directory to store midspan location files.")
    parser.add_argument("--pole", action="store_true", help="Process pole dataset only.")
    parser.add_argument("--midspan", action="store_true", help="Process midspan dataset only.")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )

    args = parser.parse_args()

    extract_all = not (args.pole or args.midspan)
    do_pole = extract_all or args.pole
    do_midspan = extract_all or args.midspan
    workers = max(1, args.workers)

    print("Extract height measurements: starting...", flush=True)
    if workers > 1:
        print(f"Using {workers} parallel workers", flush=True)

    # Statistics accumulators
    pole_stats = {'processed': 0, 'skipped_multiple_cal': 0, 'skipped_no_cal': 0, 'jobs': 0}
    midspan_stats = {'processed': 0, 'jobs': 0}

    # Process pole dataset (paths relative to project root so script works from any cwd)
    pole_data_dir = Path(args.data_dir_pole) if args.data_dir_pole else PROJECT_ROOT / "data" / "data_pole"
    pole_photos_dir = Path(args.photos_dir_pole) if args.photos_dir_pole else PROJECT_ROOT / "data" / "data_pole" / "Photos"
    pole_labels_dir = Path(args.labels_dir_pole) if args.labels_dir_pole else PROJECT_ROOT / "data" / "data_pole" / "Labels"

    if do_pole:
        job_json_files = find_job_json_files(pole_data_dir, args.json_files)
        if not pole_data_dir.exists():
            print(f"Pole data dir not found: {pole_data_dir}", flush=True)
        elif job_json_files:
            print(f"Processing {len(job_json_files)} pole jobs from {pole_data_dir}...", flush=True)
            if workers <= 1:
                for json_file in job_json_files:
                    stats = process_job(json_file=json_file, photos_dir=pole_photos_dir, labels_dir=pole_labels_dir)
                    pole_stats['processed'] += stats['processed']
                    pole_stats['skipped_multiple_cal'] += stats['skipped_multiple_cal']
                    pole_stats['skipped_no_cal'] += stats['skipped_no_cal']
                    pole_stats['jobs'] += 1
            else:
                job_args = [(j, pole_photos_dir, pole_labels_dir) for j in job_json_files]
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(_process_pole_job_worker, a) for a in job_args]
                    for future in as_completed(futures):
                        _job_id, stats = future.result()
                        pole_stats['processed'] += stats['processed']
                        pole_stats['skipped_multiple_cal'] += stats['skipped_multiple_cal']
                        pole_stats['skipped_no_cal'] += stats['skipped_no_cal']
                        pole_stats['jobs'] += 1
        else:
            print(f"No pole JSON files found in {pole_data_dir}", flush=True)

    if do_midspan:
        midspan_data_dir = Path(args.data_dir_midspan) if args.data_dir_midspan else PROJECT_ROOT / "data" / "data_midspan"
        midspan_photos_dir = Path(args.photos_dir_midspan) if args.photos_dir_midspan else PROJECT_ROOT / "data" / "data_midspan" / "Photos"
        midspan_labels_dir = Path(args.labels_dir_midspan) if args.labels_dir_midspan else PROJECT_ROOT / "data" / "data_midspan" / "Labels"
        
        stats = process_all_midspan_photos(
            photos_dir=midspan_photos_dir,
            labels_dir=midspan_labels_dir,
            data_dir=midspan_data_dir,
            workers=workers,
        )
        midspan_stats['processed'] = stats['processed']
        if stats['processed'] > 0:
            midspan_stats['jobs'] = 1  # Counted collectively

    # Print summary
    print("\n" + "="*80, flush=True)
    print("SUMMARY", flush=True)
    print("="*80, flush=True)
    
    if do_pole and pole_stats['jobs'] > 0:
        print(f"\nPole Dataset ({pole_stats['jobs']} jobs processed):", flush=True)
        print(f"  ✓ Created labels: {pole_stats['processed']:,}", flush=True)
        print(f"  ⚠️  Skipped (multiple calibrations): {pole_stats['skipped_multiple_cal']:,}", flush=True)
        print(f"  ⚠️  Skipped (no calibration data): {pole_stats['skipped_no_cal']:,}", flush=True)
        total_pole = pole_stats['processed'] + pole_stats['skipped_multiple_cal'] + pole_stats['skipped_no_cal']
        if total_pole > 0:
            success_rate = (pole_stats['processed'] / total_pole) * 100
            print(f"  Success rate: {success_rate:.1f}%", flush=True)
    
    if do_midspan and midspan_stats['processed'] > 0:
        print(f"\nMidspan Dataset:", flush=True)
        print(f"  ✓ Created labels: {midspan_stats['processed']:,}", flush=True)

    print("\n" + "="*80, flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
