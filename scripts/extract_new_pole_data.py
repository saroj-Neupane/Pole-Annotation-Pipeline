#!/usr/bin/env python3
"""
Extract pole data from New_Pole_Data and copy to data/data_pole.

Handles:
1. Extracting photos from *_Export.zip files to data/data_pole/Photos/
2. Copying JSON to data/data_pole/ (from zip when present, else standalone)
3. Running extract_height to generate Labels/_location.txt files

Usage:
    python scripts/extract_new_pole_data.py [--dry-run]
"""

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BASE_DIR_POLE, POLE_LABELS_DIR
NEW_POLE_DATA = PROJECT_ROOT / "New_Pole_Data"
DATA_POLE = BASE_DIR_POLE
PHOTOS_DIR = BASE_DIR_POLE / "Photos"
LABELS_DIR = POLE_LABELS_DIR

# Expected photo filename pattern: {job_id}_{scid}_\d+_Main.jpg
PHOTO_PATTERN = re.compile(r"^.+_\d+_\d+_Main\.jpg$")


def get_job_id_from_zip_name(zip_path: Path) -> str | None:
    """Extract job ID from Export zip filename. E.g. MIBR-FR01_Export.zip -> MIBR-FR01."""
    name = zip_path.stem
    if "_Export" in name:
        return name.split("_Export")[0].strip()
    return None


def zip_has_photos(zip_path: Path) -> bool:
    """Check if zip contains Photos/ directory with jpg files."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if "Photos/" in info.filename and info.filename.lower().endswith(".jpg"):
                    return True
    except zipfile.BadZipFile:
        pass
    return False


def zip_has_json(zip_path: Path) -> bool:
    """Check if zip contains a .json file (at root level)."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.filename.lower().endswith(".json") and "/" not in info.filename:
                    return True
    except zipfile.BadZipFile:
        pass
    return False


def extract_json_from_zip(zip_path: Path, dest_dir: Path, dry_run: bool) -> Path | None:
    """Extract JSON from zip to dest_dir. Returns dest path or None."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.filename.lower().endswith(".json") and "/" not in info.filename:
                    target = dest_dir / Path(info.filename).name
                    if not dry_run:
                        with zf.open(info) as src:
                            target.write_bytes(src.read())
                    return target
    except zipfile.BadZipFile:
        pass
    return None


def extract_photos_from_zip(zip_path: Path, dest_dir: Path, dry_run: bool) -> int:
    """Extract Photos/*.jpg from zip to dest_dir. Returns count of extracted files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if not info.filename.startswith("Photos/") or not info.filename.lower().endswith(".jpg"):
                continue
            # Use basename to avoid nested Photos/Photos/
            target_name = Path(info.filename).name
            # Skip malformed filenames (e.g. "MIBR-FR03_159 MIBR-FR02_1_Main.jpg")
            if not PHOTO_PATTERN.match(target_name):
                continue
            target_path = dest_dir / target_name
            if not dry_run:
                with zf.open(info) as src:
                    target_path.write_bytes(src.read())
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Extract New_Pole_Data to data/data_pole")
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying")
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only copy JSON files (skip zip extraction and label generation)",
    )
    args = parser.parse_args()

    DATA_POLE.mkdir(parents=True, exist_ok=True)
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    # Build set of jobs: from standalone JSON + all Export zips
    standalone_jsons = {p.stem: p for p in NEW_POLE_DATA.glob("*.json")}
    export_zips = sorted(NEW_POLE_DATA.glob("*_Export*.zip"))
    job_zips: dict[str, list[Path]] = {}
    for z in export_zips:
        jid = get_job_id_from_zip_name(z)
        if jid:
            job_zips.setdefault(jid, []).append(z)

    # All unique job IDs
    all_jobs = set(standalone_jsons) | set(job_zips)
    if not all_jobs:
        print("No JSON files or Export zips found in New_Pole_Data/")
        return 1

    total_photos = 0
    for job_id in sorted(all_jobs):
        print(f"\n--- {job_id} ---")

        # JSON: prefer from zip, else standalone
        dest_json = DATA_POLE / f"{job_id}.json"
        zip_with_json = None
        for z in job_zips.get(job_id, []):
            if zip_has_json(z):
                zip_with_json = z
                break

        if zip_with_json:
            extracted = extract_json_from_zip(zip_with_json, DATA_POLE, args.dry_run)
            if extracted:
                print(f"  JSON: from {zip_with_json.name} -> {dest_json.relative_to(PROJECT_ROOT)}")
        elif job_id in standalone_jsons:
            if not args.dry_run:
                shutil.copy2(standalone_jsons[job_id], dest_json)
            print(f"  JSON: {standalone_jsons[job_id].name} -> {dest_json.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  JSON: none (no zip JSON or standalone)")
            if args.json_only:
                continue

        if args.json_only:
            continue

        # Photos: from best zip (prefer one with photos)
        zips = job_zips.get(job_id, [])
        best_zip = None
        for z in zips:
            if zip_has_photos(z):
                best_zip = z
                break
        if not best_zip and zips:
            best_zip = zips[0]

        if best_zip:
            n = extract_photos_from_zip(best_zip, PHOTOS_DIR, args.dry_run)
            total_photos += n
            print(f"  Photos: extracted {n} from {best_zip.name}")
        else:
            print(f"  Photos: none")

    if not args.json_only and total_photos > 0 and not args.dry_run:
        print("\n--- Running extract_height ---")
        result = subprocess.run(
            [
                "python",
                str(PROJECT_ROOT / "scripts" / "extract_height.py"),
                "--pole",
                "--data-dir-pole",
                str(DATA_POLE),
                "--photos-dir-pole",
                str(PHOTOS_DIR),
                "--labels-dir-pole",
                str(LABELS_DIR),
            ],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print("extract_height failed")
            return result.returncode
    elif args.dry_run:
        print(f"\nDry-run: would extract {total_photos} photos, then run extract_height")

    print(f"\nDone. Total photos: {total_photos}")
    return 0


if __name__ == "__main__":
    exit(main())
