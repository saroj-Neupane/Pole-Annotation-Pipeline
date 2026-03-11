#!/usr/bin/env python3
"""
Download official HRNet-W32 COCO pretrained weights and remap to custom key names.

The official model is trained on COCO human pose (17 keypoints).
This script extracts the backbone weights and remaps them to match the custom
HRNet implementation key naming used in this project.

Architecture notes:
- Official HRNet layer1: Bottleneck (64→256) — CANNOT be loaded (shape mismatch)
- Official HRNet stage2-4: BasicBlock — CAN be loaded with key remapping
- Stem (conv1/bn1/conv2/bn2): direct match ✓
- Transitions 1: input shape mismatch (official 256ch vs custom 64ch) ✗
- Transition2_3 and Transition3_4: may match ✓
- Stage2-4 branches: full match after key remapping ✓

Usage:
    python scripts/download_hrnet_pretrained.py
    python scripts/download_hrnet_pretrained.py --verify   # also run a forward pass check
"""

import sys
import argparse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Official mmpose HRNet-W32 COCO 256x192
COCO_HRNET_URL = (
    "https://download.openmmlab.com/mmpose/top_down/hrnet/"
    "hrnet_w32_coco_256x192-c78dce93_20200708.pth"
)
OUTPUT_PATH = PROJECT_ROOT / "models" / "hrnet_w32.pth"


def _remap_keys(official_sd: dict) -> dict:
    """Remap official HRNet key names to custom implementation key names."""
    remapped = {}
    loaded_from = {}  # custom_key → official_key (for reporting)

    # ── Stem ──────────────────────────────────────────────────────────────────
    for k in [
        "conv1.weight",
        "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var", "bn1.num_batches_tracked",
        "conv2.weight",
        "bn2.weight", "bn2.bias", "bn2.running_mean", "bn2.running_var", "bn2.num_batches_tracked",
    ]:
        if k in official_sd:
            remapped[k] = official_sd[k]
            loaded_from[k] = k

    # ── Stage 2 branches (32-ch and 64-ch BasicBlocks × 4 blocks) ─────────────
    for branch_idx, custom_branch in enumerate(["stage2_branch1", "stage2_branch2"]):
        for block_idx in range(4):
            off_pfx = f"stage2.0.branches.{branch_idx}.{block_idx}"
            cus_pfx = f"{custom_branch}.{block_idx}"
            for s in [
                "conv1.weight",
                "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var", "bn1.num_batches_tracked",
                "conv2.weight",
                "bn2.weight", "bn2.bias", "bn2.running_mean", "bn2.running_var", "bn2.num_batches_tracked",
            ]:
                ok = f"{off_pfx}.{s}"
                ck = f"{cus_pfx}.{s}"
                if ok in official_sd:
                    remapped[ck] = official_sd[ok]
                    loaded_from[ck] = ok

    # ── Transition2_3: Conv2d(64→128, stride=2) ───────────────────────────────
    # Official: transition2.2.0.{0=Conv, 1=BN}
    for suffix, cus_idx in [("weight", 0), ("bias", 0)]:
        ok = f"transition2.2.0.0.{suffix}"
        ck = f"transition2_3.0.{suffix}"
        if ok in official_sd:
            remapped[ck] = official_sd[ok]
            loaded_from[ck] = ok
    for suffix in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
        ok = f"transition2.2.0.1.{suffix}"
        ck = f"transition2_3.1.{suffix}"
        if ok in official_sd:
            remapped[ck] = official_sd[ok]
            loaded_from[ck] = ok

    # ── Stage 3 branches (32-ch, 64-ch, 128-ch BasicBlocks × 4 blocks) ────────
    for branch_idx, custom_branch in enumerate(
        ["stage3_branch1", "stage3_branch2", "stage3_branch3"]
    ):
        for block_idx in range(4):
            off_pfx = f"stage3.0.branches.{branch_idx}.{block_idx}"
            cus_pfx = f"{custom_branch}.{block_idx}"
            for s in [
                "conv1.weight",
                "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var", "bn1.num_batches_tracked",
                "conv2.weight",
                "bn2.weight", "bn2.bias", "bn2.running_mean", "bn2.running_var", "bn2.num_batches_tracked",
            ]:
                ok = f"{off_pfx}.{s}"
                ck = f"{cus_pfx}.{s}"
                if ok in official_sd:
                    remapped[ck] = official_sd[ok]
                    loaded_from[ck] = ok

    # ── Transition3_4: Conv2d(128→256, stride=2) ──────────────────────────────
    # Official: transition3.3.0.{0=Conv, 1=BN}
    for suffix in ["weight", "bias"]:
        ok = f"transition3.3.0.0.{suffix}"
        ck = f"transition3_4.0.{suffix}"
        if ok in official_sd:
            remapped[ck] = official_sd[ok]
            loaded_from[ck] = ok
    for suffix in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
        ok = f"transition3.3.0.1.{suffix}"
        ck = f"transition3_4.1.{suffix}"
        if ok in official_sd:
            remapped[ck] = official_sd[ok]
            loaded_from[ck] = ok

    # ── Stage 4 branches (32-ch, 64-ch, 128-ch, 256-ch BasicBlocks × 4) ──────
    for branch_idx, custom_branch in enumerate(
        ["stage4_branch1", "stage4_branch2", "stage4_branch3", "stage4_branch4"]
    ):
        for block_idx in range(4):
            off_pfx = f"stage4.0.branches.{branch_idx}.{block_idx}"
            cus_pfx = f"{custom_branch}.{block_idx}"
            for s in [
                "conv1.weight",
                "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var", "bn1.num_batches_tracked",
                "conv2.weight",
                "bn2.weight", "bn2.bias", "bn2.running_mean", "bn2.running_var", "bn2.num_batches_tracked",
            ]:
                ok = f"{off_pfx}.{s}"
                ck = f"{cus_pfx}.{s}"
                if ok in official_sd:
                    remapped[ck] = official_sd[ok]
                    loaded_from[ck] = ok

    return remapped, loaded_from


def _shape_validate(remapped: dict, model_sd: dict) -> tuple[dict, list]:
    """Keep only tensors whose shape matches the custom model. Return (valid, mismatches)."""
    valid = {}
    mismatches = []
    for k, v in remapped.items():
        if k in model_sd:
            if v.shape == model_sd[k].shape:
                valid[k] = v
            else:
                mismatches.append(
                    f"  SHAPE MISMATCH {k}: pretrained {tuple(v.shape)} vs model {tuple(model_sd[k].shape)}"
                )
        else:
            mismatches.append(f"  KEY NOT IN MODEL: {k}")
    return valid, mismatches


def _print_summary(valid: dict, mismatches: list, total_model_params: int):
    loaded = len(valid)
    print(f"\n{'─'*60}")
    print(f"  Loaded:   {loaded} tensors")
    print(f"  Skipped:  {len(mismatches)} (shape/key mismatch)")
    print(f"  Coverage: {loaded}/{total_model_params} backbone tensors "
          f"({100*loaded/max(1,total_model_params):.1f}%)")
    if mismatches:
        print("\n  Mismatches (expected for layer1 + transition1):")
        for m in mismatches[:10]:
            print(m)
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches)-10} more")
    print(f"{'─'*60}\n")


def download_and_remap(url: str, output_path: Path, verify: bool = False):
    import torch
    from src.models import HRNet

    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_path = output_path.parent / "hrnet_w32_coco_raw.pth"
    if not raw_path.exists():
        print(f"Downloading from:\n  {url}\n")
        def _progress(count, block_size, total):
            pct = min(100, count * block_size * 100 // total)
            print(f"\r  Progress: {pct}%", end="", flush=True)
        urllib.request.urlretrieve(url, raw_path, _progress)
        print("\n  Download complete.")
    else:
        print(f"Found cached download: {raw_path}")

    print("Loading official weights...")
    official = torch.load(raw_path, map_location="cpu")
    if "state_dict" in official:
        official_sd = official["state_dict"]
    else:
        official_sd = official

    # Strip common prefixes
    official_sd = {
        k.replace("backbone.", "").replace("module.", ""): v
        for k, v in official_sd.items()
    }

    print(f"Official checkpoint: {len(official_sd)} keys")
    print("Sample official keys:")
    for k in list(official_sd.keys())[:8]:
        print(f"  {k}")

    print("\nRemapping keys to custom HRNet naming...")
    remapped, loaded_from = _remap_keys(official_sd)

    # Shape-validate against actual model
    dummy_backbone = HRNet(width=32)
    model_sd = dummy_backbone.state_dict()
    valid, mismatches = _shape_validate(remapped, model_sd)

    _print_summary(valid, mismatches, len(model_sd))

    # Save remapped backbone weights
    torch.save(valid, output_path)
    print(f"Saved remapped backbone weights → {output_path}")

    if verify:
        print("\nVerifying: loading into KeypointDetector and running forward pass...")
        from src.models import KeypointDetector
        import torch

        model = KeypointDetector(num_keypoints=2, heatmap_size=(48, 64), weights_path=output_path)
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 192, 256)
            out = model(dummy)
        print(f"  Forward pass OK. Output shape: {tuple(out.shape)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=COCO_HRNET_URL, help="Override download URL")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output path for remapped weights")
    parser.add_argument("--verify", action="store_true", help="Run a forward pass after loading")
    args = parser.parse_args()

    download_and_remap(args.url, args.output, verify=args.verify)


if __name__ == "__main__":
    main()
