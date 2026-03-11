"""
Utility functions for loading datasets using metadata.csv.

This module provides helpers for:
- Loading datasets from CSV metadata
- Validating splits match CSV
- Filtering images based on metadata
- Computing statistics from metadata
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
import hashlib


def get_dataset_metadata(dataset_dir: Path) -> pd.DataFrame:
    """
    Load metadata.csv for a dataset.

    Args:
        dataset_dir: Path to dataset directory (e.g., 'datasets/pole_detection')

    Returns:
        DataFrame with metadata
    """
    metadata_file = Path(dataset_dir) / 'metadata.csv'

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"metadata.csv not found at {metadata_file}\n"
            f"Run: python scripts/generate_metadata.py"
        )

    return pd.read_csv(metadata_file)


def get_split_images(dataset_dir: Path, split: str = 'train') -> List[Path]:
    """
    Get list of image paths for a specific split (from CSV metadata).

    Args:
        dataset_dir: Path to dataset directory
        split: One of 'train', 'val', 'test'

    Returns:
        List of image file paths
    """
    df = get_dataset_metadata(dataset_dir)

    split_df = df[df['split'] == split]

    if len(split_df) == 0:
        raise ValueError(f"No images found for split '{split}'")

    dataset_dir = Path(dataset_dir)
    image_paths = [
        dataset_dir / 'images' / split / filename
        for filename in split_df['image_filename']
    ]

    return image_paths


def validate_split_matches_csv(dataset_dir: Path) -> bool:
    """
    Verify that folder structure matches CSV metadata.

    This ensures that:
    - Every image in CSV exists in folder
    - Every image in folder is listed in CSV
    - Every image has a corresponding label

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        True if validation passes
    """
    df = get_dataset_metadata(dataset_dir)
    dataset_dir = Path(dataset_dir)

    all_valid = True

    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]

        if len(split_df) == 0:
            continue

        # Check that all images in CSV exist
        for _, row in split_df.iterrows():
            img_path = dataset_dir / 'images' / split / row['image_filename']
            label_path = dataset_dir / 'labels' / split / f"{Path(row['image_filename']).stem}.txt"

            if not img_path.exists():
                print(f"❌ Image in CSV but not on disk: {img_path}")
                all_valid = False

            if not row['label_exists'] and not label_path.exists():
                print(f"⚠️  Label missing for: {row['image_filename']}")

        # Check that all images in folder are in CSV
        img_dir = dataset_dir / 'images' / split
        if img_dir.exists():
            for img_path in img_dir.glob('*.jpg'):
                if img_path.name not in split_df['image_filename'].values:
                    print(f"❌ Image on disk but not in CSV: {img_path}")
                    all_valid = False

    if all_valid:
        print("✅ Split validation passed - CSV matches folder structure")

    return all_valid


def get_split_statistics(dataset_dir: Path) -> dict:
    """
    Compute statistics for each split.

    Returns:
        Dict with statistics per split
    """
    df = get_dataset_metadata(dataset_dir)

    stats = {}

    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]

        if len(split_df) == 0:
            continue

        stats[split] = {
            'num_images': len(split_df),
            'total_size_mb': split_df['image_size_kb'].sum() / 1024,
            'avg_image_size_kb': split_df['image_size_kb'].mean(),
            'num_with_labels': split_df['label_exists'].sum(),
            'avg_visible_keypoints': split_df['num_visible_keypoints'].mean(),
            'num_with_ppi': (split_df['ppi'] > 0).sum(),
        }

    return stats


def filter_images_by_keypoints(
    dataset_dir: Path, split: str = 'train', min_keypoints: int = 3
) -> List[Path]:
    """
    Get images with at least N visible keypoints.

    Useful for filtering out low-quality samples during training.

    Args:
        dataset_dir: Path to dataset directory
        split: One of 'train', 'val', 'test'
        min_keypoints: Minimum number of visible keypoints

    Returns:
        List of image file paths
    """
    df = get_dataset_metadata(dataset_dir)

    split_df = df[(df['split'] == split) & (df['num_visible_keypoints'] >= min_keypoints)]

    if len(split_df) == 0:
        raise ValueError(
            f"No images with >= {min_keypoints} keypoints in {split} split"
        )

    dataset_dir = Path(dataset_dir)
    image_paths = [
        dataset_dir / 'images' / split / filename
        for filename in split_df['image_filename']
    ]

    return image_paths


def compute_split_hash(dataset_dir: Path) -> str:
    """
    Compute a hash of the split definition for reproducibility verification.

    This hash can be used to verify that the same split is being used across
    different runs or machines.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        SHA256 hash of the split definition
    """
    df = get_dataset_metadata(dataset_dir)

    # Create a canonical representation of the split
    split_str = '|'.join(
        f"{row['split']}:{row['image_filename']}"
        for _, row in df.iterrows()
    )

    return hashlib.sha256(split_str.encode()).hexdigest()


def print_split_summary(dataset_dir: Path) -> None:
    """Print a summary of the dataset split."""
    df = get_dataset_metadata(dataset_dir)

    print(f"\n{'='*60}")
    print(f"Dataset: {Path(dataset_dir).name}")
    print(f"{'='*60}")

    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]

        if len(split_df) == 0:
            continue

        pct = 100 * len(split_df) / len(df)
        total_size = split_df['image_size_kb'].sum() / 1024
        avg_keypoints = split_df['num_visible_keypoints'].mean()

        print(f"\n{split.upper()}:")
        print(f"  Images:      {len(split_df):,} ({pct:.1f}%)")
        print(f"  Total size:  {total_size:,.0f} MB")
        print(f"  Avg size:    {split_df['image_size_kb'].mean():.0f} KB")
        print(f"  Avg keypts:  {avg_keypoints:.1f}")

    print(f"\n{'='*60}\n")


def example_usage():
    """Examples of how to use these utilities."""

    print("\n" + "="*60)
    print("Example Usage")
    print("="*60)

    # Example 1: Load dataset
    print("\n1. Load metadata:")
    print("""
    from src.dataset_utils import get_dataset_metadata

    df = get_dataset_metadata('datasets/pole_detection')
    train_images = df[df['split'] == 'train']
    print(f"Training images: {len(train_images)}")
    """)

    # Example 2: Get images for split
    print("\n2. Get image paths for split:")
    print("""
    from src.dataset_utils import get_split_images

    train_paths = get_split_images('datasets/pole_detection', split='train')
    for path in train_paths[:5]:
        print(path)
    """)

    # Example 3: Validate split
    print("\n3. Validate split matches CSV:")
    print("""
    from src.dataset_utils import validate_split_matches_csv

    is_valid = validate_split_matches_csv('datasets/pole_detection')
    """)

    # Example 4: Get statistics
    print("\n4. Compute split statistics:")
    print("""
    from src.dataset_utils import get_split_statistics, print_split_summary

    stats = get_split_statistics('datasets/pole_detection')
    print_split_summary('datasets/pole_detection')
    """)

    # Example 5: Filter by keypoints
    print("\n5. Filter images by quality (min keypoints):")
    print("""
    from src.dataset_utils import filter_images_by_keypoints

    # Only train on high-quality samples
    good_images = filter_images_by_keypoints(
        'datasets/ruler_marking_detection',
        split='train',
        min_keypoints=4
    )
    print(f"High-quality images: {len(good_images)}")
    """)

    # Example 6: Compute split hash
    print("\n6. Verify reproducibility across runs:")
    print("""
    from src.dataset_utils import compute_split_hash

    hash1 = compute_split_hash('datasets/pole_detection')
    # ... train model, run elsewhere, etc ...
    hash2 = compute_split_hash('datasets/pole_detection')

    assert hash1 == hash2, "Split has changed!"
    """)

    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    example_usage()
