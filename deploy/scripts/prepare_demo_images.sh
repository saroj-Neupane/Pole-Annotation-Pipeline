#!/usr/bin/env bash
# Prepare demo images for the /api/demo/random endpoint and demo page.
# Copies selected images to inference/pole/images and inference/midspan/images
# for use in the /api/demo/random endpoint and demo page.
#
# Usage: ./deploy/scripts/prepare_demo_images.sh [--select-interactive]
#   --select-interactive: Interactively select images to include as demos

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

POLE_ANNOTATED_DIR="inference/pole/annotated_photos"
MIDSPAN_ANNOTATED_DIR="inference/midspan/annotated_photos"
POLE_IMAGES_DIR="inference/pole/images"
MIDSPAN_IMAGES_DIR="inference/midspan/images"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🎬 Preparing demo images for upload${NC}\n"

# Create target directories
mkdir -p "$POLE_IMAGES_DIR" "$MIDSPAN_IMAGES_DIR"

# Count existing images
pole_count=$(find "$POLE_IMAGES_DIR" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
midspan_count=$(find "$MIDSPAN_IMAGES_DIR" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)

echo "Current demo images:"
echo "  Pole: $pole_count"
echo "  Midspan: $midspan_count"
echo ""

if [ "$1" == "--select-interactive" ]; then
    echo -e "${YELLOW}Interactive selection not yet implemented.${NC}"
    echo "For now, manually copy images to:"
    echo "  Pole demos:   $POLE_IMAGES_DIR"
    echo "  Midspan demos: $MIDSPAN_IMAGES_DIR"
    echo ""
    exit 0
fi

# Auto-select strategy: pick N diverse images from annotated_photos
echo -e "${YELLOW}📋 Auto-selecting diverse demo images...${NC}\n"

copy_demo_images() {
    local src_dir="$1"
    local target_dir="$2"
    local label="$3"
    local limit="${4:-5}"  # Default to 5 images per category

    if [ ! -d "$src_dir" ]; then
        echo "Skip $label (source not found: $src_dir)"
        return 0
    fi

    # Count available images
    local available=$(find "$src_dir" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)

    if [ "$available" -eq 0 ]; then
        echo "Skip $label (no images in $src_dir)"
        return 0
    fi

    # Clear existing demo images
    rm -f "$target_dir"/*

    # Copy up to $limit diverse images (sort alphabetically for consistency)
    find "$src_dir" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) -print0 | \
        sort -z | \
        head -z -n "$limit" | \
        xargs -0 -I {} cp {} "$target_dir/" 2>/dev/null || true

    local copied=$(find "$target_dir" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
    echo "✅ $label: Selected $copied/$available images"
}

copy_demo_images "$POLE_ANNOTATED_DIR" "$POLE_IMAGES_DIR" "Pole images" 5
copy_demo_images "$MIDSPAN_ANNOTATED_DIR" "$MIDSPAN_IMAGES_DIR" "Midspan images" 5

echo ""
echo -e "${GREEN}✨ Demo images ready!${NC}\n"
echo "Demo images ready under inference/."
