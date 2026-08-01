#!/usr/bin/env bash
# Build and run the app in Docker locally (test before deploying).
# Usage: from repo root: ./deploy/run_docker_local.sh
# Optional: mount local models for annotation/calibration:
#   ./deploy/run_docker_local.sh --mount-models
# Demo images: mount inference/ (inference/pole/images, inference/midspan/images).
#   Run ./deploy/scripts/prepare_demo_images.sh first to populate.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_NAME="pole-annotation-app:local"

echo "Building image $IMAGE_NAME..."
docker build -f deploy/Dockerfile -t "$IMAGE_NAME" .

echo ""
echo "Running container (port 7860)..."
echo "  App: http://localhost:7860"
echo "  Stop: Ctrl+C or docker stop \$(docker ps -q --filter ancestor=$IMAGE_NAME)"
echo ""

EXTRA_ARGS=()
if [ -f "$ROOT_DIR/.env" ]; then
  EXTRA_ARGS+=( --env-file "$ROOT_DIR/.env" )
  echo "  Using .env for config"
fi
# Mount inference/ so demo can list photos (inference/pole/images, inference/midspan/images)
mkdir -p "$ROOT_DIR/inference/pole/images" "$ROOT_DIR/inference/midspan/images"
EXTRA_ARGS+=( -v "$ROOT_DIR/inference:/app/inference:ro" )
echo "  Mounting inference/ (demo images)"
if [[ " $* " = *" --mount-models "* ]]; then
  if [ -d "$ROOT_DIR/models" ]; then
    EXTRA_ARGS+=( -v "$ROOT_DIR/models:/app/models:ro" )
    echo "  Mounting local models (read-only)"
  else
    echo "  Warning: --mount-models given but $ROOT_DIR/models not found"
  fi
fi

docker run --rm -p 7860:7860 \
  -e ENVIRONMENT=development \
  -e BASE_URL=http://localhost:7860 \
  "${EXTRA_ARGS[@]}" \
  "$IMAGE_NAME"
