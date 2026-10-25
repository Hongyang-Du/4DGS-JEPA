#!/usr/bin/env bash
# Build the image (once) and start a long-lived dev container with the repo mounted.
#   GPU=4 bash docker/run_docker.sh          # pick the GPU index
#   docker exec -it 4dgs-jepa bash           # attach
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-4dgs-jepa:latest}"
NAME="${NAME:-4dgs-jepa}"
GPU="${GPU:-0}"

docker build -t "$IMAGE" -f "$REPO/docker/Dockerfile" "$REPO/docker"

docker run -d --name "$NAME" \
  --gpus "\"device=${GPU}\"" \
  --ipc=host --shm-size=8g \
  -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$REPO":/workspace/4DGS-JEPA \
  "$IMAGE" sleep infinity

echo "Container '$NAME' is up on GPU ${GPU}. Attach with: docker exec -it $NAME bash"
