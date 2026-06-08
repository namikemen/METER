#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$1" != "--" ]; then
    echo "Usage: $0 -- <training command>"
    echo "Example: $0 -- python meter_train.py --data-dir /workspace/data/nyu_depth_v2 --output-dir /workspace/outputs/run_001"
    exit 1
fi

shift

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ROOT="${SERVER_ROOT:-${HOME}/METER}"
DATA_DIR="${DATA_DIR:-${SERVER_ROOT}/data}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${SERVER_ROOT}/outputs}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${SERVER_ROOT}/checkpoints}"
IMAGE_NAME="${IMAGE_NAME:-meter:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-meter-training}"

mkdir -p "$DATA_DIR" "$OUTPUTS_DIR" "$CHECKPOINTS_DIR"

docker build -t "$IMAGE_NAME" -f "$REPO_DIR/docker/Dockerfile" "$REPO_DIR/docker"

docker run --rm --gpus all --ipc host \
    --name "$CONTAINER_NAME" \
    -v "$REPO_DIR:/workspace/METER" \
    -v "$DATA_DIR:/workspace/data" \
    -v "$OUTPUTS_DIR:/workspace/outputs" \
    -v "$CHECKPOINTS_DIR:/workspace/checkpoints" \
    -w /workspace/METER \
    "$IMAGE_NAME" \
    "$@"
