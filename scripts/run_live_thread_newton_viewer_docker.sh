#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
IMAGE="${IMAGE:-thread-reconstruction-newton:latest}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8765}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:---gpus all}"

cd "$PROJECT_DIR"

docker run --rm $DOCKER_GPU_ARGS \
  --network host \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e PYTHONPATH=/workspace/thread_reconstruction:/workspace/thread_reconstruction/src \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$PWD:/workspace/thread_reconstruction" \
  -w /workspace/thread_reconstruction \
  "$IMAGE" \
  python3 tools/live_thread_newton_viewer.py \
    --server-host "$SERVER_HOST" \
    --server-port "$SERVER_PORT"
