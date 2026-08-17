#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
IMAGE="${IMAGE:-thread-reconstruction-newton:latest}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:---gpus all}"
OUT_DIR="${OUT_DIR:-/workspace/thread_reconstruction/cable_wrap_rod_v1}"
HOST_OUT_DIR="${HOST_OUT_DIR:-cable_wrap_rod_v1}"
FRAMES="${FRAMES:-90}"
FPS="${FPS:-30}"
NODES="${NODES:-96}"
CABLE_RADIUS="${CABLE_RADIUS:-0.0015}"
ROD_RADIUS="${ROD_RADIUS:-0.012}"
ROD_LENGTH="${ROD_LENGTH:-0.10}"
WRAP_TURNS="${WRAP_TURNS:-2.25}"
START_GAP="${START_GAP:-0.003}"

cd "$PROJECT_DIR"

docker run --rm $DOCKER_GPU_ARGS \
  -e PYTHONPATH=/workspace/thread_reconstruction:/workspace/thread_reconstruction/src \
  -v "$PWD:/workspace/thread_reconstruction" \
  -w /workspace/thread_reconstruction \
  "$IMAGE" \
  bash -lc "python3 -m pip install --no-cache-dir -q usd-core >/dev/null 2>&1 || true; \
    python3 tools/simulate_cable_wrap_rod_newton_viewer.py \
      --out-dir '$OUT_DIR' \
      --frames '$FRAMES' \
      --fps '$FPS' \
      --nodes '$NODES' \
      --cable-radius '$CABLE_RADIUS' \
      --rod-radius '$ROD_RADIUS' \
      --rod-length '$ROD_LENGTH' \
      --wrap-turns '$WRAP_TURNS' \
      --start-gap '$START_GAP'"

tar -czf cable_wrap_rod_v1.tgz "$HOST_OUT_DIR"

echo
echo "Saved:"
echo "  $PROJECT_DIR/$HOST_OUT_DIR/cable_wrap_rod_newton.usd"
echo "  $PROJECT_DIR/$HOST_OUT_DIR/cable_wrap_rod_states.npz"
echo "  $PROJECT_DIR/$HOST_OUT_DIR/obj_frames/"
echo "Archive:"
echo "  $PROJECT_DIR/cable_wrap_rod_v1.tgz"
