#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-$HOME/stereo_dataset}"
OBJ_NAME="${OBJ_NAME:-newton_thread_cloth_pad_v15_obj}"
ARCHIVE="${ARCHIVE:-${OBJ_NAME}.tgz}"
OUT_NAME="${OUT_NAME:-newton_thread_cloth_pad_v15}"

cd "$DATASET_DIR"

tar -xzf "$ARCHIVE"

docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/workspace/.cache \
  -e NUMBA_CACHE_DIR=/workspace/.cache/numba \
  -e TI_CACHE_PATH=/workspace/.cache/taichi \
  -v "$HOME/stereo_dataset:/workspace" \
  -w /workspace \
  genesis-tools \
  python tools/render_scene_obj_genesis_sequence.py \
    --obj-dir "/workspace/$OBJ_NAME" \
    --out-dir "/workspace/genesis_sim/outputs/$OUT_NAME" \
    --gif "/workspace/genesis_sim/outputs/$OUT_NAME/thread_cloth_pad_v15.gif" \
    --camera-fit union \
    --view pad-bird \
    --distance-scale 2.4 \
    --max-frames 20

echo "GIF: $DATASET_DIR/genesis_sim/outputs/$OUT_NAME/thread_cloth_pad_v15.gif"
