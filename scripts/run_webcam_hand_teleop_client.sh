#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8765}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
XY_SCALE="${XY_SCALE:-0.055}"
Z_SCALE="${Z_SCALE:-0.10}"
MAX_ABS_DELTA="${MAX_ABS_DELTA:-0.025}"
PINCH_CLOSE_THRESHOLD="${PINCH_CLOSE_THRESHOLD:-0.045}"
PINCH_OPEN_THRESHOLD="${PINCH_OPEN_THRESHOLD:-0.075}"

cd "$PROJECT_DIR"

python3 tools/webcam_hand_teleop_client.py \
  --server-host "$SERVER_HOST" \
  --server-port "$SERVER_PORT" \
  --camera-index "$CAMERA_INDEX" \
  --xy-scale "$XY_SCALE" \
  --z-scale "$Z_SCALE" \
  --max-abs-delta "$MAX_ABS_DELTA" \
  --pinch-close-threshold "$PINCH_CLOSE_THRESHOLD" \
  --pinch-open-threshold "$PINCH_OPEN_THRESHOLD"
