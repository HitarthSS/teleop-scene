#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
IMAGE="${IMAGE:-thread-reconstruction-newton:latest}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8765}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:---gpus all}"
SCENE_NPZ="${SCENE_NPZ:-/workspace/thread_reconstruction/newton_frame_scene_000000/newton_frame_scene.npz}"
URDF="${URDF:-/workspace/thread_reconstruction/assets/dvrk/psm1_si.urdf}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/workspace/thread_reconstruction/assets}"
JOINTS="${JOINTS:-/workspace/thread_reconstruction/thread_with_gripper_pose_frames/episode_0000_inputs/joint_000000.npy}"
JAW="${JAW:-/workspace/thread_reconstruction/thread_with_gripper_pose_frames/episode_0000_inputs/jaw_000000.npy}"
VIEWER_CAMERA_SPEED="${VIEWER_CAMERA_SPEED:-0.002}"
VIEWER_MAX_CAMERA_SPEED="${VIEWER_MAX_CAMERA_SPEED:-0.005}"
VIEWER_SCROLL_SCALE="${VIEWER_SCROLL_SCALE:-0.03}"
VIEWER_LOCK_CAMERA_SPEED="${VIEWER_LOCK_CAMERA_SPEED:-true}"
VIEWER_DISTANCE_SCALE="${VIEWER_DISTANCE_SCALE:-2.8}"
VIEWER_CAMERA_MODE="${VIEWER_CAMERA_MODE:-surgical}"
VIEWER_CAMERA_FOLLOW="${VIEWER_CAMERA_FOLLOW:-true}"
VIEWER_CAMERA_FOLLOW_ALPHA="${VIEWER_CAMERA_FOLLOW_ALPHA:-0.12}"
VIEWER_TOP_DISTANCE_SCALE="${VIEWER_TOP_DISTANCE_SCALE:-2.4}"
VIEWER_FULL_DISTANCE_SCALE="${VIEWER_FULL_DISTANCE_SCALE:-3.2}"
VIEWER_GRIPPER_DISTANCE_SCALE="${VIEWER_GRIPPER_DISTANCE_SCALE:-8.0}"
VIEWER_GRIPPER_CAMERA_RADIUS="${VIEWER_GRIPPER_CAMERA_RADIUS:-0.012}"

if [[ "$VIEWER_LOCK_CAMERA_SPEED" == "0" || "$VIEWER_LOCK_CAMERA_SPEED" == "false" ]]; then
  LOCK_CAMERA_SPEED_FLAG="--no-lock-camera-speed"
else
  LOCK_CAMERA_SPEED_FLAG="--lock-camera-speed"
fi

if [[ "$VIEWER_CAMERA_FOLLOW" == "0" || "$VIEWER_CAMERA_FOLLOW" == "false" ]]; then
  CAMERA_FOLLOW_FLAG="--no-camera-follow"
else
  CAMERA_FOLLOW_FLAG="--camera-follow"
fi

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
    --server-port "$SERVER_PORT" \
    --scene-npz "$SCENE_NPZ" \
    --urdf "$URDF" \
    --package-root "$PACKAGE_ROOT" \
    --joints "$JOINTS" \
    --jaw "$JAW" \
    --camera-mode "$VIEWER_CAMERA_MODE" \
    "$CAMERA_FOLLOW_FLAG" \
    --camera-follow-alpha "$VIEWER_CAMERA_FOLLOW_ALPHA" \
    --camera-speed "$VIEWER_CAMERA_SPEED" \
    --max-camera-speed "$VIEWER_MAX_CAMERA_SPEED" \
    "$LOCK_CAMERA_SPEED_FLAG" \
    --scroll-scale "$VIEWER_SCROLL_SCALE" \
    --distance-scale "$VIEWER_DISTANCE_SCALE" \
    --top-distance-scale "$VIEWER_TOP_DISTANCE_SCALE" \
    --full-distance-scale "$VIEWER_FULL_DISTANCE_SCALE" \
    --gripper-distance-scale "$VIEWER_GRIPPER_DISTANCE_SCALE" \
    --gripper-camera-radius "$VIEWER_GRIPPER_CAMERA_RADIUS"
