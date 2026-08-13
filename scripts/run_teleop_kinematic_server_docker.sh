#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
IMAGE="${IMAGE:-thread-reconstruction-newton:latest}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:---gpus all}"
COMMAND_PORT="${COMMAND_PORT:-8765}"
RATE_HZ="${RATE_HZ:-90}"
GRASP_GATE_RADIUS="${GRASP_GATE_RADIUS:-0.0015}"
GRASP_GATE_JAW_OPEN_FRACTION="${GRASP_GATE_JAW_OPEN_FRACTION:-1.0}"

SCENE_NPZ="${SCENE_NPZ:-/workspace/thread_reconstruction/newton_frame_scene_000000/newton_frame_scene.npz}"
URDF="${URDF:-/workspace/home/thread_recon_run/assets/dvrk/psm1_si.urdf}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/workspace/home/thread_recon_run/assets}"
JOINTS="${JOINTS:-/workspace/thread_reconstruction/thread_with_gripper_pose_frames/episode_0000_inputs/joint_000000.npy}"
JAW="${JAW:-/workspace/thread_reconstruction/thread_with_gripper_pose_frames/episode_0000_inputs/jaw_000000.npy}"

cd "$PROJECT_DIR"

docker run --rm $DOCKER_GPU_ARGS \
  --network host \
  -e PYTHONPATH=/workspace/thread_reconstruction:/workspace/thread_reconstruction/src \
  -v "$PWD:/workspace/thread_reconstruction" \
  -v "$HOME:/workspace/home" \
  -w /workspace/thread_reconstruction \
  "$IMAGE" \
  python3 tools/teleop_kinematic_server.py \
    --scene-npz "$SCENE_NPZ" \
    --urdf "$URDF" \
    --package-root "$PACKAGE_ROOT" \
    --joints "$JOINTS" \
    --jaw "$JAW" \
    --command-port "$COMMAND_PORT" \
    --rate-hz "$RATE_HZ" \
    --grasp-gate-radius "$GRASP_GATE_RADIUS" \
    --grasp-gate-jaw-open-fraction "$GRASP_GATE_JAW_OPEN_FRACTION"
