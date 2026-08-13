#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/thread_recon_run/thread_reconstruction-kf}"
IMAGE="${IMAGE:-thread-reconstruction-newton:latest}"
OUT_NAME="${OUT_NAME:-psm_robot_motion_v1}"
SCENE_NPZ="${SCENE_NPZ:-/workspace/thread_reconstruction/newton_frame_scene_000000/newton_frame_scene.npz}"
TOOL_INCLUDE_REGEX="${TOOL_INCLUDE_REGEX:-tool_wrist|tool_wrist_sca}"
TOOL_EXCLUDE_REGEX="${TOOL_EXCLUDE_REGEX:-tool_main_link}"
URDF="${URDF:-/workspace/home/thread_recon_run/assets/dvrk/psm1_si.urdf}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/workspace/home/thread_recon_run/assets}"
JOINTS="${JOINTS:-/workspace/thread_reconstruction/thread_with_gripper_pose_frames/episode_0000_inputs/joint_000000.npy}"
JAW="${JAW:-/workspace/thread_reconstruction/thread_with_gripper_pose_frames/episode_0000_inputs/jaw_000000.npy}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_NAME"

echo "== PSM URDF robot-only Newton animation =="
docker run --rm --gpus all \
  -e PYTHONPATH=/workspace/thread_reconstruction:/workspace/thread_reconstruction/src \
  -v "$PWD:/workspace/thread_reconstruction" \
  -v "$HOME:/workspace/home" \
  -w /workspace/thread_reconstruction \
  "$IMAGE" \
  bash -lc "
    set -euo pipefail
    python3 -m pip install --no-cache-dir usd-core trimesh

    python3 tools/animate_psm_urdf_robot_newton.py \
      --scene-npz '$SCENE_NPZ' \
      --urdf '$URDF' \
      --package-root '$PACKAGE_ROOT' \
      --joints '$JOINTS' \
      --jaw '$JAW' \
      --out-dir '/workspace/thread_reconstruction/$OUT_NAME' \
      --include-mesh-regex '$TOOL_INCLUDE_REGEX' \
      --exclude-mesh-regex '$TOOL_EXCLUDE_REGEX' \
      --frames 12 \
      --fps 12 \
      --jaw-scale 0.05 \
      --wrist-yaw-delta 0.35 \
      --wrist-pitch-delta 0.25 \
      --insertion-delta 0.0
  " 2>&1 | tee "$OUT_NAME/psm_robot_motion_log.txt"

echo "== Package outputs =="
tar -czf "$HOME/thread_recon_run/${OUT_NAME}.tgz" "$OUT_NAME"

echo "Outputs:"
echo "  $PROJECT_DIR/$OUT_NAME/psm_robot_motion_newton.usd"
echo "  $PROJECT_DIR/$OUT_NAME/obj_frames"
echo "  $PROJECT_DIR/$OUT_NAME/psm_robot_motion_report.txt"
echo "Archive:"
echo "  $HOME/thread_recon_run/${OUT_NAME}.tgz"
