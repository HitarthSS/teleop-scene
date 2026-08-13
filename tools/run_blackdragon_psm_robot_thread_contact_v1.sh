#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/thread_recon_run/thread_reconstruction-kf}"
IMAGE="${IMAGE:-thread-reconstruction-newton:latest}"
OUT_NAME="${OUT_NAME:-psm_robot_thread_drag_v6}"
SCENE_NPZ="${SCENE_NPZ:-/workspace/thread_reconstruction/newton_frame_scene_000000/newton_frame_scene.npz}"
THREAD_DIAMETER_M="${THREAD_DIAMETER_M:-0.0006}"
THREAD_RADIUS_M="${THREAD_RADIUS_M:-0.0003}"
FRAMES="${FRAMES:-24}"
FPS="${FPS:-12}"
SUBSTEPS="${SUBSTEPS:-8}"
DT="${DT:-0.0008333333}"
ITERATIONS="${ITERATIONS:-50}"
BENCHMARK_NO_EXPORTS="${BENCHMARK_NO_EXPORTS:-false}"
TOOL_INCLUDE_REGEX="${TOOL_INCLUDE_REGEX:-tool_wrist|tool_wrist_sca}"
TOOL_EXCLUDE_REGEX="${TOOL_EXCLUDE_REGEX:-tool_main_link}"
URDF="${URDF:-/workspace/home/thread_recon_run/assets/dvrk/psm1_si.urdf}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/workspace/home/thread_recon_run/assets}"
JOINTS="${JOINTS:-/workspace/thread_reconstruction/thread_with_gripper_pose_frames/episode_0000_inputs/joint_000000.npy}"
JAW="${JAW:-/workspace/thread_reconstruction/thread_with_gripper_pose_frames/episode_0000_inputs/jaw_000000.npy}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_NAME"
if [[ "$BENCHMARK_NO_EXPORTS" == "true" ]]; then
  BENCHMARK_EXPORT_ARG="--benchmark-no-exports"
else
  BENCHMARK_EXPORT_ARG="--no-benchmark-no-exports"
fi

echo "== PSM robot motion + reconstructed Newton thread contact =="
docker run --rm --gpus all \
  -e PYTHONPATH=/workspace/thread_reconstruction:/workspace/thread_reconstruction/src \
  -v "$PWD:/workspace/thread_reconstruction" \
  -v "$HOME:/workspace/home" \
  -w /workspace/thread_reconstruction \
  "$IMAGE" \
  bash -lc "
    set -euo pipefail
    python3 -m pip install --no-cache-dir usd-core trimesh

    python3 tools/animate_psm_robot_with_thread_contact.py \
      --scene-npz '$SCENE_NPZ' \
      --urdf '$URDF' \
      --package-root '$PACKAGE_ROOT' \
      --joints '$JOINTS' \
      --jaw '$JAW' \
      --out-dir '/workspace/thread_reconstruction/$OUT_NAME' \
      --include-mesh-regex '$TOOL_INCLUDE_REGEX' \
      --exclude-mesh-regex '$TOOL_EXCLUDE_REGEX' \
      --frames '$FRAMES' \
      --fps '$FPS' \
      --substeps '$SUBSTEPS' \
      --dt '$DT' \
      --iterations '$ITERATIONS' \
      --thread-radius-m '$THREAD_RADIUS_M' \
      --jaw-scale 0.02 \
      --target-thread nearest-end \
      --reach-frames 8 \
      --grasp-frame 8 \
      --hard-grasp \
      --kinematic-drag \
      --attachment-span 5 \
      --drag-falloff-nodes 16 \
      --drag-x 0.018 \
      --drag-y 0.000 \
      --drag-z 0.012 \
      --ik-joints 'yaw,pitch,insertion,roll,wrist_pitch,wrist_yaw' \
      --ik-iters 60 \
      --ik-damping 0.00001 \
      --ik-max-step 0.08 \
      --ik-tol 0.0008 \
      --grasp-points-per-link 16 \
      --physics-start-frame 8 \
      --jaw-collision-radius 0.0010 \
      --jaw-min-half-height 0.0014 \
      --jaw-span-scale 0.90 \
      --show-jaw-colliders \
      --contact-stiffness 30000 \
      --contact-damping 20 \
      --friction 2.0 \
      --contact-buffer-size 8192 \
      $BENCHMARK_EXPORT_ARG
  " 2>&1 | tee "$OUT_NAME/psm_robot_thread_contact_log.txt"

echo "== Package outputs =="
tar -czf "$HOME/thread_recon_run/${OUT_NAME}.tgz" "$OUT_NAME"

echo "Outputs:"
echo "  $PROJECT_DIR/$OUT_NAME/psm_robot_thread_contact.usd"
echo "  $PROJECT_DIR/$OUT_NAME/obj_frames"
echo "  $PROJECT_DIR/$OUT_NAME/psm_robot_thread_contact_report.txt"
echo "Archive:"
echo "  $HOME/thread_recon_run/${OUT_NAME}.tgz"
