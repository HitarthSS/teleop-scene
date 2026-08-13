#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/thread_recon_run/thread_reconstruction-kf}"
IMAGE="${IMAGE:-thread-reconstruction-newton:latest}"
SCENE_NPZ="${SCENE_NPZ:-/workspace/thread_reconstruction/newton_frame_scene_000000/newton_frame_scene.npz}"
OUT_NAME="${OUT_NAME:-newton_thread_cloth_pad_v15}"
OBJ_NAME="${OBJ_NAME:-newton_thread_cloth_pad_v15_obj}"

cd "$PROJECT_DIR"

echo "== Newton cloth-pad simulation =="
docker run --rm --gpus all \
  -e PYTHONPATH=/workspace/thread_reconstruction:/workspace/thread_reconstruction/src \
  -v "$PWD:/workspace/thread_reconstruction" \
  -w /workspace/thread_reconstruction \
  "$IMAGE" \
  python3 tools/simulate_reconstructed_thread_newton_physics.py \
    --scene-npz "$SCENE_NPZ" \
    --out-dir "/workspace/thread_reconstruction/$OUT_NAME" \
    --steps 60 \
    --dt 0.0008333333 \
    --save-every 3 \
    --iterations 80 \
    --num-nodes 64 \
    --drop-height 0.003 \
    --pad-mode cloth \
    --contact-stiffness 120000 \
    --contact-damping 25 \
    --cloth-tri-ke 20000 \
    --cloth-tri-ka 20000 \
    --cloth-tri-kd 20 \
    --cloth-edge-ke 1.0 \
    --cloth-edge-kd 0.05 \
    --cloth-particle-radius 0.0015 \
    --contact-buffer-size 4096 \
    --render-pad-margin 0.025 \
    --no-ground \
    --no-closed

echo "== Debug report =="
docker run --rm \
  -e PYTHONPATH=/workspace/thread_reconstruction:/workspace/thread_reconstruction/src \
  -v "$PWD:/workspace/thread_reconstruction" \
  -w /workspace/thread_reconstruction \
  "$IMAGE" \
  python3 tools/debug_newton_render_scene.py \
    --scene-npz "/workspace/thread_reconstruction/$OUT_NAME/reconstructed_thread_newton_physics.npz" \
    --out "/workspace/thread_reconstruction/$OUT_NAME/debug_report.txt"

echo "== Export OBJ sequence =="
docker run --rm --gpus all \
  -e PYTHONPATH=/workspace/thread_reconstruction:/workspace/thread_reconstruction/src \
  -v "$PWD:/workspace/thread_reconstruction" \
  -v "$HOME:/workspace/home" \
  -w /workspace/thread_reconstruction \
  "$IMAGE" \
  python3 tools/export_frame_scene_obj_sequence.py \
    --scene-npz "/workspace/thread_reconstruction/$OUT_NAME/reconstructed_thread_newton_physics.npz" \
    --urdf /workspace/home/thread_recon_run/assets/dvrk/psm1_si.urdf \
    --package-root /workspace/home/thread_recon_run/assets \
    --joints /workspace/thread_reconstruction/thread_with_gripper_pose_frames/episode_0000_inputs/joint_000000.npy \
    --jaw /workspace/thread_reconstruction/thread_with_gripper_pose_frames/episode_0000_inputs/jaw_000000.npy \
    --out-dir "/workspace/thread_reconstruction/$OBJ_NAME" \
    --stride 1 \
    --max-frames 20 \
    --thread-radius 0.003 \
    --thread-visual-offset 0.002 \
    --include-mesh-regex 'tool_wrist|tool_wrist_sca' \
    --exclude-mesh-regex 'tool_main'

echo "== Package for Unicorn =="
tar -czf "$HOME/thread_recon_run/${OBJ_NAME}.tgz" \
  "$OBJ_NAME" \
  "$OUT_NAME/debug_report.txt" \
  tools/render_scene_obj_genesis_sequence.py \
  tools/render_scene_obj_genesis.py \
  tools/run_unicorn_render_cloth_pad_v15.sh

echo "Wrote: $HOME/thread_recon_run/${OBJ_NAME}.tgz"
echo "Copy to Unicorn with:"
echo "scp $HOME/thread_recon_run/${OBJ_NAME}.tgz hitarth@unicorn1.ucsd.edu:~/stereo_dataset/"
