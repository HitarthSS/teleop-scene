#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/thread_recon_run/thread_reconstruction-kf}"
IMAGE="${IMAGE:-thread-reconstruction-newton:latest}"
OUT_NAME="${OUT_NAME:-thread_gripper_contact_v2}"
SCENE_NPZ="${SCENE_NPZ:-/workspace/thread_reconstruction/newton_frame_scene_000000/newton_frame_scene.npz}"
THREAD_DIAMETER_M="${THREAD_DIAMETER_M:-0.0006}"
TOOL_INCLUDE_REGEX="${TOOL_INCLUDE_REGEX:-tool_wrist|tool_wrist_sca}"
TOOL_EXCLUDE_REGEX="${TOOL_EXCLUDE_REGEX:-tool_main_link}"
URDF="${URDF:-/workspace/home/thread_recon_run/assets/dvrk/psm1_si.urdf}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/workspace/home/thread_recon_run/assets}"
JOINTS="${JOINTS:-/workspace/thread_reconstruction/thread_with_gripper_pose_frames/episode_0000_inputs/joint_000000.npy}"
JAW="${JAW:-/workspace/thread_reconstruction/thread_with_gripper_pose_frames/episode_0000_inputs/jaw_000000.npy}"
IMAGE_PATH="${IMAGE_PATH:-/workspace/thread_reconstruction/thread_with_gripper_pose_frames/episode_0000/colors/left_image_000000.jpg}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_NAME"

echo "== 2D camera reprojection check =="
docker run --rm \
  -e PYTHONPATH=/workspace/thread_reconstruction:/workspace/thread_reconstruction/src \
  -v "$PWD:/workspace/thread_reconstruction" \
  -v "$HOME:/workspace/home" \
  -w /workspace/thread_reconstruction \
  "$IMAGE" \
  python3 tools/render_frame_scene_with_psm_urdf.py \
    --scene-npz "$SCENE_NPZ" \
    --urdf "$URDF" \
    --package-root "$PACKAGE_ROOT" \
    --joints "$JOINTS" \
    --jaw "$JAW" \
    --image "$IMAGE_PATH" \
    --out "/workspace/thread_reconstruction/$OUT_NAME/thread_robot_reprojection.png" \
    --thread-diameter-m "$THREAD_DIAMETER_M" \
    --include-mesh-regex "$TOOL_INCLUDE_REGEX" \
    --exclude-mesh-regex "$TOOL_EXCLUDE_REGEX" \
    --mesh-alpha 0.72 \
    --list-rendered-meshes \
  2>&1 | tee "$OUT_NAME/thread_robot_reprojection_log.txt"

echo "== Resting pad + collision-enabled jaw grab =="
docker run --rm --gpus all \
  -e PYTHONPATH=/workspace/thread_reconstruction:/workspace/thread_reconstruction/src \
  -v "$PWD:/workspace/thread_reconstruction" \
  -v "$HOME:/workspace/home" \
  -w /workspace/thread_reconstruction \
  "$IMAGE" \
  bash -lc "
    set -euo pipefail
    set +e
    python3 - <<'PY'
try:
    from pxr import Usd
    print('usd-core already installed')
except Exception:
    raise SystemExit(42)
PY
    status=\$?
    set -e
    if [ \"\$status\" -eq 42 ]; then
      python3 -m pip install --no-cache-dir usd-core
    elif [ \"\$status\" -ne 0 ]; then
      exit \"\$status\"
    fi

    mkdir -p '/workspace/thread_reconstruction/$OUT_NAME/rest'
    python3 tools/export_thread_pad_resting_scene.py \
      --scene-npz '$SCENE_NPZ' \
      --urdf '$URDF' \
      --package-root '$PACKAGE_ROOT' \
      --joints '$JOINTS' \
      --jaw '$JAW' \
      --out-dir '/workspace/thread_reconstruction/$OUT_NAME/rest' \
      --thread-diameter-m '$THREAD_DIAMETER_M' \
      --include-mesh-regex '$TOOL_INCLUDE_REGEX' \
      --exclude-mesh-regex '$TOOL_EXCLUDE_REGEX' \
      --thread-side -1 \
      --pad-nx 31 \
      --pad-ny 31 \
      --pad-margin 0.002 \
      --pad-min-half 0.003

    python3 tools/simulate_gripper_grab_thread_preview.py \
      --rest-scene-npz '/workspace/thread_reconstruction/$OUT_NAME/rest/thread_on_pad_resting_scene.npz' \
      --urdf '$URDF' \
      --package-root '$PACKAGE_ROOT' \
      --joints '$JOINTS' \
      --jaw '$JAW' \
      --out-dir '/workspace/thread_reconstruction/$OUT_NAME/grab' \
      --include-mesh-regex '$TOOL_INCLUDE_REGEX' \
      --exclude-mesh-regex '$TOOL_EXCLUDE_REGEX' \
      --interaction-mode collision-jaws \
      --frames 18 \
      --fps 12 \
      --substeps 8 \
      --iterations 60 \
      --gravity 0.0 \
      --drag-distance 0.010 \
      --lift-distance 0.003 \
      --close-frames 6 \
      --jaw-open-separation 0.006 \
      --jaw-closed-separation 0.0009 \
      --jaw-collision-radius 0.0011 \
      --jaw-collision-samples 7 \
      --contact-stiffness 200000 \
      --contact-damping 10 \
      --friction 2.0 \
      --contact-buffer-size 8192
  " 2>&1 | tee "$OUT_NAME/gripper_contact_log.txt"

echo "== Package outputs =="
tar -czf "$HOME/thread_recon_run/${OUT_NAME}.tgz" "$OUT_NAME"

echo "Outputs:"
echo "  $PROJECT_DIR/$OUT_NAME/thread_robot_reprojection.png"
echo "  $PROJECT_DIR/$OUT_NAME/rest/thread_on_pad_resting_newton.usd"
echo "  $PROJECT_DIR/$OUT_NAME/grab/thread_gripper_grab_preview.usd"
echo "  $PROJECT_DIR/$OUT_NAME/grab/obj_frames"
echo "  $PROJECT_DIR/$OUT_NAME/grab/thread_gripper_grab_report.txt"
echo "Archive:"
echo "  $HOME/thread_recon_run/${OUT_NAME}.tgz"
