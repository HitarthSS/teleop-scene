#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/thread_recon_run/thread_reconstruction-kf}"
IMAGE="${IMAGE:-thread-reconstruction-newton:latest}"
OUT_NAME="${OUT_NAME:-thread_robot_accuracy_v4}"
SCENE_NPZ="${SCENE_NPZ:-/workspace/thread_reconstruction/newton_frame_scene_000000/newton_frame_scene.npz}"
THREAD_DIAMETER_M="${THREAD_DIAMETER_M:-0.0006}"
TOOL_INCLUDE_REGEX="${TOOL_INCLUDE_REGEX:-instruments/420006}"
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
    --mesh-alpha 0.72 \
    --list-rendered-meshes \
  2>&1 | tee "$OUT_NAME/thread_robot_reprojection_log.txt"

echo "== Newton USD/OBJ export =="
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
    python3 tools/export_thread_robot_newton_scene.py \
      --scene-npz '$SCENE_NPZ' \
      --urdf '$URDF' \
      --package-root '$PACKAGE_ROOT' \
      --joints '$JOINTS' \
      --jaw '$JAW' \
      --out-dir '/workspace/thread_reconstruction/$OUT_NAME' \
      --thread-diameter-m '$THREAD_DIAMETER_M' \
      --include-mesh-regex '$TOOL_INCLUDE_REGEX'
  " 2>&1 | tee "$OUT_NAME/thread_robot_usd_obj_log.txt"

echo "== Package outputs =="
tar -czf "$HOME/thread_recon_run/${OUT_NAME}.tgz" "$OUT_NAME"

echo "Outputs:"
echo "  $PROJECT_DIR/$OUT_NAME/thread_robot_reprojection.png"
echo "  $PROJECT_DIR/$OUT_NAME/thread_robot_newton.usd"
echo "  $PROJECT_DIR/$OUT_NAME/thread_robot_newton.obj"
echo "  $PROJECT_DIR/$OUT_NAME/thread_robot_scene_report.txt"
echo "Archive:"
echo "  $HOME/thread_recon_run/${OUT_NAME}.tgz"
