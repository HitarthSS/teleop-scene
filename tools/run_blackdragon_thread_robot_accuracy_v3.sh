#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/thread_recon_run/thread_reconstruction-kf}"
IMAGE="${IMAGE:-thread-reconstruction-newton:latest}"
OUT_NAME="${OUT_NAME:-thread_robot_accuracy_v3}"
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

echo "== Newton GL renders =="
docker run --rm --gpus all \
  -e PYOPENGL_PLATFORM=egl \
  -e PYTHONPATH=/workspace/thread_reconstruction:/workspace/thread_reconstruction/src \
  -v "$PWD:/workspace/thread_reconstruction" \
  -v "$HOME:/workspace/home" \
  -w /workspace/thread_reconstruction \
  "$IMAGE" \
  bash -lc "
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    if ! ldconfig -p 2>/dev/null | grep -q 'libGL.so.1'; then
      apt-get update
      apt-get install -y --no-install-recommends \
        libgl1 \
        libglx0 \
        libegl1 \
        libgles2 \
        libx11-6 \
        libxext6 \
        libxrender1
    fi
    set +e
    python3 - <<'PY'
try:
    import pyglet
    print('pyglet already installed:', pyglet.version)
except Exception:
    raise SystemExit(42)
PY
    status=\$?
    set -e
    if [ \"\$status\" -eq 42 ]; then
      python3 -m pip install --no-cache-dir 'pyglet>=2.0'
    elif [ \"\$status\" -ne 0 ]; then
      exit \"\$status\"
    fi
    python3 tools/render_thread_robot_newton_gl.py \
      --scene-npz '$SCENE_NPZ' \
      --urdf '$URDF' \
      --package-root '$PACKAGE_ROOT' \
      --joints '$JOINTS' \
      --jaw '$JAW' \
      --out '/workspace/thread_reconstruction/$OUT_NAME/newton_thread_robot_camera.png' \
      --report '/workspace/thread_reconstruction/$OUT_NAME/newton_thread_robot_camera_report.txt' \
      --thread-diameter-m '$THREAD_DIAMETER_M' \
      --include-mesh-regex '$TOOL_INCLUDE_REGEX' \
      --view camera \
      --distance-scale 2.2 \
      --width 960 \
      --height 720 \
      --fov 38
    python3 tools/render_thread_robot_newton_gl.py \
      --scene-npz '$SCENE_NPZ' \
      --urdf '$URDF' \
      --package-root '$PACKAGE_ROOT' \
      --joints '$JOINTS' \
      --jaw '$JAW' \
      --out '/workspace/thread_reconstruction/$OUT_NAME/newton_thread_robot_oblique.png' \
      --report '/workspace/thread_reconstruction/$OUT_NAME/newton_thread_robot_oblique_report.txt' \
      --thread-diameter-m '$THREAD_DIAMETER_M' \
      --include-mesh-regex '$TOOL_INCLUDE_REGEX' \
      --view oblique \
      --distance-scale 2.2 \
      --width 960 \
      --height 720 \
      --fov 38
  " 2>&1 | tee "$OUT_NAME/newton_thread_robot_gl_log.txt"

echo "== Package outputs =="
tar -czf "$HOME/thread_recon_run/${OUT_NAME}.tgz" "$OUT_NAME"

echo "Outputs:"
echo "  $PROJECT_DIR/$OUT_NAME/thread_robot_reprojection.png"
echo "  $PROJECT_DIR/$OUT_NAME/newton_thread_robot_camera.png"
echo "  $PROJECT_DIR/$OUT_NAME/newton_thread_robot_oblique.png"
echo "  $PROJECT_DIR/$OUT_NAME/newton_thread_robot_camera_report.txt"
echo "Archive:"
echo "  $HOME/thread_recon_run/${OUT_NAME}.tgz"
