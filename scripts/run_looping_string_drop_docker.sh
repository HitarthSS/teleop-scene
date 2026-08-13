#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_looping_string_drop_docker.sh [data-root]

Runs the updated thread reconstruction code on these ROS2 bag folders, then
runs NVIDIA Newton with high stiffness and no pinned endpoints:
  looping_string
  looping_string_move
  looping_string_move_two

Environment:
  IMAGE              Docker image name.
                     Default: thread-reconstruction-newton:latest
  REBUILD            If 1, rebuild Docker image first. Default: 0
  MAX_FRAMES         Offline reconstruction frame cap per bag. Default: 25
  STRIDE             Offline reconstruction stride. Default: 1
  START_INDEX        Offline reconstruction start index. Default: 0
  DEVICE             Newton/Warp device, e.g. cuda:0 or cpu. Default: cuda:0
  NUM_NODES          Newton rod nodes. Default: 65
  STEPS              Newton steps. Default: 1200
  ITERATIONS         Newton solver iterations. Default: 80
  STRETCH_STIFFNESS  Newton stretch stiffness. Default: 1.0e9
  BEND_STIFFNESS     Newton bend stiffness. Default: 1.0e7
  STRETCH_DAMPING    Newton stretch damping. Default: 1.0e4
  BEND_DAMPING       Newton bend damping. Default: 1.0e3
  GROUND             If 1, add a ground plane. Default: 0

Example:
  REBUILD=1 MAX_FRAMES=12 scripts/run_looping_string_drop_docker.sh ~/thread_recon_run
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
data_root="${1:-${DATA_ROOT:-/mnt/data/hitarth/pixal3d}}"
image="${IMAGE:-thread-reconstruction-newton:latest}"

if [[ "${REBUILD:-0}" == "1" ]] || ! docker image inspect "${image}" >/dev/null 2>&1; then
    docker build -f "${repo_root}/docker/Dockerfile.newton" -t "${image}" "${repo_root}"
fi

if [[ ! -d "${data_root}" ]]; then
    echo "Data root not found on host: ${data_root}" >&2
    exit 1
fi

ground_args=()
if [[ "${GROUND:-0}" == "1" ]]; then
    ground_args=(--ground)
fi

docker run --rm --gpus all \
    -e PYTHONPATH=/workspace/thread_reconstruction/src \
    -e MAX_FRAMES="${MAX_FRAMES:-25}" \
    -e STRIDE="${STRIDE:-1}" \
    -e START_INDEX="${START_INDEX:-0}" \
    -e DEVICE="${DEVICE:-cuda:0}" \
    -e NUM_NODES="${NUM_NODES:-65}" \
    -e STEPS="${STEPS:-1200}" \
    -e ITERATIONS="${ITERATIONS:-80}" \
    -e STRETCH_STIFFNESS="${STRETCH_STIFFNESS:-1.0e9}" \
    -e BEND_STIFFNESS="${BEND_STIFFNESS:-1.0e7}" \
    -e STRETCH_DAMPING="${STRETCH_DAMPING:-1.0e4}" \
    -e BEND_DAMPING="${BEND_DAMPING:-1.0e3}" \
    -v "${repo_root}:/workspace/thread_reconstruction" \
    -v "${data_root}:/workspace/data" \
    -w /workspace/thread_reconstruction \
    "${image}" \
    bash -lc '
set -euo pipefail
bags=(looping_string looping_string_move looping_string_move_two)
for bag in "${bags[@]}"; do
    bag_dir="/workspace/data/${bag}"
    if [[ ! -d "${bag_dir}" ]]; then
        echo "Skipping ${bag}: missing ${bag_dir}" >&2
        continue
    fi

    recon_out="/workspace/thread_reconstruction/offline_recon_out/${bag}"
    sim_root="/workspace/thread_reconstruction/newton_drop/${bag}"

    python3 tools/offline_rosbag_reconstruct.py \
        --bag "${bag_dir}" \
        --out "${recon_out}" \
        --max-frames "${MAX_FRAMES:-25}" \
        --stride "${STRIDE:-1}" \
        --start-index "${START_INDEX:-0}" \
        --time

    shopt -s nullglob
    samples=("${recon_out}"/*_samples.npz)
    if (( ${#samples[@]} == 0 )); then
        echo "No accepted reconstruction samples for ${bag}; skipping Newton." >&2
        continue
    fi

    for sample in "${samples[@]}"; do
        stem="$(basename "${sample}" _samples.npz)"
        out_dir="${sim_root}/${stem}"
        python3 tools/simulate_newton_thread.py "${sample}" \
            --out-dir "${out_dir}" \
            --device "${DEVICE}" \
            --num-nodes "${NUM_NODES}" \
            --steps "${STEPS}" \
            --iterations "${ITERATIONS}" \
            --stretch-stiffness "${STRETCH_STIFFNESS}" \
            --bend-stiffness "${BEND_STIFFNESS}" \
            --stretch-damping "${STRETCH_DAMPING}" \
            --bend-damping "${BEND_DAMPING}" \
            --pin none \
            --preview '"${ground_args[*]}"'
    done
done
'
