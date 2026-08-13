#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_newton_thread_docker.sh <input-relative-to-data-root-or-absolute> [simulate_newton_thread.py args...]

Environment:
  DATA_ROOT      Host directory containing input recon files.
                 Default: /mnt/data/hitarth/pixal3d
  IMAGE          Docker image name.
                 Default: thread-reconstruction-newton:latest
  REBUILD        If 1, rebuild the Docker image before running.
                 Default: 0

Examples:
  DATA_ROOT=/mnt/data/hitarth/pixal3d scripts/run_newton_thread_docker.sh \
    outputs/good_frame_samples.npz --device cuda:0 --pin both --ground

  scripts/run_newton_thread_docker.sh \
    /mnt/data/hitarth/pixal3d/outputs/good_frame_samples.npz \
    --device cuda:0 --out-dir /workspace/thread_reconstruction/newton_thread_sim_good
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
    usage
    exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
data_root="${DATA_ROOT:-/mnt/data/hitarth/pixal3d}"
image="${IMAGE:-thread-reconstruction-newton:latest}"
input_arg="$1"
shift

if [[ "${REBUILD:-0}" == "1" ]] || ! docker image inspect "${image}" >/dev/null 2>&1; then
    docker build -f "${repo_root}/docker/Dockerfile.newton" -t "${image}" "${repo_root}"
fi

if [[ "${input_arg}" = /* ]]; then
    input_abs="${input_arg}"
else
    input_abs="${data_root%/}/${input_arg}"
fi

if [[ ! -f "${input_abs}" ]]; then
    echo "Input file not found on host: ${input_abs}" >&2
    exit 1
fi

input_parent="$(cd "$(dirname "${input_abs}")" && pwd)"
input_base="$(basename "${input_abs}")"

docker run --rm --gpus all \
    -v "${repo_root}:/workspace/thread_reconstruction" \
    -v "${input_parent}:/workspace/input:ro" \
    -w /workspace/thread_reconstruction \
    "${image}" \
    python3 tools/simulate_newton_thread.py "/workspace/input/${input_base}" \
    --out-dir /workspace/thread_reconstruction/newton_thread_sim_good \
    "$@"
