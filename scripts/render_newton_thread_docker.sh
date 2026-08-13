#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  scripts/render_newton_thread_docker.sh <newton_thread_simulation.npz> [render_newton_thread.py args...]

Environment:
  IMAGE          Docker image name.
                 Default: thread-reconstruction-newton:latest

Example:
  scripts/render_newton_thread_docker.sh \
    newton_thread_sim_good/newton_thread_simulation.npz \
    --gif --show-input
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
    usage
    exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
image="${IMAGE:-thread-reconstruction-newton:latest}"
input_arg="$1"
shift

if [[ "${input_arg}" = /* ]]; then
    input_abs="${input_arg}"
else
    input_abs="${repo_root%/}/${input_arg}"
fi

if [[ ! -f "${input_abs}" ]]; then
    echo "Simulation file not found on host: ${input_abs}" >&2
    exit 1
fi

input_parent="$(cd "$(dirname "${input_abs}")" && pwd)"
input_base="$(basename "${input_abs}")"

docker run --rm --gpus all \
    -v "${repo_root}:/workspace/thread_reconstruction" \
    -v "${input_parent}:/workspace/input:ro" \
    -w /workspace/thread_reconstruction \
    "${image}" \
    python3 tools/render_newton_thread.py "/workspace/input/${input_base}" \
    --out-dir /workspace/thread_reconstruction/newton_thread_render \
    "$@"
