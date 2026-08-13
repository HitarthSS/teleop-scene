#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_and_render_newton_thread_docker.sh <input_samples_or_spline> [simulate args...]

Runs NVIDIA Newton on a thread_reconstruction output, then renders the saved
trajectory as 3D tube frames and a GIF.

Environment:
  DATA_ROOT      Host directory containing input recon files.
                 Default: /mnt/data/hitarth/pixal3d
  IMAGE          Docker image name.
                 Default: thread-reconstruction-newton:latest
  REBUILD        If 1, rebuild the Docker image before running.
                 Default: 0

Example:
  scripts/run_and_render_newton_thread_docker.sh \
    /home/hitarth/thread_recon_out/looping_string_move/frame_00000_1783807453123325096_samples.npz \
    --device cuda:0 --pin both
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
    usage
    exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

"${script_dir}/run_newton_thread_docker.sh" "$@"

"${script_dir}/render_newton_thread_docker.sh" \
    "${repo_root}/newton_thread_sim_good/newton_thread_simulation.npz" \
    --gif \
    --show-input

echo
echo "Newton simulation:"
echo "  ${repo_root}/newton_thread_sim_good/newton_thread_simulation.npz"
echo "Rendered views:"
echo "  ${repo_root}/newton_thread_render/frame_0000.png"
echo "  ${repo_root}/newton_thread_render/newton_thread.gif"
