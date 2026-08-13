#!/usr/bin/env python3
"""Render saved NVIDIA Newton thread simulation states as 3D tube frames.

This is an offline visualizer for ``newton_thread_simulation.npz``. It does not
rerun physics; it renders the saved rod centerline states as a thick tube so the
result looks like a simulated object rather than a line plot.
"""

import argparse
import math
from pathlib import Path

import numpy as np


def set_mpl_cache():
    import os
    import tempfile

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "thread_reconstruction_matplotlib"),
    )


def make_frame_indices(num_states, requested):
    if requested <= 0 or requested >= num_states:
        return np.arange(num_states, dtype=int)
    return np.unique(np.linspace(0, num_states - 1, requested).round().astype(int))


def orthonormal_frames(points):
    tangents = np.gradient(points, axis=0)
    nrm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(nrm, 1e-12)

    normals = np.zeros_like(points)
    binormals = np.zeros_like(points)
    up = np.array([0.0, 0.0, 1.0])
    alt = np.array([0.0, 1.0, 0.0])

    for i, t in enumerate(tangents):
        ref = up if abs(float(np.dot(t, up))) < 0.9 else alt
        n = np.cross(t, ref)
        n = n / max(np.linalg.norm(n), 1e-12)
        b = np.cross(t, n)
        b = b / max(np.linalg.norm(b), 1e-12)
        normals[i] = n
        binormals[i] = b
    return normals, binormals


def tube_mesh(points, radius, sides):
    normals, binormals = orthonormal_frames(points)
    theta = np.linspace(0.0, 2.0 * math.pi, sides + 1)
    circle = np.cos(theta)[:, None] * normals[:, None, :] + np.sin(theta)[:, None] * binormals[:, None, :]
    rings = points[:, None, :] + radius * circle
    return rings[:, :, 0], rings[:, :, 1], rings[:, :, 2]


def axis_limits(states, radius):
    pts = states.reshape(-1, 3)
    lo = pts.min(axis=0) - 3.0 * radius
    hi = pts.max(axis=0) + 3.0 * radius
    ctr = 0.5 * (lo + hi)
    span = float(np.max(hi - lo))
    span = max(span, 1e-6)
    lo = ctr - 0.5 * span
    hi = ctr + 0.5 * span
    return lo, hi


def render_frame(ax, state, initial, drive_trajectory, frame_i, radius, sides, limits, title):
    ax.clear()
    x, y, z = tube_mesh(state, radius, sides)
    ax.plot_surface(x, y, z, color="#d62728", linewidth=0, antialiased=True, shade=True, alpha=0.96)

    if initial is not None:
        ax.plot(initial[:, 0], initial[:, 1], initial[:, 2], color="0.45", linewidth=1.2, alpha=0.45)
    if drive_trajectory is not None:
        upto = min(frame_i + 1, len(drive_trajectory))
        ax.plot(
            drive_trajectory[:upto, 0],
            drive_trajectory[:upto, 1],
            drive_trajectory[:upto, 2],
            color="#1f77b4",
            linewidth=2.0,
            alpha=0.85,
        )

    ax.scatter(state[0, 0], state[0, 1], state[0, 2], color="#00d5ff", s=45, depthshade=False)
    ax.scatter(state[-1, 0], state[-1, 1], state[-1, 2], color="#ffd400", s=45, depthshade=False)

    lo, hi = limits
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("simulation", help="newton_thread_simulation.npz")
    parser.add_argument("--out-dir", default="newton_thread_render")
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--tube-sides", type=int, default=16)
    parser.add_argument("--radius", type=float, default=None, help="tube radius in meters; defaults to saved radius")
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--show-input", action="store_true", help="overlay the original reconstruction centerline")
    parser.add_argument("--gif", action="store_true", help="also write newton_thread.gif")
    parser.add_argument("--gif-fps", type=int, default=12)
    args = parser.parse_args()

    set_mpl_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    data = np.load(args.simulation, allow_pickle=True)
    states = np.asarray(data["states"], dtype=float)
    times = np.asarray(data["times"], dtype=float)
    initial = np.asarray(data["initial_centerline"], dtype=float) if args.show_input else None
    radius = float(args.radius if args.radius is not None else data.get("radius", 0.001))
    indices = make_frame_indices(len(states), args.num_frames)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drive_trajectory = np.asarray(data["drive_trajectory"], dtype=float) if "drive_trajectory" in data.files else None
    limit_states = states[indices]
    if drive_trajectory is not None:
        limit_states = np.concatenate([limit_states.reshape(-1, 3), drive_trajectory], axis=0)
    limits = axis_limits(limit_states, radius)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    saved_paths = []
    for out_i, state_i in enumerate(indices):
        t = times[state_i] if state_i < len(times) else float(state_i)
        render_frame(
            ax,
            states[state_i],
            initial,
            drive_trajectory,
            state_i,
            radius,
            args.tube_sides,
            limits,
            title=f"NVIDIA Newton thread simulation  t={t:.3f}s",
        )
        ax.view_init(elev=args.elev, azim=args.azim)
        fig.tight_layout()
        path = out_dir / f"frame_{out_i:04d}.png"
        fig.savefig(path, dpi=args.dpi)
        saved_paths.append(path)

    if args.gif:
        def update(anim_i):
            state_i = int(indices[anim_i])
            t = times[state_i] if state_i < len(times) else float(state_i)
            render_frame(
                ax,
                states[state_i],
                initial,
                drive_trajectory,
                state_i,
                radius,
                args.tube_sides,
                limits,
                title=f"NVIDIA Newton thread simulation  t={t:.3f}s",
            )
            ax.view_init(elev=args.elev, azim=args.azim)
            return []

        from matplotlib.animation import FuncAnimation

        anim = FuncAnimation(fig, update, frames=len(indices), interval=1000 / args.gif_fps, blit=False)
        anim.save(out_dir / "newton_thread.gif", writer=PillowWriter(fps=args.gif_fps), dpi=args.dpi)

    plt.close(fig)
    print(f"rendered {len(saved_paths)} frame(s) to {out_dir}")
    if args.gif:
        print(f"wrote {out_dir / 'newton_thread.gif'}")


if __name__ == "__main__":
    main()
