#!/usr/bin/env python3
"""Live matplotlib viewer for the UDP thread teleop state stream."""

from __future__ import annotations

import argparse
import json
import socket
import time

import numpy as np


def equal_axes(ax, points, pad=0.01):
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = max(float(np.max(hi - lo)) * 0.5 + pad, pad)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8765)
    parser.add_argument("--local-port", type=int, default=0)
    parser.add_argument("--subscribe-hz", type=float, default=10.0)
    parser.add_argument("--draw-hz", type=float, default=30.0)
    parser.add_argument("--fixed-bounds", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trail", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", int(args.local_port)))
    sock.setblocking(False)
    server = (args.server_host, int(args.server_port))

    plt.ion()
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    line, = ax.plot([], [], [], color="tab:blue", linewidth=3, label="thread")
    start_scatter = ax.scatter([], [], [], color="cyan", s=45, label="start")
    end_scatter = ax.scatter([], [], [], color="gold", s=45, label="end")
    target_scatter = ax.scatter([], [], [], color="magenta", s=55, marker="x", label="target")
    jaw_scatter = ax.scatter([], [], [], color="black", s=45, label="jaw grasp")
    trail_line, = ax.plot([], [], [], color="tab:red", linewidth=1, alpha=0.65, label="grabbed-end trail")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper right")
    ax.set_title("Live Newton Thread Teleop")

    last_sub = 0.0
    last_draw = 0.0
    state = None
    trail = []
    fixed_points = None

    print(f"viewer subscribing to udp://{args.server_host}:{args.server_port}")
    print("close the plot window or press Ctrl+C to stop")

    while plt.fignum_exists(fig.number):
        now = time.perf_counter()
        if now - last_sub > 1.0 / max(float(args.subscribe_hz), 1.0):
            msg = {"type": "viewer_subscribe", "subscribe": True}
            sock.sendto(json.dumps(msg, separators=(",", ":")).encode("utf-8"), server)
            last_sub = now

        while True:
            try:
                data, _addr = sock.recvfrom(65507)
                state = json.loads(data.decode("utf-8"))
            except BlockingIOError:
                break

        if state is not None and now - last_draw > 1.0 / max(float(args.draw_hz), 1.0):
            nodes = np.asarray(state.get("thread_nodes_newton", []), dtype=np.float64)
            if nodes.ndim == 2 and nodes.shape[1] == 3 and len(nodes) > 0:
                line.set_data(nodes[:, 0], nodes[:, 1])
                line.set_3d_properties(nodes[:, 2])
                start_scatter._offsets3d = ([nodes[0, 0]], [nodes[0, 1]], [nodes[0, 2]])
                end_scatter._offsets3d = ([nodes[-1, 0]], [nodes[-1, 1]], [nodes[-1, 2]])

                target = np.asarray(state.get("target_newton", [np.nan, np.nan, np.nan]), dtype=np.float64)
                jaw = np.asarray(state.get("jaw_grasp_newton", [np.nan, np.nan, np.nan]), dtype=np.float64)
                target_scatter._offsets3d = ([target[0]], [target[1]], [target[2]])
                jaw_scatter._offsets3d = ([jaw[0]], [jaw[1]], [jaw[2]])

                idx = int(state.get("target_thread_idx", 0))
                idx = min(max(idx, 0), len(nodes) - 1)
                if args.trail:
                    trail.append(nodes[idx].copy())
                    trail[:] = trail[-300:]
                    trail_np = np.asarray(trail, dtype=np.float64)
                    trail_line.set_data(trail_np[:, 0], trail_np[:, 1])
                    trail_line.set_3d_properties(trail_np[:, 2])

                if fixed_points is None:
                    fixed_points = np.vstack([nodes, target.reshape(1, 3), jaw.reshape(1, 3)])
                bounds_points = fixed_points if args.fixed_bounds else np.vstack([nodes, target.reshape(1, 3), jaw.reshape(1, 3)])
                equal_axes(ax, bounds_points)

                ax.set_title(
                    "Live Newton Thread Teleop | "
                    f"grip={state.get('grip')} "
                    f"disp={state.get('target_displacement_m', 0.0):.4f} m "
                    f"update={1000.0 * state.get('perf', {}).get('update_seconds', 0.0):.2f} ms"
                )
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                last_draw = now

        plt.pause(0.001)


if __name__ == "__main__":
    main()
