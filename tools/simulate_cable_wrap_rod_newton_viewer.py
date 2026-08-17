#!/usr/bin/env python3
"""Newton viewer animation of a cable wrapping around a cylinder rod.

This is a controlled-trajectory demo: the rod is fixed and the cable centerline
is procedurally driven into a helix around it. It is useful for checking cable
visualization, scale, and viewer behavior before adding a contact solver.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from export_frame_scene_obj import tube_mesh, write_obj
from export_thread_robot_newton_scene import make_empty_newton_model, make_usd_viewer


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1.0e-12)


def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def cylinder_mesh(radius, length, sides=48, segments=8, center=(0.0, 0.0, 0.0)):
    center = np.asarray(center, dtype=np.float64)
    verts = []
    faces = []
    for i in range(segments + 1):
        z = -0.5 * length + length * i / max(segments, 1)
        for j in range(sides):
            theta = 2.0 * math.pi * j / sides
            verts.append(center + np.asarray([radius * math.cos(theta), radius * math.sin(theta), z]))

    for i in range(segments):
        a0 = i * sides
        a1 = (i + 1) * sides
        for j in range(sides):
            a = a0 + j
            b = a0 + (j + 1) % sides
            c = a1 + (j + 1) % sides
            d = a1 + j
            faces.append([a, b, c])
            faces.append([a, c, d])

    bottom_center = len(verts)
    top_center = bottom_center + 1
    verts.append(center + np.asarray([0.0, 0.0, -0.5 * length]))
    verts.append(center + np.asarray([0.0, 0.0, 0.5 * length]))
    top_base = segments * sides
    for j in range(sides):
        faces.append([bottom_center, (j + 1) % sides, j])
        faces.append([top_center, top_base + j, top_base + (j + 1) % sides])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def cable_path(frame_alpha, args):
    """Generate a cable that progressively wraps around the z-axis rod."""
    n = int(args.nodes)
    s = np.linspace(0.0, 1.0, n)
    alpha = smoothstep(frame_alpha)

    rod_clearance = float(args.rod_radius) + float(args.cable_radius) * float(args.clearance_radius_multiplier)
    start_x = rod_clearance + float(args.start_gap)
    lead_x = float(args.lead_length)
    total_theta = 2.0 * math.pi * float(args.wrap_turns)

    points = np.zeros((n, 3), dtype=np.float64)
    for i, u in enumerate(s):
        wrap_u = np.clip((u - float(args.lead_fraction)) / max(1.0 - float(args.lead_fraction), 1.0e-6), 0.0, 1.0)
        wrap_weight = smoothstep(wrap_u)
        theta = alpha * total_theta * wrap_u
        radius = (1.0 - alpha * wrap_weight) * start_x + (alpha * wrap_weight) * rod_clearance
        z = -0.5 * float(args.rod_length) + float(args.rod_length) * wrap_u
        helix = np.asarray([radius * math.cos(theta), radius * math.sin(theta), z], dtype=np.float64)

        # The leading tail starts very close to the rod, then feeds into the
        # helix smoothly so the first frames already show near-contact.
        lead = np.asarray([start_x + lead_x * (float(args.lead_fraction) - u), 0.0, -0.5 * float(args.rod_length)], dtype=np.float64)
        tail_blend = smoothstep(u / max(float(args.lead_fraction), 1.0e-6))
        points[i] = (1.0 - tail_blend) * lead + tail_blend * helix

    # Add a small axial travel as it wraps, so the animation reads as a snake
    # sliding around the rod rather than simply shrinking in place.
    points[:, 2] += alpha * float(args.axial_slide)
    return points


def log_mesh(viewer, wp, name, verts, faces, color, roughness=0.5, metallic=0.0):
    points_wp = wp.array(np.asarray(verts, dtype=np.float32), dtype=wp.vec3)
    indices_wp = wp.array(np.asarray(faces, dtype=np.int32).reshape(-1), dtype=wp.int32)
    try:
        viewer.log_mesh(
            name,
            points_wp,
            indices_wp,
            color=color,
            roughness=roughness,
            metallic=metallic,
            backface_culling=False,
        )
    except TypeError:
        viewer.log_mesh(name, points_wp, indices_wp, backface_culling=False)


def write_frame_obj(path, rod_v, rod_f, cable_v, cable_f):
    write_obj(
        Path(path),
        [
            ("rod", "tool", rod_v, rod_f),
            ("cable", "thread", cable_v, cable_f),
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--nodes", type=int, default=96)
    parser.add_argument("--cable-radius", type=float, default=0.0015)
    parser.add_argument("--cable-sides", type=int, default=16)
    parser.add_argument("--rod-radius", type=float, default=0.012)
    parser.add_argument("--rod-length", type=float, default=0.10)
    parser.add_argument("--rod-sides", type=int, default=64)
    parser.add_argument("--rod-segments", type=int, default=12)
    parser.add_argument("--wrap-turns", type=float, default=2.25)
    parser.add_argument("--start-gap", type=float, default=0.003)
    parser.add_argument("--lead-length", type=float, default=0.030)
    parser.add_argument("--lead-fraction", type=float, default=0.18)
    parser.add_argument("--clearance-radius-multiplier", type=float, default=1.15)
    parser.add_argument("--axial-slide", type=float, default=0.010)
    parser.add_argument("--save-obj-frames", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_dir = out_dir / "obj_frames"
    if args.save_obj_frames:
        obj_dir.mkdir(parents=True, exist_ok=True)

    import warp as wp

    model, state = make_empty_newton_model()
    usd_path = out_dir / "cable_wrap_rod_newton.usd"
    viewer = make_usd_viewer(usd_path, args.fps)
    viewer.set_model(model)

    rod_v, rod_f = cylinder_mesh(
        args.rod_radius,
        args.rod_length,
        sides=args.rod_sides,
        segments=args.rod_segments,
    )

    states = []
    frame_count = max(int(args.frames), 2)
    for frame in range(frame_count):
        t = frame / float(frame_count - 1)
        cable = cable_path(t, args)
        states.append(cable)
        cable_v, cable_f = tube_mesh(cable, args.cable_radius, args.cable_sides)

        viewer.begin_frame(frame / float(args.fps))
        viewer.log_state(state)
        log_mesh(viewer, wp, "/wrap_demo/rod", rod_v, rod_f, color=(0.18, 0.20, 0.22), roughness=0.28, metallic=0.6)
        log_mesh(viewer, wp, "/wrap_demo/cable", cable_v, cable_f, color=(0.86, 0.82, 0.68), roughness=0.52)
        viewer.end_frame()

        if args.save_obj_frames:
            write_frame_obj(obj_dir / f"frame_{frame:04d}.obj", rod_v, rod_f, cable_v, cable_f)

    viewer.close()

    npz_path = out_dir / "cable_wrap_rod_states.npz"
    np.savez(
        npz_path,
        states=np.asarray(states, dtype=np.float64),
        rod_radius=np.asarray(args.rod_radius, dtype=np.float64),
        rod_length=np.asarray(args.rod_length, dtype=np.float64),
        cable_radius=np.asarray(args.cable_radius, dtype=np.float64),
        fps=np.asarray(args.fps, dtype=np.float64),
    )

    report = out_dir / "cable_wrap_rod_report.txt"
    report.write_text(
        "\n".join(
            [
                "script: simulate_cable_wrap_rod_newton_viewer.py",
                f"usd: {usd_path}",
                f"npz: {npz_path}",
                f"frames: {frame_count}",
                f"fps: {args.fps}",
                f"nodes: {args.nodes}",
                f"rod_radius_m: {args.rod_radius}",
                f"rod_length_m: {args.rod_length}",
                f"cable_radius_m: {args.cable_radius}",
                f"wrap_turns: {args.wrap_turns}",
                f"start_gap_m: {args.start_gap}",
                "mode: controlled trajectory, not contact dynamics",
            ]
        )
        + "\n"
    )

    print(f"USD output: {usd_path}")
    print(f"NPZ states: {npz_path}")
    if args.save_obj_frames:
        print(f"OBJ frames: {obj_dir}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
