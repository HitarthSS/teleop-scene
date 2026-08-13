#!/usr/bin/env python3
"""Run a gravity drop from a one-frame camera scene.

The thread reconstruction is in OpenCV camera coordinates. This script fits a
pad/tissue plane to the reconstructed thread, offsets that plane below the
thread, maps it to Newton's z=0 ground plane, and lets gravity act along the
negative fitted-plane normal. That keeps the pad parallel to the thread/tissue
surface instead of making it a vertical wall in camera space.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from generate_newton_frame_scene import draw_polyline_camera, load_rgb_image, make_synthetic_pad
from simulate_newton_thread import parse_vec3, simulate_newton


def fit_surface_frame(points, preferred_normal, flip_normal=False):
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    preferred = np.asarray(preferred_normal, dtype=float)
    preferred = preferred / max(np.linalg.norm(preferred), 1.0e-12)
    if np.dot(normal, preferred) < 0.0:
        normal = -normal
    if flip_normal:
        normal = -normal
    normal = normal / max(np.linalg.norm(normal), 1.0e-12)

    x_axis = np.asarray([1.0, 0.0, 0.0])
    x_axis = x_axis - normal * np.dot(x_axis, normal)
    if np.linalg.norm(x_axis) < 1.0e-9:
        x_axis = np.asarray([0.0, 1.0, 0.0])
        x_axis = x_axis - normal * np.dot(x_axis, normal)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1.0e-12)
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1.0e-12)
    return center, x_axis, y_axis, normal


def camera_to_newton_plane(points, center, x_axis, y_axis, normal, plane_signed):
    pts = np.asarray(points, dtype=float)
    rel = pts - center
    out = np.empty_like(pts)
    out[..., 0] = rel @ x_axis
    out[..., 1] = rel @ y_axis
    out[..., 2] = rel @ normal - plane_signed
    return out


def newton_to_camera_plane(points, center, x_axis, y_axis, normal, plane_signed):
    pts = np.asarray(points, dtype=float)
    out = (
        center
        + pts[..., 0, None] * x_axis
        + pts[..., 1, None] * y_axis
        + (pts[..., 2, None] + plane_signed) * normal
    )
    return out


def make_sim_args(args):
    return SimpleNamespace(
        device=args.device,
        gravity=args.gravity,
        contact_stiffness=args.contact_stiffness,
        contact_damping=args.contact_damping,
        friction=args.friction,
        radius=args.radius,
        stretch_stiffness=args.stretch_stiffness,
        stretch_damping=args.stretch_damping,
        bend_stiffness=args.bend_stiffness,
        bend_damping=args.bend_damping,
        closed=False,
        pin="none",
        drive_end="none",
        drive_amplitude=0.0,
        drive_waypoints=2,
        drive_seed=0,
        ground=True,
        contacts=True,
        iterations=args.iterations,
        contact_alpha=args.contact_alpha,
        steps=args.steps,
        dt=args.dt,
        save_every=max(1, args.save_every),
        contact_interval=args.contact_interval,
    )


def make_pad_grid(thread_newton, tool_newton, center, x_axis, y_axis, normal, plane_signed, nx, ny, margin, sag):
    pts = np.vstack([thread_newton[:, :2], tool_newton[:, :2]]) if len(tool_newton) else thread_newton[:, :2]
    x0 = float(np.min(pts[:, 0]) - margin)
    x1 = float(np.max(pts[:, 0]) + margin)
    y0 = float(np.min(pts[:, 1]) - margin)
    y1 = float(np.max(pts[:, 1]) + margin)
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    # A visual deformable pad: sag near the thread footprint. Newton contact is
    # still the fitted plane z=0; this grid is for Genesis rendering.
    center_x = float(np.mean(thread_newton[:, 0]))
    center_y = float(np.mean(thread_newton[:, 1]))
    scale_x = max(0.5 * (x1 - x0), 1.0e-6)
    scale_y = max(0.5 * (y1 - y0), 1.0e-6)
    depression = sag * np.exp(-(((xx - center_x) / scale_x) ** 2 + ((yy - center_y) / scale_y) ** 2) * 2.5)
    pad_newton = np.stack([xx, yy, -depression], axis=-1).reshape(-1, 3)
    verts = newton_to_camera_plane(pad_newton, center, x_axis, y_axis, normal, plane_signed)

    faces = []
    for iz in range(ny - 1):
        for ix in range(nx - 1):
            a = iz * nx + ix
            b = a + 1
            c = a + nx + 1
            d = a + nx
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts, np.asarray(faces, dtype=np.int32)


def tool_points_from_scene(scene):
    pts = []
    if "tool_proxy_segments" in scene.files:
        pts.append(np.asarray(scene["tool_proxy_segments"], dtype=float).reshape(-1, 3))
    if "tool_branch_pts_3d" in scene.files and len(scene["tool_branch_pts_3d"]):
        pts.append(np.asarray(scene["tool_branch_pts_3d"], dtype=float).reshape(-1, 3))
    if "tool_position" in scene.files:
        pts.append(np.asarray(scene["tool_position"], dtype=float).reshape(1, 3))
    if not pts:
        return np.empty((0, 3), dtype=float)
    out = np.vstack(pts)
    return out[np.all(np.isfinite(out), axis=1)]


def fallback_drop_states(initial, num_states, radius):
    initial = np.asarray(initial, dtype=float)
    floor_z = max(float(radius), 1.0e-4)
    drop = max(float(initial[:, 2].min() - floor_z), 0.0)
    states = []
    for i in range(num_states):
        u = i / max(num_states - 1, 1)
        smooth = u * u * (3.0 - 2.0 * u)
        state = initial.copy()
        state[:, 2] = np.maximum(initial[:, 2] - smooth * drop, floor_z)
        states.append(state)
    return np.asarray(states, dtype=float)


def write_drop_preview(out_dir, scene, states_cam, fx, fy, cx, cy, image_path, width, height, radius):
    if image_path:
        image = load_rgb_image(image_path, width, height)
        canvas = (0.78 * image + 0.22 * make_synthetic_pad(width, height)).astype(np.uint8)
    else:
        canvas = make_synthetic_pad(width, height)

    draw_polyline_camera(
        canvas,
        states_cam[0],
        fx,
        fy,
        cx,
        cy,
        color=(120, 190, 255),
        radius_m=radius,
        min_px=2,
        max_px=10,
        shadow=False,
    )
    draw_polyline_camera(
        canvas,
        states_cam[-1],
        fx,
        fy,
        cx,
        cy,
        color=(238, 231, 219),
        radius_m=radius,
        min_px=3,
        max_px=12,
        shadow=True,
    )
    cv2.imwrite(str(out_dir / "gravity_drop_final_camera.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-npz", required=True)
    parser.add_argument("--out-dir", default="newton_gravity_drop_000000")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=1.0 / 1200.0)
    parser.add_argument("--save-every", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--stretch-stiffness", type=float, default=1.0e6)
    parser.add_argument("--stretch-damping", type=float, default=1.0e3)
    parser.add_argument("--bend-stiffness", type=float, default=1.0e4)
    parser.add_argument("--bend-damping", type=float, default=1.0e2)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--gravity", type=parse_vec3, default=(0.0, 0.0, -9.81))
    parser.add_argument("--contact-interval", type=int, default=1)
    parser.add_argument("--contact-alpha", type=float, default=0.0)
    parser.add_argument("--contact-stiffness", type=float, default=3.0e4)
    parser.add_argument("--contact-damping", type=float, default=250.0)
    parser.add_argument("--friction", type=float, default=0.9)
    parser.add_argument("--drop-height", type=float, default=0.018)
    parser.add_argument(
        "--pad-y-margin",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--preferred-normal", type=parse_vec3, default=(0.0, 0.0, -1.0))
    parser.add_argument("--flip-normal", action="store_true")
    parser.add_argument("--pad-grid-x", type=int, default=32)
    parser.add_argument("--pad-grid-y", type=int, default=32)
    parser.add_argument("--pad-margin", type=float, default=0.035)
    parser.add_argument("--pad-sag", type=float, default=0.004)
    parser.add_argument("--fallback-if-static", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--static-threshold", type=float, default=1.0e-4)
    parser.add_argument("--image", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=1025.8822021484375)
    parser.add_argument("--fy", type=float, default=1025.8822021484375)
    parser.add_argument("--cx", type=float, default=167.919017)
    parser.add_argument("--cy", type=float, default=234.152707)
    args = parser.parse_args()
    if args.pad_y_margin is not None:
        args.drop_height = args.pad_y_margin

    scene = np.load(args.scene_npz)
    thread_cam = np.asarray(scene["states"][-1], dtype=float)
    tool_points = tool_points_from_scene(scene)
    radius = args.radius
    if radius is None:
        radius = float(np.asarray(scene["radius"]).reshape(-1)[0]) if "radius" in scene.files else 0.001

    center, x_axis, y_axis, normal = fit_surface_frame(thread_cam, args.preferred_normal, args.flip_normal)
    signed = (thread_cam - center) @ normal
    plane_signed = float(np.min(signed) - args.drop_height)
    thread_newton = camera_to_newton_plane(thread_cam, center, x_axis, y_axis, normal, plane_signed)
    tool_newton = (
        camera_to_newton_plane(tool_points, center, x_axis, y_axis, normal, plane_signed)
        if len(tool_points)
        else np.empty((0, 3), dtype=float)
    )
    sim_args_dict = vars(args).copy()
    sim_args_dict["radius"] = radius
    sim = simulate_newton(thread_newton, make_sim_args(SimpleNamespace(**sim_args_dict)))
    displacement = float(np.linalg.norm(sim["states"][-1].mean(axis=0) - sim["states"][0].mean(axis=0)))
    fallback_used = bool(args.fallback_if_static and displacement < args.static_threshold)
    sim_states = sim["states"]
    if fallback_used:
        print(
            "Newton motion below threshold; using deterministic drop-to-plane "
            "fallback for Genesis render verification."
        )
        sim_states = fallback_drop_states(thread_newton, len(sim["states"]), radius)

    states_cam = newton_to_camera_plane(sim_states, center, x_axis, y_axis, normal, plane_signed)
    initial_cam = newton_to_camera_plane(sim["initial_centerline"], center, x_axis, y_axis, normal, plane_signed)
    pad_vertices, pad_faces = make_pad_grid(
        thread_newton=sim_states[-1],
        tool_newton=tool_newton,
        center=center,
        x_axis=x_axis,
        y_axis=y_axis,
        normal=normal,
        plane_signed=plane_signed,
        nx=args.pad_grid_x,
        ny=args.pad_grid_y,
        margin=args.pad_margin,
        sag=args.pad_sag,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    replaced_keys = {
        "times",
        "states",
        "initial_centerline",
        "newton_states",
        "camera_to_newton_pad_y",
        "camera_to_newton_center",
        "camera_to_newton_x_axis",
        "camera_to_newton_y_axis",
        "camera_to_newton_normal",
        "camera_to_newton_plane_signed",
        "pad_grid_vertices",
        "pad_grid_faces",
        "pad_corners",
        "radius",
    }
    payload = {key: scene[key] for key in scene.files if key not in replaced_keys}
    np.savez(
        out_dir / "newton_gravity_drop_scene.npz",
        **payload,
        times=sim["times"],
        states=states_cam,
        initial_centerline=initial_cam,
        newton_states=sim_states,
        raw_newton_states=sim["states"],
        fallback_drop_used=fallback_used,
        camera_to_newton_center=center,
        camera_to_newton_x_axis=x_axis,
        camera_to_newton_y_axis=y_axis,
        camera_to_newton_normal=normal,
        camera_to_newton_plane_signed=plane_signed,
        pad_grid_vertices=pad_vertices,
        pad_grid_faces=pad_faces,
        pad_corners=np.asarray(
            [
                pad_vertices[0],
                pad_vertices[args.pad_grid_x - 1],
                pad_vertices[-1],
                pad_vertices[-args.pad_grid_x],
            ],
            dtype=float,
        ),
        radius=radius,
    )
    write_drop_preview(out_dir, scene, states_cam, args.fx, args.fy, args.cx, args.cy, args.image, args.width, args.height, radius)
    print(f"saved gravity drop scene to {out_dir / 'newton_gravity_drop_scene.npz'}")
    print(f"preview: {out_dir / 'gravity_drop_final_camera.png'}")
    print(f"saved states: {len(states_cam)}")
    print(f"surface normal camera: {normal}")
    print(f"plane signed offset: {plane_signed:.6f}")
    print(f"min/max initial Newton height: {thread_newton[:, 2].min():.6f}, {thread_newton[:, 2].max():.6f}")
    print(f"min/max raw final Newton height: {sim['states'][-1, :, 2].min():.6f}, {sim['states'][-1, :, 2].max():.6f}")
    print(f"min/max saved final height: {sim_states[-1, :, 2].min():.6f}, {sim_states[-1, :, 2].max():.6f}")
    print(
        "Raw Newton center displacement: "
        f"{displacement:.6f} m"
    )
    print(f"fallback_drop_used: {fallback_used}")


if __name__ == "__main__":
    main()
