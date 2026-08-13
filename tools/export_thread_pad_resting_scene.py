#!/usr/bin/env python3
"""Export frame-0 thread/tool with a deformable-pad rest configuration.

This is a static, physics-ready setup step. It does not drop the thread. It
places the reconstructed thread centerline exactly one thread radius above a
fitted pad plane, preserving the in-plane thread shape, then exports:

  - OBJ/USD scene in Newton z-up viewer coordinates
  - NPZ scene bundle with camera-frame thread and pad grid
  - text diagnostics proving the thread rests on the pad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from export_frame_scene_obj import tube_mesh, write_obj
from render_frame_scene_with_psm_urdf import (
    choose_ee_link,
    forward_kinematics,
    joint_values_by_name,
    load_mesh_instances,
    parse_urdf,
    scene_thread_radius,
    transform_points,
)
from render_thread_robot_newton_gl import cam_to_newton_view, visible_bbox
from export_thread_robot_newton_scene import write_newton_usd


def normalize(v, fallback):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1.0e-12:
        return np.asarray(fallback, dtype=np.float64)
    return v / n


def fit_pad_frame(points):
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(points - center, full_matrices=False)
    x_axis = normalize(vh[0], (1.0, 0.0, 0.0))
    normal = normalize(vh[-1], (0.0, 0.0, 1.0))
    # In this dataset the pad normal should point roughly forward in OpenCV
    # camera coordinates, matching the side where the thread lives.
    if normal[2] < 0.0:
        normal = -normal
    y_axis = normalize(np.cross(normal, x_axis), (0.0, 1.0, 0.0))
    x_axis = normalize(np.cross(y_axis, normal), (1.0, 0.0, 0.0))
    return center, x_axis, y_axis, normal


def local_coordinates(points, center, x_axis, y_axis, normal):
    rel = np.asarray(points, dtype=np.float64) - center.reshape(1, 3)
    return np.column_stack([rel @ x_axis, rel @ y_axis, rel @ normal])


def camera_from_local(local, center, x_axis, y_axis, normal):
    local = np.asarray(local, dtype=np.float64)
    return (
        center.reshape(1, 3)
        + local[:, 0:1] * x_axis.reshape(1, 3)
        + local[:, 1:2] * y_axis.reshape(1, 3)
        + local[:, 2:3] * normal.reshape(1, 3)
    )


def make_pad_grid(center, x_axis, y_axis, normal, thread_local, radius, margin, min_half, nx, ny, indentation):
    low = thread_local[:, :2].min(axis=0) - margin
    high = thread_local[:, :2].max(axis=0) + margin
    half = np.maximum(0.5 * (high - low), min_half)
    mid = 0.5 * (low + high)
    low = mid - half
    high = mid + half

    xs = np.linspace(low[0], high[0], nx)
    ys = np.linspace(low[1], high[1], ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = np.zeros_like(xx)

    if indentation > 0.0:
        # Optional visual pre-deformation: shallow, smooth depression under the
        # thread. Keep off by default so "resting" means a flat initial pad.
        pts2 = thread_local[:, :2]
        grid2 = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
        d2 = np.full(len(grid2), np.inf, dtype=np.float64)
        for p in pts2:
            d2 = np.minimum(d2, np.sum((grid2 - p.reshape(1, 2)) ** 2, axis=1))
        sigma = max(4.0 * radius, 1.0e-4)
        zz.reshape(-1)[:] -= indentation * np.exp(-0.5 * d2 / (sigma * sigma))

    local = np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)])
    verts = camera_from_local(local, center, x_axis, y_axis, normal)
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = a + 1
            c = a + nx + 1
            d = a + nx
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts, local, np.asarray(faces, dtype=np.int32), low, high


def load_tool_meshes(args, scene):
    links, joints = parse_urdf(args.urdf, args.package_root)
    q = np.load(args.joints)
    jaw = np.load(args.jaw) if args.jaw else None
    values = joint_values_by_name(joints, q, jaw)
    fk = forward_kinematics(links, joints, values)
    ee_link = choose_ee_link(links, args.ee_link)
    if ee_link not in fk:
        raise ValueError(f"EE link {ee_link!r} was not reachable in URDF FK")

    cam_to_ee = np.asarray(scene["tool_cam_to_ee"], dtype=np.float64)
    cam_to_base = cam_to_ee @ np.linalg.inv(fk[ee_link])
    instances = load_mesh_instances(
        links,
        fk,
        cam_to_base,
        include_link=args.include_link_regex,
        exclude_link=args.exclude_link_regex,
        include_mesh=args.include_mesh_regex,
        exclude_mesh=args.exclude_mesh_regex,
    )
    meshes = []
    for i, (verts, faces, transform, link, mesh_path) in enumerate(instances):
        meshes.append(
            {
                "name": f"/psm/mesh_{i:03d}_{link}",
                "label": f"{link}: {mesh_path}",
                "verts_cam": transform_points(verts, transform),
                "faces": np.asarray(faces, dtype=np.int32),
                "color": (0.34, 0.36, 0.37),
                "roughness": 0.22,
                "metallic": 0.75,
            }
        )
    return ee_link, instances, meshes


def write_newton_obj(path, meshes):
    obj_meshes = []
    for mesh in meshes:
        if mesh["label"] == "thread":
            material = "thread"
        elif mesh["label"] == "deformable_pad":
            material = "pad"
        else:
            material = "tool"
        obj_meshes.append(
            (
                mesh["name"].strip("/").replace("/", "_") or "mesh",
                material,
                mesh["verts_newton"]
                if "verts_newton" in mesh
                else cam_to_newton_view(mesh["verts_cam"]),
                mesh["faces"],
            )
        )
    write_obj(Path(path), obj_meshes)


def write_report(
    path,
    args,
    scene,
    thread_original,
    thread_resting,
    pad_vertices,
    pad_faces,
    basis,
    radius,
    ee_link,
    tool_instances,
    tool_meshes_local,
):
    center, x_axis, y_axis, normal = basis
    local_rest = local_coordinates(thread_resting, center, x_axis, y_axis, normal)
    local_pad = local_coordinates(pad_vertices, center, x_axis, y_axis, normal)
    top_z = float(np.median(local_pad[:, 2]))
    signed = local_rest[:, 2] - top_z
    side = float(getattr(args, "thread_side", 1.0))
    signed_surface_clearance = side * signed - radius
    abs_surface_clearance = np.abs(signed) - radius
    z = thread_resting[:, 2]
    px = args.fx * (2.0 * radius) / np.maximum(z, 1.0e-6)
    thread_low, thread_high, thread_valid = visible_bbox(thread_resting, args.fx, args.fy, args.cx, args.cy)
    original_delta = np.linalg.norm(thread_resting - thread_original, axis=1)

    tool_lines = []
    if tool_meshes_local:
        tool_local = np.vstack([np.asarray(mesh["verts_newton"], dtype=np.float64) for mesh in tool_meshes_local])
        inside = (
            (tool_local[:, 0] >= args._pad_low[0])
            & (tool_local[:, 0] <= args._pad_high[0])
            & (tool_local[:, 1] >= args._pad_low[1])
            & (tool_local[:, 1] <= args._pad_high[1])
        )
        tool_lines.extend(
            [
                f"tool_local_z_m_min_median_max: {tool_local[:, 2].min():.9g}, {np.median(tool_local[:, 2]):.9g}, {tool_local[:, 2].max():.9g}",
                f"tool_vertices_inside_pad_xy: {int(np.count_nonzero(inside))} / {len(tool_local)}",
            ]
        )
        if np.any(inside):
            tool_lines.append(
                "tool_inside_pad_xy_z_m_min_median_max: "
                f"{tool_local[inside, 2].min():.9g}, "
                f"{np.median(tool_local[inside, 2]):.9g}, "
                f"{tool_local[inside, 2].max():.9g}"
            )

    lines = [
        f"scene_npz: {args.scene_npz}",
        f"urdf: {args.urdf}",
        f"ee_link: {ee_link}",
        f"state_index: {args.state_index}",
        f"thread_nodes: {len(thread_resting)}",
        f"thread_radius_m: {radius:.9g}",
        f"thread_diameter_m: {2.0 * radius:.9g}",
        f"thread_projected_diameter_px_min_median_max: {px.min():.3f}, {np.median(px):.3f}, {px.max():.3f}",
        f"thread_bbox_px_low_high_valid: {thread_low}, {thread_high}, {thread_valid}",
        f"thread_resting_projection_delta_m_min_median_max: {original_delta.min():.9g}, {np.median(original_delta):.9g}, {original_delta.max():.9g}",
        f"pad_grid_vertices_faces: {len(pad_vertices)}, {len(pad_faces)}",
        f"pad_grid_shape: {args.pad_nx}, {args.pad_ny}",
        f"pad_normal_camera: {normal}",
        f"pad_x_axis_camera: {x_axis}",
        f"pad_y_axis_camera: {y_axis}",
        f"thread_side: {side:.0f}",
        f"thread_center_to_pad_signed_distance_m_min_median_max: {signed.min():.9g}, {np.median(signed):.9g}, {signed.max():.9g}",
        f"thread_surface_clearance_to_pad_m_min_median_max: {signed_surface_clearance.min():.9g}, {np.median(signed_surface_clearance):.9g}, {signed_surface_clearance.max():.9g}",
        f"thread_abs_surface_clearance_to_pad_m_min_median_max: {abs_surface_clearance.min():.9g}, {np.median(abs_surface_clearance):.9g}, {abs_surface_clearance.max():.9g}",
        f"tool_mesh_parts: {len(tool_instances)}",
        *tool_lines,
    ]
    if "tool_cam_to_ee" in scene.files:
        lines.append("tool_cam_to_ee:")
        for row in np.asarray(scene["tool_cam_to_ee"], dtype=np.float64):
            lines.append("  " + " ".join(f"{v:.9g}" for v in row))
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-npz", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--joints", required=True)
    parser.add_argument("--jaw", default=None)
    parser.add_argument("--package-root", default=None)
    parser.add_argument("--ee-link", default="PSM1_tool_wrist_sca_shaft_link")
    parser.add_argument("--include-link-regex", default=None)
    parser.add_argument("--exclude-link-regex", default=None)
    parser.add_argument("--include-mesh-regex", default="instruments/420006")
    parser.add_argument("--exclude-mesh-regex", default=None)
    parser.add_argument("--thread-radius-m", type=float, default=None)
    parser.add_argument("--thread-diameter-m", type=float, default=None)
    parser.add_argument("--thread-sides", type=int, default=32)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--pad-nx", type=int, default=31)
    parser.add_argument("--pad-ny", type=int, default=31)
    parser.add_argument("--pad-margin", type=float, default=0.025)
    parser.add_argument("--pad-min-half", type=float, default=0.055)
    parser.add_argument("--pad-indentation", type=float, default=0.0)
    parser.add_argument(
        "--thread-side",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="+1 puts the thread on +Z side of the pad; -1 flips it to -Z side",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fx", type=float, default=1025.8822021484375)
    parser.add_argument("--fy", type=float, default=1025.8822021484375)
    parser.add_argument("--cx", type=float, default=167.919017)
    parser.add_argument("--cy", type=float, default=234.152707)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = np.load(args.scene_npz)
    states = np.asarray(scene["states"], dtype=np.float64)
    state_index = min(max(args.state_index, 0), states.shape[0] - 1)
    thread_original = states[state_index]
    radius = scene_thread_radius(scene, args.thread_radius_m, args.thread_diameter_m)

    center, x_axis, y_axis, normal = fit_pad_frame(thread_original)
    thread_local = local_coordinates(thread_original, center, x_axis, y_axis, normal)
    thread_local[:, 2] = float(args.thread_side) * radius
    thread_resting = camera_from_local(thread_local, center, x_axis, y_axis, normal)
    pad_vertices, pad_vertices_local, pad_faces, pad_low, pad_high = make_pad_grid(
        center,
        x_axis,
        y_axis,
        normal,
        thread_local,
        radius,
        args.pad_margin,
        args.pad_min_half,
        args.pad_nx,
        args.pad_ny,
        args.pad_indentation,
    )
    args._pad_low = pad_low
    args._pad_high = pad_high
    thread_v, thread_f = tube_mesh(thread_resting, radius, args.thread_sides)
    thread_v_local, _thread_f_local = tube_mesh(thread_local, radius, args.thread_sides)
    ee_link, tool_instances, tool_meshes = load_tool_meshes(args, scene)
    tool_meshes_local = []
    for mesh in tool_meshes:
        local_verts = local_coordinates(mesh["verts_cam"], center, x_axis, y_axis, normal)
        mesh = dict(mesh)
        mesh["verts_newton"] = local_verts
        tool_meshes_local.append(mesh)

    meshes = [
        {
            "name": "/deformable_pad",
            "label": "deformable_pad",
            "verts_cam": pad_vertices,
            "verts_newton": pad_vertices_local,
            "faces": pad_faces,
            "color": (0.84, 0.43, 0.38),
            "roughness": 0.80,
            "metallic": 0.0,
        },
        {
            "name": "/thread",
            "label": "thread",
            "verts_cam": thread_v,
            "verts_newton": thread_v_local,
            "faces": thread_f,
            "color": (0.90, 0.86, 0.76),
            "roughness": 0.55,
            "metallic": 0.0,
        },
        *tool_meshes_local,
    ]

    obj_path = out_dir / "thread_on_pad_resting_newton.obj"
    usd_path = out_dir / "thread_on_pad_resting_newton.usd"
    npz_path = out_dir / "thread_on_pad_resting_scene.npz"
    report_path = out_dir / "thread_on_pad_resting_report.txt"

    write_newton_obj(obj_path, meshes)
    write_newton_usd(usd_path, meshes, args.fps, args.device)
    np.savez(
        npz_path,
        thread_original_camera=thread_original,
        thread_resting_camera=thread_resting,
        thread_radius=np.asarray(radius, dtype=np.float64),
        pad_grid_vertices_camera=pad_vertices,
        pad_grid_vertices_newton=pad_vertices_local,
        pad_grid_faces=pad_faces,
        thread_resting_newton=thread_local,
        pad_local_low=pad_low,
        pad_local_high=pad_high,
        pad_center_camera=center,
        pad_x_axis_camera=x_axis,
        pad_y_axis_camera=y_axis,
        pad_normal_camera=normal,
        tool_cam_to_ee=np.asarray(scene["tool_cam_to_ee"], dtype=np.float64)
        if "tool_cam_to_ee" in scene.files
        else np.eye(4, dtype=np.float64),
    )
    write_report(
        report_path,
        args,
        scene,
        thread_original,
        thread_resting,
        pad_vertices,
        pad_faces,
        (center, x_axis, y_axis, normal),
        radius,
        ee_link,
        tool_instances,
        tool_meshes_local,
    )

    print(f"OBJ: {obj_path}")
    print(f"USD: {usd_path}")
    print(f"NPZ: {npz_path}")
    print(f"report: {report_path}")
    print(f"thread_diameter_m: {2.0 * radius:.9g}")
    print(f"pad_grid_vertices_faces: {len(pad_vertices)}, {len(pad_faces)}")


if __name__ == "__main__":
    main()
