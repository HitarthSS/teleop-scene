#!/usr/bin/env python3
"""Render the reconstructed thread and PSM tool with Newton ViewerGL.

This is intentionally a no-pad render. It is for checking whether the
reconstructed thread centerline, thread diameter, and PSM URDF pose agree
before adding deformable-background physics.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from export_frame_scene_obj import tube_mesh
from render_frame_scene_with_psm_urdf import (
    choose_ee_link,
    forward_kinematics,
    joint_values_by_name,
    load_mesh_instances,
    parse_urdf,
    project,
    scene_thread_radius,
    transform_points,
)


def cam_to_newton_view(points: np.ndarray) -> np.ndarray:
    """OpenCV camera coords -> z-up Newton viewer coords.

    OpenCV: x right, y down, z forward.
    Viewer: x right, y forward/depth, z up.
    """

    points = np.asarray(points, dtype=np.float64)
    return np.column_stack([points[:, 0], points[:, 2], -points[:, 1]])


def visible_bbox(points_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray, int]:
    points_cam = np.asarray(points_cam, dtype=np.float64)
    valid = np.isfinite(points_cam).all(axis=1) & (points_cam[:, 2] > 1.0e-6)
    if not np.any(valid):
        nan = np.asarray([np.nan, np.nan], dtype=np.float64)
        return nan, nan, 0
    uv = project(points_cam[valid], fx, fy, cx, cy)
    return uv.min(axis=0), uv.max(axis=0), int(np.count_nonzero(valid))


def load_scene_geometry(args):
    scene = np.load(args.scene_npz)
    states = np.asarray(scene["states"], dtype=np.float64)
    if states.ndim != 3 or states.shape[-1] != 3:
        raise ValueError(f"expected scene states with shape (F, N, 3), got {states.shape}")
    state_index = min(max(args.state_index, 0), states.shape[0] - 1)
    thread_cam = states[state_index]
    thread_radius = scene_thread_radius(scene, args.thread_radius_m, args.thread_diameter_m)
    thread_v_cam, thread_f = tube_mesh(thread_cam, thread_radius, args.thread_sides)

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
    tool_instances = load_mesh_instances(
        links,
        fk,
        cam_to_base,
        include_link=args.include_link_regex,
        exclude_link=args.exclude_link_regex,
        include_mesh=args.include_mesh_regex,
        exclude_mesh=args.exclude_mesh_regex,
    )

    meshes = [
        {
            "name": "/thread",
            "label": "thread",
            "verts_cam": thread_v_cam,
            "faces": np.asarray(thread_f, dtype=np.int32),
            "color": (0.90, 0.86, 0.76),
            "roughness": 0.55,
            "metallic": 0.0,
        }
    ]
    for i, (verts, faces, transform, link, mesh_path) in enumerate(tool_instances):
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
    return scene, thread_cam, thread_radius, ee_link, tool_instances, meshes


def compute_camera(meshes, view: str, distance_scale: float):
    all_v = np.vstack([cam_to_newton_view(mesh["verts_cam"]) for mesh in meshes if len(mesh["verts_cam"])])
    low = all_v.min(axis=0)
    high = all_v.max(axis=0)
    center = 0.5 * (low + high)
    extent = high - low
    radius = max(float(np.linalg.norm(extent)) * 0.5, 0.035)

    if view == "camera":
        pos = np.asarray([0.0, max(0.0, center[1] - radius * distance_scale), center[2]], dtype=np.float64)
        target = center
        yaw = 90.0
        pitch = math.degrees(math.atan2(target[2] - pos[2], max(target[1] - pos[1], 1.0e-6)))
    elif view == "bird":
        pos = center + np.asarray([0.0, -0.20, radius * distance_scale], dtype=np.float64)
        target = center
        yaw = 90.0
        pitch = -math.degrees(math.atan2(pos[2] - target[2], max(target[1] - pos[1], 1.0e-6)))
    else:
        pos = center + np.asarray([0.65, -1.0, 0.55], dtype=np.float64) * radius * distance_scale
        target = center
        horiz = target[:2] - pos[:2]
        yaw = math.degrees(math.atan2(horiz[1], horiz[0]))
        pitch = math.degrees(math.atan2(target[2] - pos[2], max(np.linalg.norm(horiz), 1.0e-6)))

    return pos, target, pitch, yaw, low, high


def write_report(path, args, scene, thread_cam, thread_radius, ee_link, tool_instances, meshes, camera_info):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    thread_diam = 2.0 * thread_radius
    z = thread_cam[:, 2]
    px = args.fx * thread_diam / np.maximum(z, 1.0e-6)
    thread_low, thread_high, thread_valid = visible_bbox(thread_cam, args.fx, args.fy, args.cx, args.cy)
    tool_points = np.vstack([m["verts_cam"] for m in meshes[1:] if len(m["verts_cam"])]) if len(meshes) > 1 else np.empty((0, 3))
    tool_low, tool_high, tool_valid = visible_bbox(tool_points, args.fx, args.fy, args.cx, args.cy)
    pos, target, pitch, yaw, low, high = camera_info

    lines = [
        f"scene_npz: {args.scene_npz}",
        f"urdf: {args.urdf}",
        f"ee_link: {ee_link}",
        f"state_index: {args.state_index}",
        f"thread_nodes: {len(thread_cam)}",
        f"thread_radius_m: {thread_radius:.9g}",
        f"thread_diameter_m: {thread_diam:.9g}",
        f"thread_projected_diameter_px_min_median_max: {px.min():.3f}, {np.median(px):.3f}, {px.max():.3f}",
        f"thread_depth_m_min_median_max: {z.min():.6f}, {np.median(z):.6f}, {z.max():.6f}",
        f"thread_bbox_px_low_high_valid: {thread_low}, {thread_high}, {thread_valid}",
        f"tool_mesh_parts: {len(tool_instances)}",
        f"tool_bbox_px_low_high_valid: {tool_low}, {tool_high}, {tool_valid}",
        f"newton_view_bounds_low: {low}",
        f"newton_view_bounds_high: {high}",
        f"newton_camera_pos: {pos}",
        f"newton_camera_target: {target}",
        f"newton_camera_pitch_yaw_deg: {pitch:.3f}, {yaw:.3f}",
        "tool_meshes:",
    ]
    for mesh in meshes[1:]:
        lines.append(f"  {mesh['label']}")
    if "tool_cam_to_ee" in scene.files:
        lines.append("tool_cam_to_ee:")
        for row in np.asarray(scene["tool_cam_to_ee"], dtype=np.float64):
            lines.append("  " + " ".join(f"{v:.9g}" for v in row))
    path.write_text("\n".join(lines) + "\n")


def save_newton_gl_render(meshes, out_path, camera_info, width, height, fov, device=None):
    try:
        import pyglet

        pyglet.options["headless"] = True
    except Exception:
        pass

    import newton
    import newton.viewer
    import warp as wp

    builder = newton.ModelBuilder()
    model = builder.finalize()
    state = model.state()

    viewer = newton.viewer.ViewerGL(width=int(width), height=int(height), headless=True)
    viewer.set_model(model)
    pos, _target, pitch, yaw, _low, _high = camera_info
    viewer.set_camera(pos=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), pitch=float(pitch), yaw=float(yaw))
    if hasattr(viewer, "camera") and hasattr(viewer.camera, "fov"):
        viewer.camera.fov = float(fov)

    viewer.begin_frame(0.0)
    viewer.log_state(state)
    for mesh in meshes:
        verts = cam_to_newton_view(mesh["verts_cam"]).astype(np.float32)
        faces = np.asarray(mesh["faces"], dtype=np.int32).reshape(-1)
        points_wp = wp.array(verts, dtype=wp.vec3, device=device)
        indices_wp = wp.array(faces, dtype=wp.int32, device=device)
        try:
            viewer.log_mesh(
                mesh["name"],
                points_wp,
                indices_wp,
                color=mesh["color"],
                roughness=mesh["roughness"],
                metallic=mesh["metallic"],
                backface_culling=False,
            )
        except TypeError:
            viewer.log_mesh(
                mesh["name"],
                points_wp,
                indices_wp,
                backface_culling=False,
            )
    viewer.end_frame()
    frame = viewer.get_frame(render_ui=False)
    rgb = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    viewer.close()


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
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--view", choices=("camera", "oblique", "bird"), default="camera")
    parser.add_argument("--distance-scale", type=float, default=2.2)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=38.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fx", type=float, default=1025.8822021484375)
    parser.add_argument("--fy", type=float, default=1025.8822021484375)
    parser.add_argument("--cx", type=float, default=167.919017)
    parser.add_argument("--cy", type=float, default=234.152707)
    args = parser.parse_args()

    scene, thread_cam, thread_radius, ee_link, tool_instances, meshes = load_scene_geometry(args)
    if not tool_instances:
        raise RuntimeError("URDF loaded no renderable PSM STL/OBJ meshes")
    camera_info = compute_camera(meshes, args.view, args.distance_scale)
    write_report(
        args.report,
        args,
        scene,
        thread_cam,
        thread_radius,
        ee_link,
        tool_instances,
        meshes,
        camera_info,
    )
    save_newton_gl_render(
        meshes,
        args.out,
        camera_info,
        width=args.width,
        height=args.height,
        fov=args.fov,
        device=args.device,
    )
    print(f"Newton GL render: {args.out}")
    print(f"report: {args.report}")
    print(f"tool mesh parts: {len(tool_instances)}")
    print(f"thread_diameter_m: {2.0 * thread_radius:.9g}")


if __name__ == "__main__":
    main()
