#!/usr/bin/env python3
"""Live Newton ViewerGL client for the teleop UDP state stream."""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
import types

import numpy as np

from export_frame_scene_obj import tube_mesh
from render_frame_scene_with_psm_urdf import (
    choose_ee_link,
    forward_kinematics,
    joint_values_by_name,
    load_mesh_instances,
    parse_urdf,
    transform_points,
)
from render_thread_robot_newton_gl import cam_to_newton_view


def load_robot_context(args):
    if not args.scene_npz or not args.urdf or not args.joints:
        return None
    scene = np.load(args.scene_npz, allow_pickle=True)
    links, joints = parse_urdf(args.urdf, args.package_root)
    q_recorded = np.load(args.joints)
    jaw_recorded = np.load(args.jaw) if args.jaw else None
    base_values = joint_values_by_name(joints, q_recorded, jaw_recorded)
    fk0 = forward_kinematics(links, joints, base_values)
    ee_link = choose_ee_link(links, args.ee_link)
    cam_to_ee = np.asarray(scene["tool_cam_to_ee"], dtype=np.float64)
    cam_to_base = cam_to_ee @ np.linalg.inv(fk0[ee_link])
    return {
        "links": links,
        "joints": joints,
        "base_values": base_values,
        "cam_to_base": cam_to_base,
    }


def robot_meshes_from_state(robot_context, joint_values, args):
    if robot_context is None:
        return []
    values = dict(robot_context["base_values"])
    for name, value in (joint_values or {}).items():
        if name in values:
            values[name] = float(value)
    fk = forward_kinematics(robot_context["links"], robot_context["joints"], values)
    meshes = []
    for i, (verts, faces, transform, link, _mesh_path) in enumerate(
        load_mesh_instances(
            robot_context["links"],
            fk,
            robot_context["cam_to_base"],
            include_link=args.include_link_regex,
            exclude_link=args.exclude_link_regex,
            include_mesh=args.include_mesh_regex,
            exclude_mesh=args.exclude_mesh_regex,
        )
    ):
        verts_cam = transform_points(verts, transform)
        meshes.append((f"/teleop/psm/{i:03d}_{link}", cam_to_newton_view(verts_cam), np.asarray(faces, dtype=np.int32)))
    return meshes


def make_sphere_mesh(center, radius=0.0012, rings=8, sectors=16):
    center = np.asarray(center, dtype=np.float64).reshape(3)
    verts = []
    faces = []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(sectors):
            theta = 2.0 * math.pi * s / sectors
            verts.append(
                center
                + radius
                * np.asarray(
                    [
                        math.sin(phi) * math.cos(theta),
                        math.sin(phi) * math.sin(theta),
                        math.cos(phi),
                    ],
                    dtype=np.float64,
                )
            )
    for r in range(rings):
        for s in range(sectors):
            a = r * sectors + s
            b = r * sectors + (s + 1) % sectors
            c = (r + 1) * sectors + (s + 1) % sectors
            d = (r + 1) * sectors + s
            faces.append([a, b, c])
            faces.append([a, c, d])
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def compute_camera(points, distance_scale=2.8):
    points = np.asarray(points, dtype=np.float64)
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = 0.5 * (low + high)
    extent = high - low
    radius = max(float(np.linalg.norm(extent)) * 0.5, 0.035)
    pos = center + np.asarray([0.55, -1.0, 0.55], dtype=np.float64) * radius * distance_scale
    target = center
    horiz = target[:2] - pos[:2]
    yaw = math.degrees(math.atan2(horiz[1], horiz[0]))
    pitch = math.degrees(math.atan2(target[2] - pos[2], max(np.linalg.norm(horiz), 1.0e-6)))
    return pos, pitch, yaw


def log_mesh(viewer, wp, name, verts, faces, color, roughness=0.55, metallic=0.0):
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


def configure_viewer_navigation(viewer, args):
    """Tune Newton ViewerGL controls for millimeter-scale scenes."""
    if hasattr(viewer, "camera_speed"):
        viewer.camera_speed = float(args.camera_speed)

    original_scroll = getattr(viewer, "on_mouse_scroll", None)
    if callable(original_scroll) and float(args.scroll_scale) != 1.0:
        scroll_scale = float(args.scroll_scale)

        def scaled_mouse_scroll(self, x, y, scroll_x, scroll_y):
            return original_scroll(x, y, scroll_x * scroll_scale, scroll_y * scroll_scale)

        viewer.on_mouse_scroll = types.MethodType(scaled_mouse_scroll, viewer)


def clamp_viewer_navigation(viewer, args):
    if bool(args.lock_camera_speed) and hasattr(viewer, "camera_speed"):
        viewer.camera_speed = float(args.camera_speed)
    elif hasattr(viewer, "camera_speed") and args.max_camera_speed is not None:
        viewer.camera_speed = min(float(viewer.camera_speed), float(args.max_camera_speed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8765)
    parser.add_argument("--local-port", type=int, default=0)
    parser.add_argument("--subscribe-hz", type=float, default=10.0)
    parser.add_argument("--draw-hz", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--thread-radius", type=float, default=0.0003)
    parser.add_argument("--thread-sides", type=int, default=16)
    parser.add_argument("--point-radius", type=float, default=0.0012)
    parser.add_argument("--distance-scale", type=float, default=2.8)
    parser.add_argument("--fixed-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-speed", type=float, default=0.01)
    parser.add_argument("--max-camera-speed", type=float, default=0.025)
    parser.add_argument("--lock-camera-speed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scroll-scale", type=float, default=0.15)
    parser.add_argument("--scene-npz", default=None)
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--joints", default=None)
    parser.add_argument("--jaw", default=None)
    parser.add_argument("--package-root", default=None)
    parser.add_argument("--ee-link", default="PSM1_tool_wrist_sca_shaft_link")
    parser.add_argument("--include-link-regex", default=None)
    parser.add_argument("--exclude-link-regex", default=None)
    parser.add_argument("--include-mesh-regex", default="tool_wrist|tool_wrist_sca")
    parser.add_argument("--exclude-mesh-regex", default="tool_main_link")
    args = parser.parse_args()

    import newton
    import newton.viewer
    import warp as wp

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", int(args.local_port)))
    sock.setblocking(False)
    server = (args.server_host, int(args.server_port))
    robot_context = load_robot_context(args)

    builder = newton.ModelBuilder()
    model = builder.finalize()
    state = model.state()
    viewer = newton.viewer.ViewerGL(width=int(args.width), height=int(args.height), headless=False)
    viewer.set_model(model)
    configure_viewer_navigation(viewer, args)

    latest = None
    last_sub = 0.0
    last_draw = 0.0
    camera_set = False

    print(f"Newton ViewerGL subscribing to udp://{args.server_host}:{args.server_port}")
    print(
        "Viewer navigation: "
        f"camera_speed={args.camera_speed} m/s, "
        f"lock_camera_speed={args.lock_camera_speed}, "
        f"scroll_scale={args.scroll_scale}"
    )
    print("Close the Newton viewer window or press Ctrl+C to stop.")

    while True:
        now = time.perf_counter()
        if now - last_sub > 1.0 / max(float(args.subscribe_hz), 1.0):
            sock.sendto(json.dumps({"type": "newton_viewer_subscribe", "subscribe": True}).encode("utf-8"), server)
            last_sub = now

        while True:
            try:
                data, _addr = sock.recvfrom(65507)
                latest = json.loads(data.decode("utf-8"))
            except BlockingIOError:
                break

        if latest is None or now - last_draw < 1.0 / max(float(args.draw_hz), 1.0):
            time.sleep(0.001)
            continue

        nodes = np.asarray(latest.get("thread_nodes_newton", []), dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 2:
            time.sleep(0.001)
            continue

        target = np.asarray(latest.get("target_newton", nodes[0]), dtype=np.float64)
        jaw = np.asarray(latest.get("jaw_grasp_newton", nodes[0]), dtype=np.float64)
        attach_driver = np.asarray(latest.get("attach_driver_newton", jaw), dtype=np.float64)
        robot_meshes = robot_meshes_from_state(robot_context, latest.get("joint_values", {}), args)
        robot_points = np.vstack([mesh[1] for mesh in robot_meshes]) if robot_meshes else np.empty((0, 3))
        all_points = np.vstack([nodes, target.reshape(1, 3), jaw.reshape(1, 3), attach_driver.reshape(1, 3), robot_points])
        if not camera_set or not args.fixed_camera:
            pos, pitch, yaw = compute_camera(all_points, args.distance_scale)
            viewer.set_camera(pos=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), pitch=float(pitch), yaw=float(yaw))
            clamp_viewer_navigation(viewer, args)
            camera_set = True

        thread_v, thread_f = tube_mesh(nodes, args.thread_radius, args.thread_sides)
        target_v, target_f = make_sphere_mesh(target, args.point_radius)
        jaw_v, jaw_f = make_sphere_mesh(jaw, args.point_radius)
        driver_v, driver_f = make_sphere_mesh(attach_driver, args.point_radius * 0.9)
        start_v, start_f = make_sphere_mesh(nodes[0], args.point_radius * 0.8)
        end_v, end_f = make_sphere_mesh(nodes[-1], args.point_radius * 0.8)

        viewer.begin_frame(float(latest.get("time", now)))
        viewer.log_state(state)
        log_mesh(viewer, wp, "/teleop/thread", thread_v, thread_f, color=(0.86, 0.82, 0.72), roughness=0.55)
        log_mesh(viewer, wp, "/teleop/target", target_v, target_f, color=(1.0, 0.0, 1.0), roughness=0.3)
        log_mesh(viewer, wp, "/teleop/jaw_grasp", jaw_v, jaw_f, color=(0.02, 0.02, 0.02), roughness=0.3)
        log_mesh(viewer, wp, "/teleop/attach_driver", driver_v, driver_f, color=(0.0, 1.0, 0.25), roughness=0.3)
        log_mesh(viewer, wp, "/teleop/start", start_v, start_f, color=(0.0, 0.9, 1.0), roughness=0.3)
        log_mesh(viewer, wp, "/teleop/end", end_v, end_f, color=(1.0, 0.85, 0.0), roughness=0.3)
        for name, verts, faces in robot_meshes:
            log_mesh(viewer, wp, name, verts, faces, color=(0.34, 0.36, 0.37), roughness=0.22, metallic=0.75)
        viewer.end_frame()
        clamp_viewer_navigation(viewer, args)
        last_draw = now


if __name__ == "__main__":
    main()
