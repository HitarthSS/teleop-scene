#!/usr/bin/env python3
"""Live Newton ViewerGL client for the teleop UDP state stream."""

from __future__ import annotations

import argparse
import json
import math
import socket
import time

import numpy as np

from export_frame_scene_obj import tube_mesh


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
    args = parser.parse_args()

    import newton
    import newton.viewer
    import warp as wp

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", int(args.local_port)))
    sock.setblocking(False)
    server = (args.server_host, int(args.server_port))

    builder = newton.ModelBuilder()
    model = builder.finalize()
    state = model.state()
    viewer = newton.viewer.ViewerGL(width=int(args.width), height=int(args.height), headless=False)
    viewer.set_model(model)

    latest = None
    last_sub = 0.0
    last_draw = 0.0
    camera_set = False

    print(f"Newton ViewerGL subscribing to udp://{args.server_host}:{args.server_port}")
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
        all_points = np.vstack([nodes, target.reshape(1, 3), jaw.reshape(1, 3)])
        if not camera_set or not args.fixed_camera:
            pos, pitch, yaw = compute_camera(all_points, args.distance_scale)
            viewer.set_camera(pos=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), pitch=float(pitch), yaw=float(yaw))
            camera_set = True

        thread_v, thread_f = tube_mesh(nodes, args.thread_radius, args.thread_sides)
        target_v, target_f = make_sphere_mesh(target, args.point_radius)
        jaw_v, jaw_f = make_sphere_mesh(jaw, args.point_radius)
        start_v, start_f = make_sphere_mesh(nodes[0], args.point_radius * 0.8)
        end_v, end_f = make_sphere_mesh(nodes[-1], args.point_radius * 0.8)

        viewer.begin_frame(float(latest.get("time", now)))
        viewer.log_state(state)
        log_mesh(viewer, wp, "/teleop/thread", thread_v, thread_f, color=(0.86, 0.82, 0.72), roughness=0.55)
        log_mesh(viewer, wp, "/teleop/target", target_v, target_f, color=(1.0, 0.0, 1.0), roughness=0.3)
        log_mesh(viewer, wp, "/teleop/jaw_grasp", jaw_v, jaw_f, color=(0.02, 0.02, 0.02), roughness=0.3)
        log_mesh(viewer, wp, "/teleop/start", start_v, start_f, color=(0.0, 0.9, 1.0), roughness=0.3)
        log_mesh(viewer, wp, "/teleop/end", end_v, end_f, color=(1.0, 0.85, 0.0), roughness=0.3)
        viewer.end_frame()
        last_draw = now


if __name__ == "__main__":
    main()
