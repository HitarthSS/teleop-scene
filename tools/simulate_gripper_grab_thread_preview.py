#!/usr/bin/env python3
"""Short Newton cable preview of a gripper grabbing the reconstructed thread.

This is a controlled first interaction:
  - thread is the Newton rod/cable
  - default mode: a small cluster of rod bodies nearest the current gripper is
    kinematically driven after the jaws close
  - optional contact mode: dynamic thread plus kinematic sphere-chain jaw
    collision proxies that close around and drag the thread
  - gripper jaws are simplified proxy meshes, not full triangle-mesh collision
  - pad and full/partial PSM meshes are exported for context

The goal is to debug thread/tool interaction before enabling expensive full
mesh collisions.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np

from export_frame_scene_obj import tube_mesh, write_obj
from export_thread_pad_resting_scene import (
    local_coordinates,
    load_tool_meshes,
)
from export_thread_robot_newton_scene import make_empty_newton_model, make_usd_viewer


def normalize(v, fallback=(1.0, 0.0, 0.0)):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1.0e-12:
        return np.asarray(fallback, dtype=np.float64)
    return v / n


def smoothstep(x):
    x = np.clip(float(x), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def set_body_kinematic(builder, wp, body_idx):
    builder.body_mass[int(body_idx)] = 0.0
    builder.body_inv_mass[int(body_idx)] = 0.0
    builder.body_inertia[int(body_idx)] = wp.mat33(0.0)
    builder.body_inv_inertia[int(body_idx)] = wp.mat33(0.0)


def set_body_translation(state, body_idx, xyz):
    body_q = state.body_q.numpy()
    body_q[int(body_idx), :3] = np.asarray(xyz, dtype=body_q.dtype)
    state.body_q.assign(body_q)
    if getattr(state, "body_qd", None) is not None:
        body_qd = state.body_qd.numpy()
        body_qd[int(body_idx), :] = 0.0
        state.body_qd.assign(body_qd)


def set_body_translations(state, body_indices, xyz):
    body_q = state.body_q.numpy()
    body_q[np.asarray(body_indices, dtype=np.int32), :3] = np.asarray(xyz, dtype=body_q.dtype)
    state.body_q.assign(body_q)
    if getattr(state, "body_qd", None) is not None:
        body_qd = state.body_qd.numpy()
        body_qd[np.asarray(body_indices, dtype=np.int32), :] = 0.0
        state.body_qd.assign(body_qd)


def body_positions(state, body_indices):
    body_q = state.body_q.numpy()
    return np.asarray([body_q[int(i), :3] for i in body_indices], dtype=np.float64)


def load_rest_scene(path):
    data = np.load(path, allow_pickle=True)
    required = [
        "thread_resting_newton",
        "thread_radius",
        "pad_grid_vertices_newton",
        "pad_grid_faces",
        "pad_center_camera",
        "pad_x_axis_camera",
        "pad_y_axis_camera",
        "pad_normal_camera",
        "tool_cam_to_ee",
    ]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{path} missing required arrays: {missing}")
    return data


def tool_meshes_in_local(args, rest_scene):
    scene_like = {"tool_cam_to_ee": np.asarray(rest_scene["tool_cam_to_ee"], dtype=np.float64)}
    ee_link, instances, meshes = load_tool_meshes(args, scene_like)
    center = np.asarray(rest_scene["pad_center_camera"], dtype=np.float64)
    x_axis = np.asarray(rest_scene["pad_x_axis_camera"], dtype=np.float64)
    y_axis = np.asarray(rest_scene["pad_y_axis_camera"], dtype=np.float64)
    normal = np.asarray(rest_scene["pad_normal_camera"], dtype=np.float64)
    out = []
    for mesh in meshes:
        mesh = dict(mesh)
        mesh["verts_newton"] = local_coordinates(mesh["verts_cam"], center, x_axis, y_axis, normal)
        out.append(mesh)
    return ee_link, instances, out


def infer_gripper_axes(tool_meshes, thread):
    jaw_meshes = [m for m in tool_meshes if "ee_link" in m["label"] or "sca_ee" in m["label"]]
    if len(jaw_meshes) >= 2:
        centers = [np.asarray(m["verts_newton"], dtype=np.float64).mean(axis=0) for m in jaw_meshes[:2]]
        sep_axis = normalize(centers[0] - centers[1], (1.0, 0.0, 0.0))
        jaw_center = 0.5 * (centers[0] + centers[1])
    elif jaw_meshes:
        verts = np.asarray(jaw_meshes[0]["verts_newton"], dtype=np.float64)
        jaw_center = verts.mean(axis=0)
        _u, _s, vh = np.linalg.svd(verts - jaw_center, full_matrices=False)
        sep_axis = normalize(vh[-1], (1.0, 0.0, 0.0))
    elif tool_meshes:
        verts = np.vstack([np.asarray(m["verts_newton"], dtype=np.float64) for m in tool_meshes])
        jaw_center = verts.mean(axis=0)
        _u, _s, vh = np.linalg.svd(verts - jaw_center, full_matrices=False)
        sep_axis = normalize(vh[-1], (1.0, 0.0, 0.0))
    else:
        jaw_center = thread.mean(axis=0)
        sep_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    idx = int(np.argmin(np.linalg.norm(thread[:, :2] - jaw_center[:2].reshape(1, 2), axis=1)))
    if 0 < idx < len(thread) - 1:
        tangent = normalize(thread[idx + 1] - thread[idx - 1], (0.0, 1.0, 0.0))
    elif idx == 0:
        tangent = normalize(thread[1] - thread[0], (0.0, 1.0, 0.0))
    else:
        tangent = normalize(thread[-1] - thread[-2], (0.0, 1.0, 0.0))
    jaw_axis = normalize(tangent - sep_axis * np.dot(tangent, sep_axis), (0.0, 1.0, 0.0))
    if np.linalg.norm(jaw_axis) < 1.0e-8:
        jaw_axis = normalize(np.cross(sep_axis, [0.0, 0.0, 1.0]), (0.0, 1.0, 0.0))
    sep_axis = normalize(sep_axis - jaw_axis * np.dot(sep_axis, jaw_axis), (1.0, 0.0, 0.0))
    return jaw_center, idx, jaw_axis, sep_axis


def choose_grip_nodes(n, grab_idx, radius_nodes):
    lo = max(0, grab_idx - radius_nodes)
    hi = min(n, grab_idx + radius_nodes + 1)
    return np.arange(lo, hi, dtype=np.int32)


def build_drive_positions(thread, grip_nodes, grab_idx, drag_vector, frame_count, close_frames):
    base = np.asarray(thread[grip_nodes], dtype=np.float64)
    out = []
    for frame in range(frame_count):
        if frame < close_frames:
            f = 0.0
        else:
            denom = max(frame_count - 1 - close_frames, 1)
            f = smoothstep((frame - close_frames) / denom)
        out.append(base + f * drag_vector.reshape(1, 3))
    return np.asarray(out, dtype=np.float64)


def add_compatible_sphere(builder, wp, body, radius, cfg, label=None):
    signatures = [
        ("add_shape_sphere", {"body": body, "pos": wp.vec3(0.0, 0.0, 0.0), "rot": wp.quat_identity(), "radius": radius, "cfg": cfg, "label": label}),
        ("add_shape_sphere", {"body": body, "radius": radius, "cfg": cfg, "label": label}),
        ("add_sphere", {"body": body, "radius": radius}),
    ]
    last_error = None
    for name, kwargs in signatures:
        fn = getattr(builder, name, None)
        if fn is None:
            continue
        try:
            accepted = inspect.signature(fn).parameters
            filtered = {k: v for k, v in kwargs.items() if k in accepted}
            fn(**filtered)
            return
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not add Newton sphere shape for jaw proxy: {last_error}")


def jaw_sphere_centers(center, jaw_axis, sep_axis, separation, length, samples):
    offsets = np.linspace(-0.5 * length, 0.5 * length, int(samples))
    centers = []
    for sign in (1.0, -1.0):
        jaw_mid = center + sign * 0.5 * separation * sep_axis
        centers.append(jaw_mid.reshape(1, 3) + offsets.reshape(-1, 1) * jaw_axis.reshape(1, 3))
    return np.vstack(centers)


def scripted_jaw_center(base_thread, initial_jaw_center, grab_idx, drag_vector, frame_f, frame_count, close_frames, jaw_offset):
    if frame_f < close_frames:
        approach = 1.0 - smoothstep(frame_f / max(close_frames, 1))
        return base_thread[grab_idx] + approach * jaw_offset
    denom = max(frame_count - 1 - close_frames, 1)
    f = smoothstep((frame_f - close_frames) / denom)
    return initial_jaw_center + f * drag_vector


def simulate_thread_collision_jaws(thread, radius, grab_idx, jaw_center, jaw_axis, sep_axis, drag_vector, jaw_offset, args):
    import newton
    import warp as wp

    if args.device:
        wp.set_device(args.device)

    builder = newton.ModelBuilder(gravity=float(args.gravity))
    builder.default_shape_cfg.ke = args.contact_stiffness
    builder.default_shape_cfg.kd = args.contact_damping
    builder.default_shape_cfg.mu = args.friction

    wp_points = [wp.vec3(float(x), float(y), float(z)) for x, y, z in thread]
    quats = newton.utils.create_parallel_transport_cable_quaternions(wp_points)
    rod_kwargs = dict(
        positions=wp_points,
        quaternions=quats,
        radius=radius,
        stretch_stiffness=args.stretch_stiffness,
        stretch_damping=args.stretch_damping,
        bend_stiffness=args.bend_stiffness,
        bend_damping=args.bend_damping,
        closed=False,
        label="gripper_contact_reconstructed_thread",
    )
    if "body_frame_origin" in inspect.signature(builder.add_rod).parameters:
        rod_kwargs["body_frame_origin"] = "com"
    rod_bodies, rod_joints = builder.add_rod(**rod_kwargs)

    initial_centers = jaw_sphere_centers(
        jaw_center_for_frame(np.asarray([thread]), grab_idx, drag_vector, 0, args.close_frames, jaw_offset),
        jaw_axis,
        sep_axis,
        args.jaw_open_separation,
        args.jaw_length,
        args.jaw_collision_samples,
    )
    jaw_cfg = builder.default_shape_cfg
    jaw_bodies = []
    for i, center in enumerate(initial_centers):
        body = builder.add_body(
            xform=wp.transform(
                wp.vec3(float(center[0]), float(center[1]), float(center[2])),
                wp.quat_identity(),
            )
        )
        add_compatible_sphere(builder, wp, body, args.jaw_collision_radius, jaw_cfg, label=f"jaw_contact_sphere_{i:02d}")
        set_body_kinematic(builder, wp, int(body))
        jaw_bodies.append(int(body))

    builder.color()
    model = builder.finalize()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model, contact_matching="latest")
    contacts = collision_pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=args.iterations,
        rigid_body_contact_buffer_size=args.contact_buffer_size,
        rigid_contact_history=True,
    )

    saved = [body_positions(state_0, rod_bodies)]
    jaw_saved = [body_positions(state_0, jaw_bodies)]
    times = [0.0]
    total_steps = (args.frames - 1) * args.substeps
    for step in range(1, total_steps + 1):
        frame_f = step / float(args.substeps)
        if frame_f < args.close_frames:
            close = smoothstep(frame_f / max(args.close_frames - 1, 1))
        else:
            close = 1.0
        sep = (1.0 - close) * args.jaw_open_separation + close * args.jaw_closed_separation
        center = scripted_jaw_center(thread, jaw_center, grab_idx, drag_vector, frame_f, args.frames, args.close_frames, jaw_offset)
        centers = jaw_sphere_centers(center, jaw_axis, sep_axis, sep, args.jaw_length, args.jaw_collision_samples)
        set_body_translations(state_0, jaw_bodies, centers)
        set_body_translations(state_1, jaw_bodies, centers)

        state_0.clear_forces()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, args.dt)
        state_0, state_1 = state_1, state_0

        if step % args.substeps == 0:
            frame_idx = step // args.substeps
            saved.append(body_positions(state_0, rod_bodies))
            jaw_saved.append(body_positions(state_0, jaw_bodies))
            times.append(step * args.dt)

    return (
        np.asarray(saved, dtype=np.float64),
        np.asarray(jaw_saved, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
        np.asarray(rod_bodies, dtype=np.int32),
        np.asarray(rod_joints, dtype=np.int32),
        np.asarray(jaw_bodies, dtype=np.int32),
    )


def simulate_thread_drive(thread, radius, grip_nodes, drive_positions, args):
    import newton
    import warp as wp

    if args.device:
        wp.set_device(args.device)

    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_shape_cfg.ke = args.contact_stiffness
    builder.default_shape_cfg.kd = args.contact_damping
    builder.default_shape_cfg.mu = args.friction
    wp_points = [wp.vec3(float(x), float(y), float(z)) for x, y, z in thread]
    quats = newton.utils.create_parallel_transport_cable_quaternions(wp_points)
    rod_kwargs = dict(
        positions=wp_points,
        quaternions=quats,
        radius=radius,
        stretch_stiffness=args.stretch_stiffness,
        stretch_damping=args.stretch_damping,
        bend_stiffness=args.bend_stiffness,
        bend_damping=args.bend_damping,
        closed=False,
        label="grabbed_reconstructed_thread",
    )
    if "body_frame_origin" in inspect.signature(builder.add_rod).parameters:
        rod_kwargs["body_frame_origin"] = "com"
    rod_bodies, rod_joints = builder.add_rod(**rod_kwargs)
    for node in grip_nodes:
        set_body_kinematic(builder, wp, int(rod_bodies[int(node)]))

    builder.color()
    model = builder.finalize()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    solver = newton.solvers.SolverVBD(model, iterations=args.iterations)

    saved = []
    times = []
    total_steps = (len(drive_positions) - 1) * args.substeps
    for node_i, node in enumerate(grip_nodes):
        body = int(rod_bodies[int(node)])
        set_body_translation(state_0, body, drive_positions[0, node_i])
        set_body_translation(state_1, body, drive_positions[0, node_i])
    saved.append(body_positions(state_0, rod_bodies))
    times.append(0.0)

    for step in range(1, total_steps + 1):
        frame_f = step / float(args.substeps)
        lower = int(np.floor(frame_f))
        upper = min(lower + 1, len(drive_positions) - 1)
        alpha = frame_f - lower
        targets = (1.0 - alpha) * drive_positions[lower] + alpha * drive_positions[upper]
        state_0.clear_forces()
        for node_i, node in enumerate(grip_nodes):
            body = int(rod_bodies[int(node)])
            set_body_translation(state_0, body, targets[node_i])
            set_body_translation(state_1, body, targets[node_i])
        solver.step(state_0, state_1, control, None, args.dt)
        state_0, state_1 = state_1, state_0
        if step % args.substeps == 0:
            frame_idx = step // args.substeps
            for node_i, node in enumerate(grip_nodes):
                set_body_translation(state_0, int(rod_bodies[int(node)]), drive_positions[frame_idx, node_i])
            saved.append(body_positions(state_0, rod_bodies))
            times.append(step * args.dt)

    return np.asarray(saved, dtype=np.float64), np.asarray(times, dtype=np.float64), np.asarray(rod_bodies, dtype=np.int32), np.asarray(rod_joints, dtype=np.int32)


def jaw_proxy_meshes(center, jaw_axis, sep_axis, separation, length, jaw_radius, sides):
    meshes = []
    for sign, name in ((1.0, "jaw_a"), (-1.0, "jaw_b")):
        c = center + sign * 0.5 * separation * sep_axis
        p0 = c - 0.5 * length * jaw_axis
        p1 = c + 0.5 * length * jaw_axis
        verts, faces = tube_mesh(np.asarray([p0, p1], dtype=np.float64), jaw_radius, sides)
        meshes.append((name, verts, faces))
    return meshes


def jaw_center_for_frame(thread_states, grab_idx, drag_vector, frame, close_frames, jaw_offset):
    base = thread_states[0, grab_idx]
    if frame < close_frames:
        approach = 1.0 - smoothstep(frame / max(close_frames, 1))
        return base + approach * jaw_offset
    return thread_states[frame, grab_idx] + 0.25 * jaw_offset


def frame_meshes(thread_state, pad_vertices, pad_faces, tool_meshes, radius, args, jaw_center, jaw_axis, sep_axis, separation):
    thread_v, thread_f = tube_mesh(thread_state, radius, args.thread_sides)
    meshes = [
        ("deformable_pad", "pad", pad_vertices, pad_faces),
        ("newton_thread", "thread", thread_v, thread_f),
    ]
    for mesh in tool_meshes:
        meshes.append(
            (
                mesh["name"].strip("/").replace("/", "_") or "tool",
                "tool",
                np.asarray(mesh["verts_newton"], dtype=np.float64),
                np.asarray(mesh["faces"], dtype=np.int32),
            )
        )
    for name, verts, faces in jaw_proxy_meshes(
        jaw_center,
        jaw_axis,
        sep_axis,
        separation,
        args.jaw_length,
        args.jaw_radius,
        args.jaw_sides,
    ):
        meshes.append((f"gripper_proxy_{name}", "tool", verts, faces))
    return meshes


def write_obj_sequence(out_dir, thread_states, pad_vertices, pad_faces, tool_meshes, radius, args, jaw_axis, sep_axis, jaw_offset):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame, thread_state in enumerate(thread_states):
        if frame < args.close_frames:
            close = smoothstep(frame / max(args.close_frames - 1, 1))
        else:
            close = 1.0
        sep = (1.0 - close) * args.jaw_open_separation + close * args.jaw_closed_separation
        jaw_center = jaw_center_for_frame(thread_states, args.grab_idx, args.drag_vector, frame, args.close_frames, jaw_offset)
        meshes = frame_meshes(thread_state, pad_vertices, pad_faces, tool_meshes, radius, args, jaw_center, jaw_axis, sep_axis, sep)
        path = out_dir / f"frame_{frame:06d}.obj"
        write_obj(path, meshes)
        paths.append(path)
        print(f"[{frame}] {path}")
    return paths


def write_usd_sequence(path, thread_states, pad_vertices, pad_faces, tool_meshes, radius, args, jaw_axis, sep_axis, jaw_offset, device=None):
    import warp as wp

    model, state = make_empty_newton_model()
    viewer = make_usd_viewer(Path(path), args.fps)
    viewer.set_model(model)
    for frame, thread_state in enumerate(thread_states):
        t = frame / float(args.fps)
        if frame < args.close_frames:
            close = smoothstep(frame / max(args.close_frames - 1, 1))
        else:
            close = 1.0
        sep = (1.0 - close) * args.jaw_open_separation + close * args.jaw_closed_separation
        jaw_center = jaw_center_for_frame(thread_states, args.grab_idx, args.drag_vector, frame, args.close_frames, jaw_offset)
        meshes = frame_meshes(thread_state, pad_vertices, pad_faces, tool_meshes, radius, args, jaw_center, jaw_axis, sep_axis, sep)
        viewer.begin_frame(t)
        viewer.log_state(state)
        for name, material, verts, faces in meshes:
            points_wp = wp.array(np.asarray(verts, dtype=np.float32), dtype=wp.vec3, device=device)
            indices_wp = wp.array(np.asarray(faces, dtype=np.int32).reshape(-1), dtype=wp.int32, device=device)
            color = (0.90, 0.86, 0.76) if material == "thread" else (0.84, 0.43, 0.38) if material == "pad" else (0.30, 0.32, 0.34)
            try:
                viewer.log_mesh(f"/{name}", points_wp, indices_wp, color=color, backface_culling=False)
            except TypeError:
                viewer.log_mesh(f"/{name}", points_wp, indices_wp, backface_culling=False)
        viewer.end_frame()
    viewer.close()


def write_report(path, args, thread0, states, radius, grip_nodes, jaw_center, jaw_axis, sep_axis, drag_vector, tool_instances):
    disp = np.linalg.norm(states[-1] - states[0], axis=1)
    lines = [
        f"rest_scene_npz: {args.rest_scene_npz}",
        f"frames: {len(states)}",
        f"fps: {args.fps}",
        f"thread_nodes: {len(thread0)}",
        f"thread_radius_m: {radius:.9g}",
        f"thread_diameter_m: {2.0 * radius:.9g}",
        f"grab_idx: {args.grab_idx}",
        f"grip_nodes: {' '.join(str(int(i)) for i in grip_nodes)}",
        f"jaw_initial_center_newton: {jaw_center}",
        f"jaw_axis_newton: {jaw_axis}",
        f"jaw_separation_axis_newton: {sep_axis}",
        f"jaw_open_closed_separation_m: {args.jaw_open_separation:.9g}, {args.jaw_closed_separation:.9g}",
        f"drag_vector_m: {drag_vector}",
        f"point_displacement_m_min_median_max: {disp.min():.9g}, {np.median(disp):.9g}, {disp.max():.9g}",
        f"grabbed_node_displacement_m: {np.linalg.norm(states[-1, args.grab_idx] - states[0, args.grab_idx]):.9g}",
        f"tool_mesh_parts: {len(tool_instances)}",
        f"interaction_mode: {args.interaction_mode}",
        "note: full PSM triangle collision is not enabled; collision mode uses simplified kinematic jaw sphere chains.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rest-scene-npz", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--joints", required=True)
    parser.add_argument("--jaw", default=None)
    parser.add_argument("--package-root", default=None)
    parser.add_argument("--ee-link", default="PSM1_tool_wrist_sca_shaft_link")
    parser.add_argument("--include-link-regex", default=None)
    parser.add_argument("--exclude-link-regex", default=None)
    parser.add_argument("--include-mesh-regex", default="tool_wrist|tool_wrist_sca")
    parser.add_argument("--exclude-mesh-regex", default="tool_main_link")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", type=int, default=28)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--dt", type=float, default=1.0 / 1200.0)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--interaction-mode", choices=("drive", "collision-jaws"), default="drive")
    parser.add_argument("--grip-radius-nodes", type=int, default=1)
    parser.add_argument("--grab-idx", type=int, default=-1)
    parser.add_argument("--drag-distance", type=float, default=0.012)
    parser.add_argument("--lift-distance", type=float, default=0.006)
    parser.add_argument("--gravity", type=float, default=0.0)
    parser.add_argument("--close-frames", type=int, default=8)
    parser.add_argument("--thread-sides", type=int, default=32)
    parser.add_argument("--jaw-sides", type=int, default=16)
    parser.add_argument("--jaw-length", type=float, default=0.011)
    parser.add_argument("--jaw-radius", type=float, default=0.00065)
    parser.add_argument("--jaw-open-separation", type=float, default=0.006)
    parser.add_argument("--jaw-closed-separation", type=float, default=0.0012)
    parser.add_argument("--jaw-collision-radius", type=float, default=0.0010)
    parser.add_argument("--jaw-collision-samples", type=int, default=7)
    parser.add_argument("--stretch-stiffness", type=float, default=5.0e3)
    parser.add_argument("--stretch-damping", type=float, default=2.0e1)
    parser.add_argument("--bend-stiffness", type=float, default=1.0e0)
    parser.add_argument("--bend-damping", type=float, default=5.0e-2)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e5)
    parser.add_argument("--contact-damping", type=float, default=0.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--contact-buffer-size", type=int, default=4096)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rest = load_rest_scene(Path(args.rest_scene_npz))
    thread = np.asarray(rest["thread_resting_newton"], dtype=np.float64)
    radius = float(np.asarray(rest["thread_radius"]).reshape(()))
    pad_vertices = np.asarray(rest["pad_grid_vertices_newton"], dtype=np.float64)
    pad_faces = np.asarray(rest["pad_grid_faces"], dtype=np.int32)
    ee_link, tool_instances, tool_meshes = tool_meshes_in_local(args, rest)

    jaw_center, inferred_idx, jaw_axis, sep_axis = infer_gripper_axes(tool_meshes, thread)
    grab_idx = inferred_idx if args.grab_idx < 0 else int(np.clip(args.grab_idx, 0, len(thread) - 1))
    args.grab_idx = grab_idx
    grip_nodes = choose_grip_nodes(len(thread), grab_idx, args.grip_radius_nodes)
    side = -1.0 if np.median(thread[:, 2]) < 0.0 else 1.0
    from_thread_to_tool = jaw_center - thread[grab_idx]
    approach = normalize(from_thread_to_tool, (0.0, 1.0, 0.0))
    away_from_pad = np.asarray([0.0, 0.0, side], dtype=np.float64)
    drag_vector = args.drag_distance * approach + args.lift_distance * away_from_pad
    args.drag_vector = drag_vector
    jaw_offset = 0.012 * approach + 0.004 * away_from_pad

    drive_positions = build_drive_positions(
        thread,
        grip_nodes,
        grab_idx,
        drag_vector,
        args.frames,
        args.close_frames,
    )
    jaw_states = None
    jaw_bodies = np.asarray([], dtype=np.int32)
    if args.interaction_mode == "collision-jaws":
        states, jaw_states, times, rod_bodies, rod_joints, jaw_bodies = simulate_thread_collision_jaws(
            thread,
            radius,
            grab_idx,
            jaw_center,
            jaw_axis,
            sep_axis,
            drag_vector,
            jaw_offset,
            args,
        )
    else:
        states, times, rod_bodies, rod_joints = simulate_thread_drive(
            thread,
            radius,
            grip_nodes,
            drive_positions,
            args,
        )

    np.savez(
        out_dir / "thread_gripper_grab_sim.npz",
        times=times,
        states_newton=states,
        initial_thread_newton=thread,
        thread_radius=np.asarray(radius, dtype=np.float64),
        grip_nodes=grip_nodes,
        grab_idx=np.asarray(grab_idx, dtype=np.int32),
        drive_positions=drive_positions,
        drag_vector=drag_vector,
        jaw_axis=jaw_axis,
        jaw_separation_axis=sep_axis,
        jaw_initial_center=jaw_center,
        pad_grid_vertices_newton=pad_vertices,
        pad_grid_faces=pad_faces,
        rod_bodies=rod_bodies,
        rod_joints=rod_joints,
        jaw_states_newton=jaw_states if jaw_states is not None else np.asarray([], dtype=np.float64),
        jaw_bodies=jaw_bodies,
        interaction_mode=np.asarray(args.interaction_mode),
    )
    obj_dir = out_dir / "obj_frames"
    write_obj_sequence(obj_dir, states, pad_vertices, pad_faces, tool_meshes, radius, args, jaw_axis, sep_axis, jaw_offset)
    write_usd_sequence(
        out_dir / "thread_gripper_grab_preview.usd",
        states,
        pad_vertices,
        pad_faces,
        tool_meshes,
        radius,
        args,
        jaw_axis,
        sep_axis,
        jaw_offset,
        device=args.device,
    )
    write_report(
        out_dir / "thread_gripper_grab_report.txt",
        args,
        thread,
        states,
        radius,
        grip_nodes,
        jaw_center,
        jaw_axis,
        sep_axis,
        drag_vector,
        tool_instances,
    )

    print(f"USD: {out_dir / 'thread_gripper_grab_preview.usd'}")
    print(f"OBJ frames: {obj_dir}")
    print(f"NPZ: {out_dir / 'thread_gripper_grab_sim.npz'}")
    print(f"report: {out_dir / 'thread_gripper_grab_report.txt'}")
    print(f"grab_idx: {grab_idx}")
    print(f"grip_nodes: {grip_nodes.tolist()}")
    print(f"grabbed displacement: {np.linalg.norm(states[-1, grab_idx] - states[0, grab_idx]):.9g} m")


if __name__ == "__main__":
    main()
