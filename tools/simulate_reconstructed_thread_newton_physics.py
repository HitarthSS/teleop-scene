#!/usr/bin/env python3
"""Physics-only Newton simulation for one reconstructed thread frame.

This deliberately avoids Newton viewer plumbing. It follows the documented
SolverVBD loop and the local gravity sanity test:

    builder.color()
    model = builder.finalize()
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    for step:
        state_in.clear_forces()
        collide(...)
        solver.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in

The viewer/USD step can be added after this NPZ shows real motion.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np


def normalize(v, fallback):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.asarray(fallback, dtype=np.float64)
    return v / n


def resample_arclength(points, n):
    points = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1.0e-10])
    points = points[keep]
    if len(points) < 3:
        raise ValueError("need at least three distinct thread points")
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, s[-1], n)
    out = np.empty((n, 3), dtype=np.float64)
    for dim in range(3):
        out[:, dim] = np.interp(target, s, points[:, dim])
    return out


def fit_plane(points):
    center = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    x_axis = normalize(vh[0], (1.0, 0.0, 0.0))
    normal = normalize(vh[-1], (0.0, 0.0, 1.0))
    if normal[2] < 0.0:
        normal = -normal
    y_axis = normalize(np.cross(normal, x_axis), (0.0, 1.0, 0.0))
    x_axis = normalize(np.cross(y_axis, normal), (1.0, 0.0, 0.0))
    return center, x_axis, y_axis, normal


def load_scene_centerline(scene_npz, num_nodes):
    data = np.load(scene_npz, allow_pickle=True)
    if "initial_centerline" in data:
        points = np.asarray(data["initial_centerline"], dtype=np.float64)
    elif "states" in data:
        points = np.asarray(data["states"][0], dtype=np.float64)
    else:
        raise ValueError(f"{scene_npz} does not contain initial_centerline or states")
    return resample_arclength(points, num_nodes), data


def to_local_drop_frame(points_cam, scene, radius, drop_height, use_scene_plane):
    if use_scene_plane and all(
        key in scene
        for key in (
            "camera_to_newton_center",
            "camera_to_newton_x_axis",
            "camera_to_newton_y_axis",
            "camera_to_newton_normal",
        )
    ):
        center = np.asarray(scene["camera_to_newton_center"], dtype=np.float64)
        x_axis = normalize(scene["camera_to_newton_x_axis"], (1.0, 0.0, 0.0))
        y_axis = normalize(scene["camera_to_newton_y_axis"], (0.0, 1.0, 0.0))
        normal = normalize(scene["camera_to_newton_normal"], (0.0, 0.0, 1.0))
    else:
        center, x_axis, y_axis, normal = fit_plane(points_cam)
    rel = points_cam - center
    raw_local = np.column_stack((rel @ x_axis, rel @ y_axis, rel @ normal))
    xy_center = np.mean(raw_local[:, :2], axis=0)
    z_min = float(np.min(raw_local[:, 2]))
    local = raw_local.copy()
    local[:, :2] -= xy_center
    local[:, 2] -= z_min
    local[:, 2] += radius + drop_height
    basis = {
        "camera_center": center,
        "camera_x_axis": x_axis,
        "camera_y_axis": y_axis,
        "camera_normal": normal,
        "local_xy_center": xy_center,
        "local_z_min": np.asarray(z_min, dtype=np.float64),
        "local_z_lift": np.asarray(radius + drop_height, dtype=np.float64),
    }
    return local, basis


def body_positions(state, body_indices):
    body_q = state.body_q.numpy()
    return np.asarray([body_q[int(i), :3] for i in body_indices], dtype=np.float64)


def body_velocities(state, body_indices):
    body_qd = state.body_qd.numpy()
    return np.asarray([body_qd[int(i), :3] for i in body_indices], dtype=np.float64)


def local_points_to_camera(points_local, basis):
    points_local = np.asarray(points_local, dtype=np.float64)
    center = np.asarray(basis["camera_center"], dtype=np.float64)
    x_axis = np.asarray(basis["camera_x_axis"], dtype=np.float64)
    y_axis = np.asarray(basis["camera_y_axis"], dtype=np.float64)
    normal = np.asarray(basis["camera_normal"], dtype=np.float64)
    xy_center = np.asarray(basis.get("local_xy_center", np.zeros(2)), dtype=np.float64)
    z_min = float(np.asarray(basis.get("local_z_min", 0.0)).reshape(()))
    z_lift = float(np.asarray(basis.get("local_z_lift", 0.0)).reshape(()))
    raw = points_local.copy()
    raw[..., 0] += xy_center[0]
    raw[..., 1] += xy_center[1]
    raw[..., 2] += z_min - z_lift
    return (
        center
        + raw[..., 0, None] * x_axis
        + raw[..., 1, None] * y_axis
        + raw[..., 2, None] * normal
    )


def local_vectors_to_camera(vectors_local, basis):
    vectors_local = np.asarray(vectors_local, dtype=np.float64)
    x_axis = np.asarray(basis["camera_x_axis"], dtype=np.float64)
    y_axis = np.asarray(basis["camera_y_axis"], dtype=np.float64)
    normal = np.asarray(basis["camera_normal"], dtype=np.float64)
    return (
        vectors_local[..., 0, None] * x_axis
        + vectors_local[..., 1, None] * y_axis
        + vectors_local[..., 2, None] * normal
    )


def make_ground_pad_from_newton_states(newton_states, basis, radius, margin):
    states = np.asarray(newton_states, dtype=np.float64)
    low = states[..., :2].min(axis=(0, 1)) - margin
    high = states[..., :2].max(axis=(0, 1)) + margin
    z = 0.0
    pad_local = np.asarray(
        [
            [low[0], low[1], z],
            [high[0], low[1], z],
            [high[0], high[1], z],
            [low[0], high[1], z],
        ],
        dtype=np.float64,
    )
    min_half = max(0.025, 8.0 * radius)
    center_xy = 0.5 * (low + high)
    half = np.maximum(0.5 * (high - low), min_half)
    pad_local[:, 0] = np.asarray([center_xy[0] - half[0], center_xy[0] + half[0], center_xy[0] + half[0], center_xy[0] - half[0]])
    pad_local[:, 1] = np.asarray([center_xy[1] - half[1], center_xy[1] - half[1], center_xy[1] + half[1], center_xy[1] + half[1]])
    return local_points_to_camera(pad_local, basis)


def make_cloth_pad_grid(points_local, radius, margin, dim_x, dim_y):
    points = np.asarray(points_local, dtype=np.float64)
    low = points[:, :2].min(axis=0) - margin
    high = points[:, :2].max(axis=0) + margin
    min_half = max(0.025, 8.0 * radius)
    center_xy = 0.5 * (low + high)
    half = np.maximum(0.5 * (high - low), min_half)
    low = center_xy - half
    high = center_xy + half
    size = high - low
    return low, float(size[0] / max(dim_x, 1)), float(size[1] / max(dim_y, 1))


def infer_cloth_grid_shape(particle_count, dim_x, dim_y):
    expected_nodes = (dim_x + 1) * (dim_y + 1)
    if particle_count == expected_nodes:
        return dim_x + 1, dim_y + 1
    expected_cells = dim_x * dim_y
    if particle_count == expected_cells:
        return dim_x, dim_y
    side = int(round(np.sqrt(particle_count)))
    if side * side == particle_count:
        return side, side
    return particle_count, 1


def make_grid_faces(nx, ny):
    faces = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            a = y * nx + x
            b = a + 1
            c = a + nx
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return np.asarray(faces, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-npz", required=True)
    parser.add_argument("--out-dir", default="newton_reconstructed_thread_physics")
    parser.add_argument("--num-nodes", type=int, default=64)
    parser.add_argument("--radius", type=float, default=-1.0)
    parser.add_argument("--drop-height", type=float, default=0.035)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=1.0 / 1200.0)
    parser.add_argument("--save-every", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--use-scene-plane", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ground", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--closed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stretch-stiffness", type=float, default=1.0e3)
    parser.add_argument("--stretch-damping", type=float, default=1.0e1)
    parser.add_argument("--bend-stiffness", type=float, default=1.0e-1)
    parser.add_argument("--bend-damping", type=float, default=1.0e-2)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e5)
    parser.add_argument("--contact-damping", type=float, default=0.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--contact-buffer-size", type=int, default=4096)
    parser.add_argument("--render-pad-margin", type=float, default=0.015)
    parser.add_argument("--pad-mode", choices=("ground", "cloth"), default="ground")
    parser.add_argument("--cloth-dim-x", type=int, default=18)
    parser.add_argument("--cloth-dim-y", type=int, default=18)
    parser.add_argument("--cloth-mass", type=float, default=0.00025)
    parser.add_argument("--cloth-tri-ke", type=float, default=5.0e3)
    parser.add_argument("--cloth-tri-ka", type=float, default=5.0e3)
    parser.add_argument("--cloth-tri-kd", type=float, default=5.0e-1)
    parser.add_argument("--cloth-edge-ke", type=float, default=5.0e-2)
    parser.add_argument("--cloth-edge-kd", type=float, default=1.0e-3)
    parser.add_argument("--cloth-particle-radius", type=float, default=0.0015)
    args = parser.parse_args()

    import newton
    import warp as wp

    if args.device:
        wp.set_device(args.device)

    points_cam, scene = load_scene_centerline(Path(args.scene_npz), args.num_nodes)
    radius = float(args.radius)
    if radius <= 0.0 and "radius" in scene:
        radius = float(np.asarray(scene["radius"]).reshape(()))
    if radius <= 0.0:
        radius = 0.001
    points, basis = to_local_drop_frame(
        points_cam,
        scene,
        radius=radius,
        drop_height=args.drop_height,
        use_scene_plane=args.use_scene_plane,
    )

    builder = newton.ModelBuilder(gravity=float(args.gravity))
    builder.default_shape_cfg.ke = args.contact_stiffness
    builder.default_shape_cfg.kd = args.contact_damping
    builder.default_shape_cfg.mu = args.friction
    cloth_particles = []
    cloth_grid_shape = None
    if args.pad_mode == "cloth":
        cloth_low_xy, cloth_cell_x, cloth_cell_y = make_cloth_pad_grid(
            points,
            radius=radius,
            margin=args.render_pad_margin,
            dim_x=args.cloth_dim_x,
            dim_y=args.cloth_dim_y,
        )
        cloth_start = builder.particle_count
        builder.add_cloth_grid(
            pos=wp.vec3(float(cloth_low_xy[0]), float(cloth_low_xy[1]), 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            fix_left=True,
            fix_right=True,
            fix_top=True,
            fix_bottom=True,
            dim_x=args.cloth_dim_x,
            dim_y=args.cloth_dim_y,
            cell_x=cloth_cell_x,
            cell_y=cloth_cell_y,
            mass=args.cloth_mass,
            tri_ke=args.cloth_tri_ke,
            tri_ka=args.cloth_tri_ka,
            tri_kd=args.cloth_tri_kd,
            edge_ke=args.cloth_edge_ke,
            edge_kd=args.cloth_edge_kd,
            particle_radius=args.cloth_particle_radius,
            label="deformable_cloth_pad",
        )
        cloth_particles = list(range(cloth_start, builder.particle_count))
        cloth_grid_shape = infer_cloth_grid_shape(
            len(cloth_particles), args.cloth_dim_x, args.cloth_dim_y
        )
    elif args.ground:
        builder.add_ground_plane()

    wp_points = [wp.vec3(float(x), float(y), float(z)) for x, y, z in points]
    quats = newton.utils.create_parallel_transport_cable_quaternions(wp_points)
    rod_kwargs = dict(
        positions=wp_points,
        quaternions=quats,
        radius=radius,
        stretch_stiffness=args.stretch_stiffness,
        stretch_damping=args.stretch_damping,
        bend_stiffness=args.bend_stiffness,
        bend_damping=args.bend_damping,
        closed=args.closed,
        label="reconstructed_thread_physics",
    )
    if "body_frame_origin" in inspect.signature(builder.add_rod).parameters:
        rod_kwargs["body_frame_origin"] = "com"
    rod_bodies, rod_joints = builder.add_rod(**rod_kwargs)
    rod_mass_before = np.asarray([float(builder.body_mass[int(i)]) for i in rod_bodies])

    builder.color()
    model = builder.finalize()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = None
    collision_pipeline = None
    needs_contacts = bool(args.ground or args.pad_mode == "cloth")
    if needs_contacts:
        collision_pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = collision_pipeline.contacts()
    solver_kwargs = {"iterations": args.iterations}
    if needs_contacts:
        solver_kwargs.update(
            rigid_body_contact_buffer_size=args.contact_buffer_size,
            rigid_contact_history=True,
            particle_enable_self_contact=True,
            particle_self_contact_radius=max(radius, args.cloth_particle_radius),
            particle_self_contact_margin=max(2.0 * radius, 2.0 * args.cloth_particle_radius),
            particle_enable_tile_solve=True,
        )
    solver = newton.solvers.SolverVBD(model, **solver_kwargs)

    states = [body_positions(state_0, rod_bodies)]
    velocities = [body_velocities(state_0, rod_bodies)]
    cloth_states = []
    if cloth_particles:
        particle_q = state_0.particle_q.numpy()
        cloth_states.append(
            np.asarray([particle_q[int(i)] for i in cloth_particles], dtype=np.float64)
        )
    times = [0.0]
    for step in range(1, args.steps + 1):
        state_0.clear_forces()
        if collision_pipeline is not None:
            collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, args.dt)
        state_0, state_1 = state_1, state_0
        if step % args.save_every == 0 or step == args.steps:
            states.append(body_positions(state_0, rod_bodies))
            velocities.append(body_velocities(state_0, rod_bodies))
            if cloth_particles:
                particle_q = state_0.particle_q.numpy()
                cloth_states.append(
                    np.asarray([particle_q[int(i)] for i in cloth_particles], dtype=np.float64)
                )
            times.append(step * args.dt)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    newton_states = np.asarray(states, dtype=np.float64)
    newton_velocities = np.asarray(velocities, dtype=np.float64)
    states_cam = local_points_to_camera(newton_states, basis)
    velocities_cam = local_vectors_to_camera(newton_velocities, basis)
    cloth_states = np.asarray(cloth_states, dtype=np.float64) if cloth_states else None
    pad_grid_vertices = None
    pad_grid_faces = None
    if cloth_states is not None and len(cloth_states):
        pad_grid_vertices = local_points_to_camera(cloth_states, basis)
        nx, ny = cloth_grid_shape
        pad_grid_faces = make_grid_faces(nx, ny)
    render_pad_corners = make_ground_pad_from_newton_states(
        newton_states,
        basis,
        radius=radius,
        margin=args.render_pad_margin,
    )
    replaced_keys = {
        "times",
        "states",
        "velocities",
        "newton_states",
        "newton_velocities",
        "initial_thread_local",
        "initial_centerline",
        "radius",
        "rod_bodies",
        "rod_joints",
        "rod_mass_before_finalize",
        "pad_corners",
        "pad_grid_vertices",
        "pad_grid_faces",
        "camera_center",
        "camera_x_axis",
        "camera_y_axis",
        "camera_normal",
        "local_xy_center",
        "local_z_min",
        "local_z_lift",
    }
    payload = {key: scene[key] for key in scene.files if key not in replaced_keys}
    np.savez(
        out_dir / "reconstructed_thread_newton_physics.npz",
        **payload,
        times=np.asarray(times, dtype=np.float64),
        # Camera-frame states are what the OBJ/Genesis renderer expects. Keep
        # the raw Newton z-up states separately for physics diagnostics.
        states=states_cam,
        velocities=velocities_cam,
        newton_states=newton_states,
        newton_velocities=newton_velocities,
        initial_centerline=states_cam[0],
        initial_thread_local=points,
        radius=radius,
        rod_bodies=np.asarray(rod_bodies, dtype=np.int32),
        rod_joints=np.asarray(rod_joints, dtype=np.int32),
        rod_mass_before_finalize=rod_mass_before,
        pad_corners=render_pad_corners,
        pad_grid_vertices=(
            pad_grid_vertices
            if pad_grid_vertices is not None
            else np.empty((0, 3), dtype=np.float64)
        ),
        pad_grid_faces=(
            pad_grid_faces
            if pad_grid_faces is not None
            else np.empty((0, 3), dtype=np.int32)
        ),
        # Pad surface normal points toward the thread side in camera coords;
        # the OBJ exporter thickens in the opposite direction.
        camera_to_newton_normal=np.asarray(basis["camera_normal"], dtype=np.float64),
        pad_mode=np.asarray(args.pad_mode),
        **basis,
    )

    start = newton_states[0].mean(axis=0)
    end = newton_states[-1].mean(axis=0)
    disp = np.linalg.norm(newton_states[-1] - newton_states[0], axis=1)
    cam_start = states_cam[0].mean(axis=0)
    cam_end = states_cam[-1].mean(axis=0)
    print(f"saved: {out_dir / 'reconstructed_thread_newton_physics.npz'}")
    print(f"saved states: {len(newton_states)}")
    print(f"newton center start: {start}")
    print(f"newton center end:   {end}")
    print(f"newton center dz:    {end[2] - start[2]:.9f} m")
    print(f"newton z min/max start: {newton_states[0, :, 2].min():.9f}, {newton_states[0, :, 2].max():.9f}")
    print(f"newton z min/max end:   {newton_states[-1, :, 2].min():.9f}, {newton_states[-1, :, 2].max():.9f}")
    print(f"camera center start: {cam_start}")
    print(f"camera center end:   {cam_end}")
    print(f"camera center delta: {cam_end - cam_start}")
    print(f"point displacement min/max: {disp.min():.9f}, {disp.max():.9f} m")
    print(f"velocity norm final min/max: {np.linalg.norm(newton_velocities[-1], axis=1).min():.9f}, {np.linalg.norm(newton_velocities[-1], axis=1).max():.9f} m/s")
    print(f"rod mass before finalize min/max: {rod_mass_before.min():.9e}, {rod_mass_before.max():.9e}")
    print(f"pad_mode: {args.pad_mode}")
    if pad_grid_vertices is not None:
        pad_start = pad_grid_vertices[0]
        pad_end = pad_grid_vertices[-1]
        pad_disp = np.linalg.norm(pad_end - pad_start, axis=1)
        print(f"pad particles: {pad_grid_vertices.shape[1]}")
        print(f"pad displacement min/max: {pad_disp.min():.9f}, {pad_disp.max():.9f} m")


if __name__ == "__main__":
    main()
