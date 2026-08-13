#!/usr/bin/env python3
"""Simulate a reconstructed thread spline with NVIDIA Newton.

The reconstructed spline is converted to a Newton rod:
``ModelBuilder.add_rod(...)`` builds capsule bodies connected by cable joints,
and ``SolverVBD`` advances the cable/rod dynamics.

Inputs can be either:
  - an offline reconstruction sample file: ``*_samples.npz`` with ``points``
  - a saved reconstruction pickle: ``*_spline.pkl`` containing a BSpline or
    ``{"thread": spline}``
"""

import argparse
import inspect
import pickle
from pathlib import Path

import numpy as np


def parse_vec3(text):
    vals = [float(v.strip()) for v in text.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return tuple(vals)


def load_points(path, num_nodes, input_units):
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "points" not in data:
            raise ValueError(f"{path} does not contain a 'points' array")
        points = np.asarray(data["points"], dtype=float)
    elif path.suffix == ".pkl":
        with path.open("rb") as f:
            obj = pickle.load(f)
        spline = obj.get("thread") if isinstance(obj, dict) else obj
        u = np.linspace(0.0, 1.0, max(200, num_nodes * 3))
        points = np.asarray(spline(u), dtype=float)
    else:
        raise ValueError("input must be a .npz or .pkl file")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected points with shape (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("input points contain NaN or inf")

    scale_by_unit = {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
    }
    points = points * scale_by_unit[input_units]
    return resample_arclength(points, num_nodes)


def resample_arclength(points, n):
    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    keep = np.concatenate([[True], seg > 1e-10])
    points = points[keep]
    if len(points) < 3:
        raise ValueError("need at least three distinct centerline points")

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, s[-1], n)
    out = np.empty((n, 3), dtype=float)
    for dim in range(3):
        out[:, dim] = np.interp(target, s, points[:, dim])
    return out


def import_newton():
    try:
        import newton
        import warp as wp
    except ModuleNotFoundError as exc:
        missing = exc.name or "newton"
        raise SystemExit(
            f"Missing dependency '{missing}'. Install NVIDIA Newton first:\n"
            '  pip install "newton[examples]"\n'
            "Newton requires Python 3.10+; GPU runs need a supported NVIDIA "
            "driver, while macOS is CPU-only."
        ) from exc
    return newton, wp


def set_body_kinematic(builder, wp, body_idx):
    builder.body_mass[body_idx] = 0.0
    builder.body_inv_mass[body_idx] = 0.0
    builder.body_inertia[body_idx] = wp.mat33(0.0)
    builder.body_inv_inertia[body_idx] = wp.mat33(0.0)


def pin_rod_bodies(builder, wp, rod_bodies, pin_mode):
    if pin_mode in ("start", "both"):
        set_body_kinematic(builder, wp, int(rod_bodies[0]))
    if pin_mode in ("end", "both"):
        set_body_kinematic(builder, wp, int(rod_bodies[-1]))


def driven_body_index(rod_bodies, drive_end):
    if drive_end == "start":
        return int(rod_bodies[0])
    if drive_end == "end":
        return int(rod_bodies[-1])
    return None


def set_body_translation(state, body_idx, xyz):
    body_q = state.body_q.numpy()
    body_q[int(body_idx), :3] = np.asarray(xyz, dtype=body_q.dtype)
    state.body_q.assign(body_q)
    if getattr(state, "body_qd", None) is not None:
        body_qd = state.body_qd.numpy()
        body_qd[int(body_idx), :] = 0.0
        state.body_qd.assign(body_qd)


def body_positions(state, body_indices):
    body_q = state.body_q.numpy()
    return np.asarray([body_q[int(i)][:3] for i in body_indices], dtype=np.float64)


def make_random_drive_trajectory(anchor, steps, dt, amplitude, waypoints, seed):
    rng = np.random.default_rng(seed)
    n_waypoints = max(int(waypoints), 2)
    key_t = np.linspace(0.0, steps * dt, n_waypoints)
    total_time = max(key_t[-1], dt)

    offsets = rng.normal(size=(n_waypoints, 3))
    offsets[:, 2] *= 0.6
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    offsets = offsets / np.maximum(norms, 1e-12)
    scales = rng.uniform(0.35, 1.0, size=(n_waypoints, 1))
    offsets = offsets * scales * float(amplitude)
    offsets[0] = 0.0

    t = np.arange(steps + 1, dtype=float) * dt
    out = np.empty((steps + 1, 3), dtype=float)
    for i, ti in enumerate(t):
        seg = min(np.searchsorted(key_t, ti, side="right") - 1, n_waypoints - 2)
        seg = max(seg, 0)
        u = (ti - key_t[seg]) / max(key_t[seg + 1] - key_t[seg], 1e-12)
        u = np.clip(u, 0.0, 1.0)
        smooth = u * u * (3.0 - 2.0 * u)
        out[i] = anchor + (1.0 - smooth) * offsets[seg] + smooth * offsets[seg + 1]
    out[-1] = anchor + offsets[-1]
    return out


def simulate_newton(points, args):
    newton, wp = import_newton()
    if args.device:
        wp.set_device(args.device)

    # Newton's ModelBuilder constructor takes a signed scalar along up_axis,
    # not a full vector. With the default Z-up builder, (0, 0, -9.81) -> -9.81.
    builder = newton.ModelBuilder(gravity=float(args.gravity[2]))
    builder.default_shape_cfg.ke = args.contact_stiffness
    builder.default_shape_cfg.kd = args.contact_damping
    builder.default_shape_cfg.mu = args.friction

    wp_points = [wp.vec3(float(x), float(y), float(z)) for x, y, z in points]
    quats = newton.utils.create_parallel_transport_cable_quaternions(wp_points)

    rod_kwargs = dict(
        positions=wp_points,
        quaternions=quats,
        radius=args.radius,
        stretch_stiffness=args.stretch_stiffness,
        stretch_damping=args.stretch_damping,
        bend_stiffness=args.bend_stiffness,
        bend_damping=args.bend_damping,
        closed=args.closed,
        label="reconstructed_thread",
    )
    if "body_frame_origin" in inspect.signature(builder.add_rod).parameters:
        rod_kwargs["body_frame_origin"] = "com"
    rod_bodies, rod_joints = builder.add_rod(**rod_kwargs)
    pin_rod_bodies(builder, wp, rod_bodies, args.pin)
    drive_body = driven_body_index(rod_bodies, args.drive_end)
    if drive_body is not None:
        set_body_kinematic(builder, wp, drive_body)

    if args.ground:
        builder.add_ground_plane()

    builder.color()
    model = builder.finalize()
    collision_pipeline = newton.CollisionPipeline(model) if args.ground or args.contacts else None
    contacts = collision_pipeline.contacts() if collision_pipeline is not None else None
    solver = newton.solvers.SolverVBD(
        model,
        iterations=args.iterations,
        rigid_avbd_contact_alpha=args.contact_alpha,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    drive_traj = None
    if drive_body is not None:
        anchor = points[0] if args.drive_end == "start" else points[-1]
        drive_traj = make_random_drive_trajectory(
            anchor=anchor,
            steps=args.steps,
            dt=args.dt,
            amplitude=args.drive_amplitude,
            waypoints=args.drive_waypoints,
            seed=args.drive_seed,
        )
        set_body_translation(state_0, drive_body, drive_traj[0])
        set_body_translation(state_1, drive_body, drive_traj[0])

    states = [body_positions(state_0, rod_bodies)]
    times = [0.0]
    saved_drive = [drive_traj[0].copy()] if drive_traj is not None else []

    for step in range(1, args.steps + 1):
        state_0.clear_forces()
        if drive_body is not None:
            set_body_translation(state_0, drive_body, drive_traj[step])
            set_body_translation(state_1, drive_body, drive_traj[step])
        refresh_contacts = (
            collision_pipeline is not None
            and (step == 1 or step % args.contact_interval == 0)
        )
        if refresh_contacts:
            collision_pipeline.collide(state_0, contacts)
        if hasattr(solver, "set_rigid_history_update"):
            solver.set_rigid_history_update(bool(refresh_contacts))
        solver.step(state_0, state_1, control, contacts, args.dt)
        state_0, state_1 = state_1, state_0

        if step % args.save_every == 0 or step == args.steps:
            states.append(body_positions(state_0, rod_bodies))
            times.append(step * args.dt)
            if drive_traj is not None:
                saved_drive.append(drive_traj[step].copy())

    result = {
        "times": np.asarray(times, dtype=np.float64),
        "states": np.asarray(states, dtype=np.float64),
        "initial_centerline": points,
        "rod_bodies": np.asarray(rod_bodies, dtype=np.int32),
        "rod_joints": np.asarray(rod_joints, dtype=np.int32),
    }
    if drive_traj is not None:
        result["drive_trajectory"] = np.asarray(saved_drive, dtype=np.float64)
        result["drive_end"] = np.asarray(args.drive_end)
    return result


def write_preview(out_dir, initial, states):
    try:
        import os
        import tempfile

        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(Path(tempfile.gettempdir()) / "thread_reconstruction_matplotlib"),
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"preview skipped: matplotlib unavailable ({exc})")
        return

    final = states[-1]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(initial[:, 0], initial[:, 1], initial[:, 2], color="0.55", lw=2, label="input centerline")
    ax.plot(final[:, 0], final[:, 1], final[:, 2], color="tab:red", lw=2, label="Newton final")
    ax.scatter(final[0, 0], final[0, 1], final[0, 2], color="cyan", s=35)
    ax.scatter(final[-1, 0], final[-1, 1], final[-1, 2], color="gold", s=35)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "newton_thread_preview_3d.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(initial[:, 0], initial[:, 1], color="0.55", lw=2, label="input centerline")
    ax.plot(final[:, 0], final[:, 1], color="tab:red", lw=2, label="Newton final")
    ax.scatter(final[0, 0], final[0, 1], color="cyan", s=35)
    ax.scatter(final[-1, 0], final[-1, 1], color="gold", s=35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "newton_thread_preview_xy.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="*_samples.npz or *_spline.pkl")
    parser.add_argument("--out-dir", default="newton_thread_sim")
    parser.add_argument("--input-units", choices=("m", "cm", "mm"), default="mm")
    parser.add_argument("--num-nodes", type=int, default=65, help="rod centerline nodes; segments = nodes - 1")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--dt", type=float, default=1.0 / 1200.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", default=None, help="Newton/Warp device, e.g. cpu or cuda:0")

    parser.add_argument("--radius", type=float, default=0.001, help="rod/capsule radius in meters")
    parser.add_argument("--stretch-stiffness", type=float, default=1.0e6)
    parser.add_argument("--stretch-damping", type=float, default=1.0e3)
    parser.add_argument("--bend-stiffness", type=float, default=1.0e4)
    parser.add_argument("--bend-damping", type=float, default=1.0e2)
    parser.add_argument(
        "--twist-stiffness",
        type=float,
        default=None,
        help="accepted for compatibility; Newton add_rod uses bend-stiffness for bend/twist",
    )
    parser.add_argument(
        "--twist-damping",
        type=float,
        default=None,
        help="accepted for compatibility; Newton add_rod uses bend-damping for bend/twist",
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--gravity", type=parse_vec3, default=(0.0, 0.0, -9.81))
    parser.add_argument("--pin", choices=("none", "start", "end", "both"), default="both")
    parser.add_argument("--closed", action="store_true")
    parser.add_argument("--drive-end", choices=("none", "start", "end"), default="none")
    parser.add_argument("--drive-amplitude", type=float, default=0.018, help="random endpoint motion amplitude in meters")
    parser.add_argument("--drive-waypoints", type=int, default=7, help="number of smooth random trajectory waypoints")
    parser.add_argument("--drive-seed", type=int, default=7)

    parser.add_argument("--ground", action="store_true")
    parser.add_argument("--contacts", action="store_true", help="enable collision pipeline without adding a ground plane")
    parser.add_argument("--contact-interval", type=int, default=10)
    parser.add_argument("--contact-alpha", type=float, default=0.0)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e4)
    parser.add_argument("--contact-damping", type=float, default=0.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="write diagnostic Matplotlib PNG previews of the Newton trajectory",
    )
    args = parser.parse_args()

    if args.num_nodes < 3:
        raise ValueError("--num-nodes must be at least 3")
    if args.save_every < 1:
        raise ValueError("--save-every must be at least 1")
    if args.contact_interval < 1:
        raise ValueError("--contact-interval must be at least 1")

    points = load_points(args.input, args.num_nodes, args.input_units)
    result = simulate_newton(points, args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "newton_thread_simulation.npz",
        **result,
        input=str(Path(args.input).resolve()),
        input_units=args.input_units,
        radius=args.radius,
        stretch_stiffness=args.stretch_stiffness,
        bend_stiffness=args.bend_stiffness,
        pin=args.pin,
    )
    if args.preview:
        write_preview(out_dir, result["initial_centerline"], result["states"])
    print(f"saved {len(result['states'])} Newton states to {out_dir / 'newton_thread_simulation.npz'}")


if __name__ == "__main__":
    main()
