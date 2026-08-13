#!/usr/bin/env python3
"""Minimal Newton gravity sanity checks.

This intentionally avoids project reconstruction code. It answers two questions:
  1. Does a plain dynamic rigid body move under gravity?
  2. Does a Newton add_rod chain move under gravity when unpinned?
"""

import argparse
import inspect

import numpy as np


def import_newton():
    import newton
    import warp as wp

    return newton, wp


def body_positions(state, body_indices):
    body_q = state.body_q.numpy()
    return np.asarray([body_q[int(i), :3] for i in body_indices], dtype=float)


def step_model(newton, model, state_0, state_1, steps, dt, iterations):
    solver = newton.solvers.SolverVBD(model, iterations=iterations)
    control = model.control()
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
    return state_0


def try_add_sphere(builder, newton, wp):
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()))
    cfg = builder.default_shape_cfg
    signatures = [
        ("add_shape_sphere", {"body": body, "pos": wp.vec3(0.0, 0.0, 0.0), "rot": wp.quat_identity(), "radius": 0.01, "cfg": cfg}),
        ("add_shape_sphere", {"body": body, "radius": 0.01, "cfg": cfg}),
        ("add_sphere", {"body": body, "radius": 0.01}),
    ]
    for name, kwargs in signatures:
        fn = getattr(builder, name, None)
        if fn is None:
            continue
        try:
            accepted = inspect.signature(fn).parameters
            filtered = {k: v for k, v in kwargs.items() if k in accepted}
            fn(**filtered)
            return body
        except Exception:
            continue
    return None


def rigid_body_test(args):
    newton, wp = import_newton()
    if args.device:
        wp.set_device(args.device)
    builder = newton.ModelBuilder(gravity=-9.81)
    body = try_add_sphere(builder, newton, wp)
    if body is None:
        print("rigid_body_test: SKIPPED (no compatible sphere API found)")
        return
    builder.color()
    model = builder.finalize()
    state_0 = model.state()
    state_1 = model.state()
    start = body_positions(state_0, [body])[0]
    final_state = step_model(newton, model, state_0, state_1, args.steps, args.dt, args.iterations)
    end = body_positions(final_state, [body])[0]
    print("rigid_body_test: RUN")
    print(f"  start: {start}")
    print(f"  end:   {end}")
    print(f"  dz:    {end[2] - start[2]:.9f}")


def rod_test(args):
    newton, wp = import_newton()
    if args.device:
        wp.set_device(args.device)
    builder = newton.ModelBuilder(gravity=-9.81)
    points = [
        wp.vec3(-0.04, 0.0, 0.15),
        wp.vec3(-0.02, 0.0, 0.15),
        wp.vec3(0.0, 0.0, 0.15),
        wp.vec3(0.02, 0.0, 0.15),
        wp.vec3(0.04, 0.0, 0.15),
    ]
    quats = newton.utils.create_parallel_transport_cable_quaternions(points)
    rod_kwargs = dict(
        positions=points,
        quaternions=quats,
        radius=0.001,
        stretch_stiffness=args.stretch_stiffness,
        stretch_damping=args.stretch_damping,
        bend_stiffness=args.bend_stiffness,
        bend_damping=args.bend_damping,
        closed=False,
        label="gravity_sanity_rod",
    )
    if "body_frame_origin" in inspect.signature(builder.add_rod).parameters:
        rod_kwargs["body_frame_origin"] = "com"
    rod_bodies, _rod_joints = builder.add_rod(**rod_kwargs)
    builder.color()
    model = builder.finalize()
    state_0 = model.state()
    state_1 = model.state()
    start = body_positions(state_0, rod_bodies)
    final_state = step_model(newton, model, state_0, state_1, args.steps, args.dt, args.iterations)
    end = body_positions(final_state, rod_bodies)
    print("rod_test: RUN")
    print(f"  start_center: {start.mean(axis=0)}")
    print(f"  end_center:   {end.mean(axis=0)}")
    print(f"  center_dz:    {end[:, 2].mean() - start[:, 2].mean():.9f}")
    print(f"  start_z_minmax: {start[:, 2].min():.9f}, {start[:, 2].max():.9f}")
    print(f"  end_z_minmax:   {end[:, 2].min():.9f}, {end[:, 2].max():.9f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=1.0 / 1200.0)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--stretch-stiffness", type=float, default=1.0e3)
    parser.add_argument("--stretch-damping", type=float, default=1.0e1)
    parser.add_argument("--bend-stiffness", type=float, default=1.0e-1)
    parser.add_argument("--bend-damping", type=float, default=1.0e-2)
    args = parser.parse_args()
    rigid_body_test(args)
    rod_test(args)


if __name__ == "__main__":
    main()
