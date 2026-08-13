#!/usr/bin/env python3
"""Run a reconstructed thread as a Newton cable scene.

This follows the structure of Newton's official cable examples: build an
``add_rod`` cable, add a static contact pad, step with ``CollisionPipeline``
and ``SolverVBD``, and let Newton's viewer stack write GL/USD/RTX output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def resample_arclength(points: np.ndarray, n: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-10])
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


def normalize(v: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.asarray(fallback, dtype=np.float64)
    return v / n


def fit_plane_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    x_axis = normalize(vh[0], (1.0, 0.0, 0.0))
    normal = normalize(vh[-1], (0.0, 0.0, 1.0))
    if normal[2] < 0.0:
        normal = -normal
    y_axis = normalize(np.cross(normal, x_axis), (0.0, 1.0, 0.0))
    x_axis = normalize(np.cross(y_axis, normal), (1.0, 0.0, 0.0))
    return center, x_axis, y_axis, normal


def load_scene_points(path: Path, num_nodes: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    if "initial_centerline" in data:
        points = np.asarray(data["initial_centerline"], dtype=np.float64)
    elif "states" in data:
        points = np.asarray(data["states"][0], dtype=np.float64)
    else:
        raise ValueError(f"{path} does not contain initial_centerline or states")
    points = resample_arclength(points, num_nodes)
    return points, {key: data[key] for key in data.files}


def make_local_thread(
    points_cam: np.ndarray,
    scene: dict[str, np.ndarray],
    radius: float,
    drop_height: float,
    use_scene_plane: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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
        center, x_axis, y_axis, normal = fit_plane_axes(points_cam)

    rel = points_cam - center
    local = np.column_stack([rel @ x_axis, rel @ y_axis, rel @ normal])
    local[:, 2] -= np.min(local[:, 2])
    local[:, 2] += radius + drop_height
    local[:, :2] -= np.mean(local[:, :2], axis=0, keepdims=True)
    basis = {
        "camera_center": center,
        "camera_x_axis": x_axis,
        "camera_y_axis": y_axis,
        "camera_normal": normal,
    }
    return local, basis


class ReconstructedCableExample:
    def __init__(self, viewer, args):
        import newton
        import warp as wp

        self.viewer = viewer
        self.args = args
        self.wp = wp
        self.fps = args.fps
        self.frame_dt = 1.0 / float(self.fps)
        self.sim_substeps = args.substeps
        self.sim_iterations = args.iterations
        self.sim_dt = self.frame_dt / float(self.sim_substeps)
        self.sim_time = 0.0
        self.saved_states: list[np.ndarray] = []
        self.saved_times: list[float] = []

        points_cam, scene = load_scene_points(Path(args.scene_npz), args.num_nodes)
        radius = float(args.radius)
        if radius <= 0.0 and "radius" in scene:
            radius = float(np.asarray(scene["radius"]).reshape(()))
        if radius <= 0.0:
            radius = 0.001
        self.radius = radius
        thread_local, basis = make_local_thread(
            points_cam,
            scene,
            radius=radius,
            drop_height=args.drop_height,
            use_scene_plane=not args.ignore_scene_plane,
        )
        if args.closed:
            gap = np.linalg.norm(thread_local[0] - thread_local[-1])
            if gap > 1e-8:
                thread_local = np.vstack([thread_local, thread_local[0]])
        self.initial_thread_local = thread_local
        self.basis = basis

        builder = newton.ModelBuilder(gravity=-abs(float(args.gravity)))
        builder.rigid_gap = args.rigid_gap_scale * radius
        builder.default_shape_cfg.ke = args.contact_stiffness
        builder.default_shape_cfg.kd = args.contact_damping
        builder.default_shape_cfg.mu = args.friction

        if args.pad:
            pad_half = np.max(np.abs(thread_local[:, :2]), axis=0) + args.pad_margin
            pad_half = np.maximum(pad_half, np.asarray([args.pad_min_half, args.pad_min_half]))
            pad_cfg = newton.ModelBuilder.ShapeConfig(
                density=1000.0,
                ke=args.contact_stiffness,
                kd=args.contact_damping,
                mu=args.friction,
            )
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(
                    wp.vec3(0.0, 0.0, -0.5 * args.pad_thickness),
                    wp.quat_identity(),
                ),
                hx=float(pad_half[0]),
                hy=float(pad_half[1]),
                hz=0.5 * args.pad_thickness,
                cfg=pad_cfg,
                color=(0.72, 0.34, 0.30),
                label="deformable_pad_proxy",
            )

        cable_cfg = newton.ModelBuilder.ShapeConfig(
            density=args.cable_density,
            ke=args.contact_stiffness,
            kd=args.contact_damping,
            mu=args.friction,
            restitution=0.0,
        )
        wp_points = [wp.vec3(float(x), float(y), float(z)) for x, y, z in thread_local]
        quats = newton.utils.create_parallel_transport_cable_quaternions(wp_points)
        rod_kwargs = dict(
            positions=wp_points,
            quaternions=quats,
            radius=radius,
            cfg=cable_cfg,
            stretch_stiffness=args.stretch_stiffness,
            stretch_damping=args.stretch_damping,
            bend_stiffness=args.bend_stiffness,
            bend_damping=args.bend_damping,
            closed=args.closed,
            label="reconstructed_thread_cable",
        )
        if "body_frame_origin" in newton.ModelBuilder.add_rod.__annotations__:
            rod_kwargs["body_frame_origin"] = "com"
        else:
            try:
                import inspect

                if "body_frame_origin" in inspect.signature(builder.add_rod).parameters:
                    rod_kwargs["body_frame_origin"] = "com"
            except (TypeError, ValueError):
                pass
        self.rod_bodies, self.rod_joints = builder.add_rod(**rod_kwargs)
        self.rod_body_mass_before_finalize = np.asarray(
            [float(builder.body_mass[int(i)]) for i in self.rod_bodies],
            dtype=np.float64,
        )

        builder.color()
        self.model = builder.finalize()
        self.collision_pipeline = None
        self.contacts = None
        if args.pad or args.contacts:
            self.collision_pipeline = newton.CollisionPipeline(self.model, contact_matching="latest")
            self.contacts = self.collision_pipeline.contacts()
        solver_kwargs = {"iterations": self.sim_iterations}
        if self.collision_pipeline is not None:
            solver_kwargs.update(
                rigid_body_contact_buffer_size=args.contact_buffer_size,
                rigid_contact_history=True,
            )
        self.solver = newton.solvers.SolverVBD(self.model, **solver_kwargs)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = args.fov
        self.record_state()
        self.graph = None
        if args.capture:
            self.capture()

    def record_state(self):
        body_q = self.state_0.body_q.numpy()
        self.saved_states.append(np.asarray([body_q[int(i), :3] for i in self.rod_bodies], dtype=np.float64))
        self.saved_times.append(float(self.sim_time))

    def capture(self):
        with self.wp.ScopedCapture() as cap:
            self.simulate()
        self.graph = cap.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.collision_pipeline is not None:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            self.wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.record_state()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def write_outputs(self):
        out_dir = Path(self.args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        states = np.asarray(self.saved_states, dtype=np.float64)
        np.savez(
            out_dir / "reconstructed_thread_cable_newton.npz",
            times=np.asarray(self.saved_times, dtype=np.float64),
            states=states,
            initial_thread_local=self.initial_thread_local,
            radius=self.radius,
            rod_bodies=np.asarray(self.rod_bodies, dtype=np.int32),
            rod_joints=np.asarray(self.rod_joints, dtype=np.int32),
            rod_body_mass_before_finalize=self.rod_body_mass_before_finalize,
            **self.basis,
        )
        start = states[0].mean(axis=0)
        end = states[-1].mean(axis=0)
        disp = np.linalg.norm(states[-1] - states[0], axis=1)
        print(f"saved: {out_dir / 'reconstructed_thread_cable_newton.npz'}")
        print(f"saved states: {len(states)}")
        print(f"center start: {start}")
        print(f"center end:   {end}")
        print(f"center dz:    {end[2] - start[2]:.6f} m")
        print(f"z min/max start: {states[0, :, 2].min():.6f}, {states[0, :, 2].max():.6f}")
        print(f"z min/max end:   {states[-1, :, 2].min():.6f}, {states[-1, :, 2].max():.6f}")
        print(f"point displacement min/max: {disp.min():.6f}, {disp.max():.6f} m")
        print(
            "rod body mass before finalize min/max: "
            f"{self.rod_body_mass_before_finalize.min():.6e}, "
            f"{self.rod_body_mass_before_finalize.max():.6e}"
        )


def build_parser():
    import newton.examples

    parser = newton.examples.create_parser()
    parser.add_argument("--scene-npz", required=True)
    parser.add_argument("--out-dir", default="newton_reconstructed_cable_000000")
    parser.add_argument("--num-nodes", type=int, default=64)
    parser.add_argument("--radius", type=float, default=-1.0)
    parser.add_argument("--closed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ignore-scene-plane", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop-height", type=float, default=0.035)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--stretch-stiffness", type=float, default=5.0e5)
    parser.add_argument("--stretch-damping", type=float, default=2.0e1)
    parser.add_argument("--bend-stiffness", type=float, default=2.0e1)
    parser.add_argument("--bend-damping", type=float, default=2.0e1)
    parser.add_argument("--cable-density", type=float, default=800.0)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e5)
    parser.add_argument("--contact-damping", type=float, default=0.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--rigid-gap-scale", type=float, default=0.5)
    parser.add_argument("--contact-buffer-size", type=int, default=512)
    parser.add_argument("--pad-margin", type=float, default=0.030)
    parser.add_argument("--pad-min-half", type=float, default=0.050)
    parser.add_argument("--pad-thickness", type=float, default=0.006)
    parser.add_argument("--pad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--contacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable collision pipeline even when the pad is disabled",
    )
    parser.add_argument("--fov", type=float, default=38.0)
    parser.add_argument(
        "--capture",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use CUDA graph capture like the Newton examples; disabled by default for debugging",
    )
    parser.add_argument(
        "--manual-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="explicitly step/render num-frames instead of relying on the viewer loop",
    )
    return parser


def main():
    import newton.examples

    parser = build_parser()
    viewer, args = newton.examples.init(parser)
    example = ReconstructedCableExample(viewer, args)
    if args.manual_run:
        for _ in range(args.num_frames):
            example.step()
            example.render()
        viewer.close()
    else:
        newton.examples.run(example, args)
    example.write_outputs()


if __name__ == "__main__":
    main()
