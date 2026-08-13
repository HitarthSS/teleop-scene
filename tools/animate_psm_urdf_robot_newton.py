#!/usr/bin/env python3
"""Robot-only Newton URDF animation test for the dVRK PSM.

No thread, no pad, no contact. This isolates whether Newton can import the PSM
URDF, apply the recorded joint state, move the jaw/arm joints, and export a
readable USD animation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from export_frame_scene_obj import write_obj
from export_thread_robot_newton_scene import make_usd_viewer
from render_frame_scene_with_psm_urdf import (
    choose_ee_link,
    forward_kinematics,
    joint_values_by_name,
    load_mesh_instances,
    parse_urdf,
    transform_points,
)
from render_thread_robot_newton_gl import cam_to_newton_view
from simulate_psm_urdf_thread_contact import (
    coord_span,
    dof_span,
    joint_short_name,
    make_importable_urdf,
    matrix_to_quat_xyzw,
    set_imported_joint_coordinates,
)


def camera_to_newton_transform():
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return t


def smoothstep(x):
    x = np.clip(float(x), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def animated_values(base_values, frame, frames, jaw_scale, wrist_yaw_delta, wrist_pitch_delta, insertion_delta):
    values = dict(base_values)
    phase = smoothstep(frame / max(frames - 1, 1))
    swing = np.sin(2.0 * np.pi * frame / max(frames - 1, 1))

    for name in list(values):
        lname = name.lower()
        if any(k in lname for k in ("jaw", "gripper", "finger", "scissor")):
            values[name] = float(base_values[name]) * ((1.0 - phase) + phase * jaw_scale)
        elif "wrist_yaw" in lname:
            values[name] = float(base_values[name]) + wrist_yaw_delta * swing
        elif "wrist_pitch" in lname:
            values[name] = float(base_values[name]) + wrist_pitch_delta * swing
        elif "insertion" in lname:
            values[name] = float(base_values[name]) + insertion_delta * phase
    return values


def values_to_q(builder, imported_joint_indices, values, q_template, qd_template):
    q = np.asarray(q_template, dtype=np.float32).copy()
    qd = np.asarray(qd_template, dtype=np.float32).copy()
    qd[:] = 0.0
    for joint_idx in imported_joint_indices:
        name = joint_short_name(builder.joint_label[joint_idx])
        if name not in values:
            continue
        q0, q1 = coord_span(builder, joint_idx)
        if q1 > q0:
            q[q0] = float(values[name])
    return q, qd


def write_robot_obj(path, links, fk, cam_to_base, args):
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
    for i, (verts, faces, transform, link, _mesh_path) in enumerate(instances):
        verts_cam = transform_points(verts, transform)
        meshes.append(
            (
                f"psm_mesh_{i:03d}_{link}",
                "tool",
                cam_to_newton_view(verts_cam),
                np.asarray(faces, dtype=np.int32),
            )
        )
    write_obj(Path(path), meshes)
    return len(instances)


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
    parser.add_argument("--include-mesh-regex", default="tool_wrist|tool_wrist_sca")
    parser.add_argument("--exclude-mesh-regex", default="tool_main_link")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--jaw-scale", type=float, default=0.15)
    parser.add_argument("--wrist-yaw-delta", type=float, default=0.25)
    parser.add_argument("--wrist-pitch-delta", type=float, default=0.18)
    parser.add_argument("--insertion-delta", type=float, default=0.0)
    parser.add_argument("--robot-target-ke", type=float, default=1.0e5)
    parser.add_argument("--robot-target-kd", type=float, default=1.0e3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import newton
    import warp as wp

    if args.device:
        wp.set_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_dir = out_dir / "obj_frames"
    obj_dir.mkdir(parents=True, exist_ok=True)

    scene = np.load(args.scene_npz, allow_pickle=True)
    links, joints = parse_urdf(args.urdf, args.package_root)
    q_recorded = np.load(args.joints)
    jaw_recorded = np.load(args.jaw) if args.jaw else None
    base_values = joint_values_by_name(joints, q_recorded, jaw_recorded)
    fk0 = forward_kinematics(links, joints, base_values)
    ee_link = choose_ee_link(links, args.ee_link)
    cam_to_ee = np.asarray(scene["tool_cam_to_ee"], dtype=np.float64)
    cam_to_base = cam_to_ee @ np.linalg.inv(fk0[ee_link])
    newton_to_base = camera_to_newton_transform() @ cam_to_base
    base_pos = newton_to_base[:3, 3]
    base_quat = matrix_to_quat_xyzw(newton_to_base[:3, :3])

    temp_urdf = make_importable_urdf(
        args.urdf,
        args.package_root,
        out_dir / "psm1_si_newton_robot_motion.urdf",
        remove_collisions=True,
    )

    builder = newton.ModelBuilder(gravity=0.0)
    joint_start = len(builder.joint_label)
    builder.add_urdf(
        str(temp_urdf),
        xform=wp.transform(
            wp.vec3(float(base_pos[0]), float(base_pos[1]), float(base_pos[2])),
            wp.quat(float(base_quat[0]), float(base_quat[1]), float(base_quat[2]), float(base_quat[3])),
        ),
        floating=False,
        parse_visuals_as_colliders=False,
        enable_self_collisions=False,
        ignore_inertial_definitions=True,
        joint_ordering="dfs",
        bodies_follow_joint_ordering=True,
    )
    joint_end = len(builder.joint_label)
    imported_joints = list(range(joint_start, joint_end))
    applied = set_imported_joint_coordinates(
        builder,
        imported_joints,
        base_values,
        args.robot_target_ke,
        args.robot_target_kd,
    )

    builder.color()
    model = builder.finalize()
    state = model.state()
    q0 = np.asarray(model.joint_q.numpy(), dtype=np.float32)
    qd0 = np.asarray(model.joint_qd.numpy(), dtype=np.float32)

    viewer = make_usd_viewer(out_dir / "psm_robot_motion_newton.usd", args.fps)
    viewer.set_model(model)

    mesh_counts = []
    frame_summaries = []
    for frame in range(args.frames):
        t = frame / float(args.fps)
        values = animated_values(
            base_values,
            frame,
            args.frames,
            args.jaw_scale,
            args.wrist_yaw_delta,
            args.wrist_pitch_delta,
            args.insertion_delta,
        )
        q, qd = values_to_q(builder, imported_joints, values, q0, qd0)
        model.joint_q.assign(q)
        model.joint_qd.assign(qd)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        viewer.begin_frame(t)
        viewer.log_state(state)
        viewer.end_frame()

        fk = forward_kinematics(links, joints, values)
        mesh_count = write_robot_obj(obj_dir / f"frame_{frame:06d}_psm_robot.obj", links, fk, cam_to_base, args)
        mesh_counts.append(mesh_count)
        jaw_values = {name: val for name, val in values.items() if "jaw" in name.lower()}
        frame_summaries.append((frame, jaw_values))
        print(f"[{frame}] t={t:.3f}s meshes={mesh_count} jaw={jaw_values}")

    viewer.close()
    report_lines = [
        f"scene_npz: {args.scene_npz}",
        f"urdf: {args.urdf}",
        f"temp_newton_urdf: {temp_urdf}",
        f"ee_link: {ee_link}",
        f"frames: {args.frames}",
        f"fps: {args.fps}",
        f"mesh_count_min_max: {min(mesh_counts)}, {max(mesh_counts)}",
        f"base_pos_newton: {base_pos}",
        f"base_quat_xyzw_newton: {base_quat}",
        "applied_recorded_joints:",
        *[f"  {name}: {value:.9g}" for _idx, name, value, *_rest in applied],
        "animated_frames:",
        *[f"  frame {frame}: {jaw_values}" for frame, jaw_values in frame_summaries],
    ]
    (out_dir / "psm_robot_motion_report.txt").write_text("\n".join(report_lines) + "\n")
    print(f"USD: {out_dir / 'psm_robot_motion_newton.usd'}")
    print(f"OBJ frames: {obj_dir}")
    print(f"report: {out_dir / 'psm_robot_motion_report.txt'}")


if __name__ == "__main__":
    main()
