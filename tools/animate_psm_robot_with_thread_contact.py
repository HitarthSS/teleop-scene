#!/usr/bin/env python3
"""Robot-motion + Newton cable contact test.

This starts from the robot-only animation that worked, then adds the
reconstructed thread as a Newton rod in the same camera-to-Newton viewer frame.
The PSM visual URDF moves through Newton articulation FK; jaw-link capsule
colliders are attached to the imported jaw links.
"""

from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path

import numpy as np

from animate_psm_urdf_robot_newton import (
    animated_values,
    camera_to_newton_transform,
    values_to_q,
    write_robot_obj,
)
from export_frame_scene_obj import tube_mesh, write_obj
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
    add_capsule_shape,
    body_link_name,
    body_positions,
    capsule_from_link_visuals,
    compile_re,
    joint_short_name,
    link_visual_points,
    make_importable_urdf,
    matrix_to_quat_xyzw,
    set_imported_joint_coordinates,
)

SCRIPT_VERSION = "psm_robot_thread_contact_v6_kinematic_endpoint_drag"


def write_thread_robot_obj(path, links, fk, cam_to_base, thread_state, radius, args):
    thread_v, thread_f = tube_mesh(thread_state, radius, args.thread_sides)
    robot_path = Path(path)
    robot_path.parent.mkdir(parents=True, exist_ok=True)
    instances = []
    from render_frame_scene_with_psm_urdf import load_mesh_instances

    for i, (verts, faces, transform, link, _mesh_path) in enumerate(
        load_mesh_instances(
            links,
            fk,
            cam_to_base,
            include_link=args.include_link_regex,
            exclude_link=args.exclude_link_regex,
            include_mesh=args.include_mesh_regex,
            exclude_mesh=args.exclude_mesh_regex,
        )
    ):
        verts_cam = transform_points(verts, transform)
        instances.append(
            (
                f"psm_mesh_{i:03d}_{link}",
                "tool",
                cam_to_newton_view(verts_cam),
                np.asarray(faces, dtype=np.int32),
            )
        )
    write_obj(
        robot_path,
        [
            ("newton_thread", "thread", thread_v, thread_f),
            *instances,
        ],
    )
    return len(instances)


def save_body_subset(state, body_indices):
    body_indices = np.asarray(body_indices, dtype=np.int32)
    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy() if getattr(state, "body_qd", None) is not None else None
    saved_q = np.asarray(body_q[body_indices], dtype=body_q.dtype).copy()
    saved_qd = None if body_qd is None else np.asarray(body_qd[body_indices], dtype=body_qd.dtype).copy()
    return body_indices, saved_q, saved_qd


def restore_body_subset(state, saved):
    body_indices, saved_q, saved_qd = saved
    body_q = state.body_q.numpy()
    body_q[body_indices] = saved_q
    state.body_q.assign(body_q)
    if saved_qd is not None and getattr(state, "body_qd", None) is not None:
        body_qd = state.body_qd.numpy()
        body_qd[body_indices] = saved_qd
        state.body_qd.assign(body_qd)


def smoothstep(x):
    x = np.clip(float(x), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def jaw_center_newton(links, fk, cam_to_base, include_regex):
    regex = compile_re(include_regex)
    pts = []
    for link, visuals in links.items():
        if regex is not None and regex.search(link) is None:
            continue
        if link not in fk:
            continue
        for visual in visuals:
            local_pts = link_visual_points([visual])
            if len(local_pts) == 0:
                continue
            cam_pts = transform_points(local_pts, cam_to_base @ fk[link])
            pts.append(cam_to_newton_view(cam_pts))
    if not pts:
        raise RuntimeError(f"Could not compute jaw center for regex {include_regex!r}")
    return np.vstack(pts).mean(axis=0)


def jaw_visual_clouds_newton(links, fk, cam_to_base, include_regex):
    regex = compile_re(include_regex)
    clouds = []
    for link, visuals in links.items():
        if regex is not None and regex.search(link) is None:
            continue
        if link not in fk:
            continue
        local_pts = link_visual_points(visuals)
        if len(local_pts) == 0:
            continue
        cam_pts = transform_points(local_pts, cam_to_base @ fk[link])
        clouds.append((link, local_pts, cam_to_newton_view(cam_pts)))
    if not clouds:
        raise RuntimeError(f"Could not compute jaw visual cloud for regex {include_regex!r}")
    return clouds


def choose_target_thread_index(clouds, thread_newton, target_mode):
    mode = str(target_mode).lower()
    if mode == "end0":
        return 0, float("nan")
    if mode == "end1":
        return len(thread_newton) - 1, float("nan")

    best_dist = np.inf
    best_thread_idx = 0
    candidates = [0, len(thread_newton) - 1] if mode == "nearest-end" else list(range(len(thread_newton)))
    candidate_array = np.asarray(candidates, dtype=np.int32)
    for _link, _local_pts, world_pts in clouds:
        d = np.linalg.norm(world_pts[:, None, :] - thread_newton[candidate_array[None, :], :], axis=2)
        flat_idx = int(np.argmin(d))
        dist = float(d.reshape(-1)[flat_idx])
        if dist < best_dist:
            best_dist = dist
            _point_idx, candidate_pos = np.unravel_index(flat_idx, d.shape)
            best_thread_idx = candidates[int(candidate_pos)]
    return int(best_thread_idx), float(best_dist)


def choose_grasp_points(links, fk, cam_to_base, thread_newton, include_regex, points_per_link, target_mode):
    clouds = jaw_visual_clouds_newton(links, fk, cam_to_base, include_regex)
    best_thread_idx, best_dist = choose_target_thread_index(clouds, thread_newton, target_mode)
    target = thread_newton[int(best_thread_idx)].copy()

    selected = []
    count = max(int(points_per_link), 1)
    for link, local_pts, world_pts in clouds:
        d = np.linalg.norm(world_pts - target.reshape(1, 3), axis=1)
        take = np.argsort(d)[: min(count, len(d))]
        selected.append((link, np.asarray(local_pts[take], dtype=np.float64).copy()))
    return selected, int(best_thread_idx), target, best_dist


def choose_grasp_points_old_unused(links, fk, cam_to_base, thread_newton, include_regex, points_per_link):
    clouds = jaw_visual_clouds_newton(links, fk, cam_to_base, include_regex)
    best_dist = np.inf
    best_thread_idx = 0
    for _link, _local_pts, world_pts in clouds:
        d = np.linalg.norm(world_pts[:, None, :] - thread_newton[None, :, :], axis=2)
        flat_idx = int(np.argmin(d))
        dist = float(d.reshape(-1)[flat_idx])
        if dist < best_dist:
            best_dist = dist
            _point_idx, best_thread_idx = np.unravel_index(flat_idx, d.shape)
    target = thread_newton[int(best_thread_idx)].copy()

    selected = []
    count = max(int(points_per_link), 1)
    for link, local_pts, world_pts in clouds:
        d = np.linalg.norm(world_pts - target.reshape(1, 3), axis=1)
        take = np.argsort(d)[: min(count, len(d))]
        selected.append((link, np.asarray(local_pts[take], dtype=np.float64).copy()))
    return selected, int(best_thread_idx), target, best_dist


def jaw_grasp_point_newton(selected_grasp_points, fk, cam_to_base):
    pts = []
    for link, local_pts in selected_grasp_points:
        if link not in fk:
            continue
        cam_pts = transform_points(local_pts, cam_to_base @ fk[link])
        pts.append(cam_to_newton_view(cam_pts))
    if not pts:
        raise RuntimeError("No selected jaw grasp points could be transformed")
    return np.vstack(pts).mean(axis=0)


def min_point_thread_distance(point, thread_state):
    return float(np.min(np.linalg.norm(thread_state - np.asarray(point, dtype=np.float64).reshape(1, 3), axis=1)))


def interpolate_frame_targets(frame_targets, frame_f):
    frame_f = float(np.clip(frame_f, 0.0, len(frame_targets) - 1))
    lo = int(np.floor(frame_f))
    hi = min(lo + 1, len(frame_targets) - 1)
    alpha = frame_f - lo
    return (1.0 - alpha) * frame_targets[lo] + alpha * frame_targets[hi]


def attachment_indices(target_idx, n_points, span):
    span = max(int(span), 1)
    if target_idx <= n_points // 2:
        return [i for i in range(target_idx, min(n_points, target_idx + span))]
    return [i for i in range(target_idx, max(-1, target_idx - span), -1)]


def attachment_positions(thread_initial, indices, target_idx, target):
    delta = np.asarray(target, dtype=np.float64) - thread_initial[int(target_idx)]
    out = []
    count = max(len(indices), 1)
    for k, idx in enumerate(indices):
        weight = max(0.0, 1.0 - k / float(count))
        out.append(thread_initial[int(idx)] + weight * delta)
    return np.asarray(out, dtype=np.float64)


def force_thread_attachment(state, rod_bodies, indices, positions):
    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy() if getattr(state, "body_qd", None) is not None else None
    for idx, pos in zip(indices, positions):
        body_q[int(rod_bodies[int(idx)]), :3] = np.asarray(pos, dtype=body_q.dtype)
        if body_qd is not None:
            body_qd[int(rod_bodies[int(idx)]), :] = 0.0
    state.body_q.assign(body_q)
    if body_qd is not None:
        state.body_qd.assign(body_qd)


def force_thread_positions(state, rod_bodies, positions):
    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy() if getattr(state, "body_qd", None) is not None else None
    for body_idx, pos in zip(rod_bodies, np.asarray(positions, dtype=np.float64)):
        body_q[int(body_idx), :3] = np.asarray(pos, dtype=body_q.dtype)
        if body_qd is not None:
            body_qd[int(body_idx), :] = 0.0
    state.body_q.assign(body_q)
    if body_qd is not None:
        state.body_qd.assign(body_qd)


def kinematic_drag_thread(thread_initial, target_idx, target, attachment_span, falloff_nodes):
    target_idx = int(target_idx)
    delta = np.asarray(target, dtype=np.float64) - thread_initial[target_idx]
    n = len(thread_initial)
    positions = np.asarray(thread_initial, dtype=np.float64).copy()
    falloff = max(float(falloff_nodes), 1.0)
    locked = max(int(attachment_span), 1)
    for i in range(n):
        d = abs(i - target_idx)
        if d < locked:
            weight = 1.0
        else:
            weight = float(np.exp(-(d - locked + 1) / falloff))
        positions[i] = thread_initial[i] + weight * delta
    return positions


def choose_ik_joint_names(values, requested):
    if requested:
        candidates = [name.strip() for name in requested.split(",") if name.strip()]
    else:
        candidates = [
            "yaw",
            "pitch",
            "insertion",
            "roll",
            "wrist_pitch",
            "wrist_yaw",
        ]
    return [name for name in candidates if name in values]


def solve_reach_ik(links, joints, base_values, cam_to_base, thread_newton, args):
    joint_names = choose_ik_joint_names(base_values, args.ik_joints)
    values = dict(base_values)
    fk_start = forward_kinematics(links, joints, values)
    selected_grasp_points, nearest_idx, target, start_surface_dist = choose_grasp_points(
        links,
        fk_start,
        cam_to_base,
        thread_newton,
        args.jaw_link_regex,
        args.grasp_points_per_link,
        args.target_thread,
    )
    grasp0 = jaw_grasp_point_newton(selected_grasp_points, fk_start, cam_to_base)
    # Put the actual visible jaw grasp point beside the thread centerline; the
    # jaw collision radius then overlaps the thread enough for contact on close.
    if args.reach_offset_axis == "z":
        target[2] += args.reach_offset
    elif args.reach_offset_axis == "y":
        target[1] += args.reach_offset
    elif args.reach_offset_axis == "x":
        target[0] += args.reach_offset

    last_err = np.inf
    for _ in range(args.ik_iters):
        fk = forward_kinematics(links, joints, values)
        grasp = jaw_grasp_point_newton(selected_grasp_points, fk, cam_to_base)
        err = target - grasp
        err_norm = float(np.linalg.norm(err))
        if err_norm < args.ik_tol:
            break
        if err_norm > last_err * 1.5:
            break
        last_err = err_norm

        jac = np.zeros((3, len(joint_names)), dtype=np.float64)
        for j, name in enumerate(joint_names):
            step = args.ik_fd_step
            old = float(values[name])
            values[name] = old + step
            c_plus = jaw_grasp_point_newton(
                selected_grasp_points,
                forward_kinematics(links, joints, values),
                cam_to_base,
            )
            values[name] = old - step
            c_minus = jaw_grasp_point_newton(
                selected_grasp_points,
                forward_kinematics(links, joints, values),
                cam_to_base,
            )
            values[name] = old
            jac[:, j] = (c_plus - c_minus) / (2.0 * step)

        lhs = jac.T @ jac + args.ik_damping * np.eye(len(joint_names))
        rhs = jac.T @ err
        try:
            dq = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            dq = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        dq = np.clip(dq, -args.ik_max_step, args.ik_max_step)
        for name, delta in zip(joint_names, dq):
            values[name] = float(values[name]) + float(delta)

    grasp_final = jaw_grasp_point_newton(selected_grasp_points, forward_kinematics(links, joints, values), cam_to_base)
    return values, {
        "ik_joint_names": joint_names,
        "selected_grasp_points": selected_grasp_points,
        "target_thread_idx": nearest_idx,
        "target_newton": target,
        "start_jaw_grasp_newton": grasp0,
        "final_jaw_grasp_newton": grasp_final,
        "start_surface_dist_m": start_surface_dist,
        "start_dist_m": float(np.linalg.norm(grasp0 - target)),
        "final_dist_m": float(np.linalg.norm(grasp_final - target)),
    }


def solve_ik_to_target(links, joints, initial_values, selected_grasp_points, cam_to_base, target, args):
    joint_names = choose_ik_joint_names(initial_values, args.ik_joints)
    values = dict(initial_values)
    target = np.asarray(target, dtype=np.float64).copy()
    last_err = np.inf
    for _ in range(args.ik_iters):
        fk = forward_kinematics(links, joints, values)
        grasp = jaw_grasp_point_newton(selected_grasp_points, fk, cam_to_base)
        err = target - grasp
        err_norm = float(np.linalg.norm(err))
        if err_norm < args.ik_tol:
            break
        if err_norm > last_err * 1.5:
            break
        last_err = err_norm

        jac = np.zeros((3, len(joint_names)), dtype=np.float64)
        for j, name in enumerate(joint_names):
            step = args.ik_fd_step
            old = float(values[name])
            values[name] = old + step
            c_plus = jaw_grasp_point_newton(
                selected_grasp_points,
                forward_kinematics(links, joints, values),
                cam_to_base,
            )
            values[name] = old - step
            c_minus = jaw_grasp_point_newton(
                selected_grasp_points,
                forward_kinematics(links, joints, values),
                cam_to_base,
            )
            values[name] = old
            jac[:, j] = (c_plus - c_minus) / (2.0 * step)

        lhs = jac.T @ jac + args.ik_damping * np.eye(len(joint_names))
        rhs = jac.T @ err
        try:
            dq = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            dq = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        dq = np.clip(dq, -args.ik_max_step, args.ik_max_step)
        for name, delta in zip(joint_names, dq):
            values[name] = float(values[name]) + float(delta)
    return values


def make_drag_frame_values(links, joints, base_values, reach_values, selected_grasp_points, cam_to_base, reach_target, args):
    frame_values = []
    frame_targets = []
    current_values = dict(base_values)
    drag_delta = np.asarray([args.drag_x, args.drag_y, args.drag_z], dtype=np.float64)
    for frame in range(args.frames):
        if frame < args.reach_frames:
            values = blend_values(base_values, reach_values, frame / max(args.reach_frames - 1, 1))
            target = np.asarray(reach_target, dtype=np.float64)
        else:
            drag_alpha = smoothstep((frame - args.grasp_frame) / max(args.frames - 1 - args.grasp_frame, 1))
            target = np.asarray(reach_target, dtype=np.float64) + drag_alpha * drag_delta
            values = solve_ik_to_target(links, joints, current_values, selected_grasp_points, cam_to_base, target, args)
            close_alpha = smoothstep((frame - args.grasp_frame) / max(args.frames - 1 - args.grasp_frame, 1))
            for name in list(values):
                lname = name.lower()
                if any(k in lname for k in ("jaw", "gripper", "finger", "scissor")):
                    values[name] = float((1.0 - close_alpha) * reach_values[name] + close_alpha * args.jaw_scale * base_values[name])
            current_values = dict(values)
        frame_values.append(values)
        frame_targets.append(target)
    return frame_values, np.asarray(frame_targets, dtype=np.float64)


def blend_values(base_values, target_values, alpha):
    out = dict(base_values)
    alpha = smoothstep(alpha)
    for name, value in target_values.items():
        if name in out:
            out[name] = float((1.0 - alpha) * base_values[name] + alpha * value)
    return out


def contact_motion_values(base_values, reach_values, frame, frames, reach_frames, jaw_scale):
    if frame < reach_frames:
        values = blend_values(base_values, reach_values, frame / max(reach_frames - 1, 1))
    else:
        values = dict(reach_values)
    if frame >= reach_frames:
        close_alpha = smoothstep((frame - reach_frames) / max(frames - 1 - reach_frames, 1))
        for name in list(values):
            lname = name.lower()
            if any(k in lname for k in ("jaw", "gripper", "finger", "scissor")):
                values[name] = float((1.0 - close_alpha) * reach_values[name] + close_alpha * jaw_scale * base_values[name])
    return values


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
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--dt", type=float, default=1.0 / 1200.0)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--thread-sides", type=int, default=32)
    parser.add_argument("--thread-radius-m", type=float, default=-1.0)
    parser.add_argument("--jaw-scale", type=float, default=0.05)
    parser.add_argument("--wrist-yaw-delta", type=float, default=0.35)
    parser.add_argument("--wrist-pitch-delta", type=float, default=0.25)
    parser.add_argument("--insertion-delta", type=float, default=0.0)
    parser.add_argument("--reach-frames", type=int, default=8)
    parser.add_argument("--ik-joints", default="yaw,pitch,insertion,roll,wrist_pitch,wrist_yaw")
    parser.add_argument("--ik-iters", type=int, default=40)
    parser.add_argument("--ik-damping", type=float, default=1.0e-5)
    parser.add_argument("--ik-fd-step", type=float, default=1.0e-4)
    parser.add_argument("--ik-max-step", type=float, default=0.08)
    parser.add_argument("--ik-tol", type=float, default=0.001)
    parser.add_argument("--target-thread", choices=("nearest", "nearest-end", "end0", "end1"), default="nearest")
    parser.add_argument("--grasp-points-per-link", type=int, default=16)
    parser.add_argument("--grasp-frame", type=int, default=9)
    parser.add_argument("--physics-start-frame", type=int, default=-1)
    parser.add_argument("--hard-grasp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--kinematic-drag", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attachment-span", type=int, default=4)
    parser.add_argument("--drag-falloff-nodes", type=float, default=14.0)
    parser.add_argument("--drag-x", type=float, default=0.0)
    parser.add_argument("--drag-y", type=float, default=0.0)
    parser.add_argument("--drag-z", type=float, default=0.0)
    parser.add_argument("--reach-offset", type=float, default=0.0)
    parser.add_argument("--reach-offset-axis", choices=("none", "x", "y", "z"), default="none")
    parser.add_argument("--jaw-link-regex", default="sca_ee_link_1|sca_ee_link_2|ee_link_1|ee_link_2")
    parser.add_argument("--jaw-collision-radius", type=float, default=0.0010)
    parser.add_argument("--jaw-min-half-height", type=float, default=0.0014)
    parser.add_argument("--jaw-span-scale", type=float, default=0.90)
    parser.add_argument("--show-jaw-colliders", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stretch-stiffness", type=float, default=3.0e3)
    parser.add_argument("--stretch-damping", type=float, default=2.0e1)
    parser.add_argument("--bend-stiffness", type=float, default=5.0e-1)
    parser.add_argument("--bend-damping", type=float, default=5.0e-2)
    parser.add_argument("--contact-stiffness", type=float, default=3.0e4)
    parser.add_argument("--contact-damping", type=float, default=20.0)
    parser.add_argument("--friction", type=float, default=2.0)
    parser.add_argument("--contact-buffer-size", type=int, default=8192)
    parser.add_argument("--benchmark-no-exports", action=argparse.BooleanOptionalAction, default=False)
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
    thread_cam = np.asarray(scene["states"][0], dtype=np.float64)
    radius = float(args.thread_radius_m)
    if radius <= 0.0:
        radius = float(np.asarray(scene["radius"]).reshape(-1)[0]) if "radius" in scene.files else 0.0003
    thread_newton = cam_to_newton_view(thread_cam)

    links, joints = parse_urdf(args.urdf, args.package_root)
    q_recorded = np.load(args.joints)
    jaw_recorded = np.load(args.jaw) if args.jaw else None
    base_values = joint_values_by_name(joints, q_recorded, jaw_recorded)
    fk0 = forward_kinematics(links, joints, base_values)
    ee_link = choose_ee_link(links, args.ee_link)
    cam_to_ee = np.asarray(scene["tool_cam_to_ee"], dtype=np.float64)
    cam_to_base = cam_to_ee @ np.linalg.inv(fk0[ee_link])
    reach_values, reach_report = solve_reach_ik(links, joints, base_values, cam_to_base, thread_newton, args)
    selected_grasp_points = reach_report["selected_grasp_points"]
    physics_start_frame = args.grasp_frame if args.physics_start_frame < 0 else args.physics_start_frame
    frame_value_sequence, frame_target_sequence = make_drag_frame_values(
        links,
        joints,
        base_values,
        reach_values,
        selected_grasp_points,
        cam_to_base,
        reach_report["target_newton"],
        args,
    )
    attach_indices = attachment_indices(
        int(reach_report["target_thread_idx"]),
        len(thread_newton),
        args.attachment_span,
    )
    newton_to_base = camera_to_newton_transform() @ cam_to_base
    base_pos = newton_to_base[:3, 3]
    base_quat = matrix_to_quat_xyzw(newton_to_base[:3, :3])

    temp_urdf = make_importable_urdf(
        args.urdf,
        args.package_root,
        out_dir / "psm1_si_thread_contact_motion.urdf",
        remove_collisions=True,
    )

    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_shape_cfg.ke = args.contact_stiffness
    builder.default_shape_cfg.kd = args.contact_damping
    builder.default_shape_cfg.mu = args.friction

    body_start = len(builder.body_label)
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
    body_end = len(builder.body_label)
    joint_end = len(builder.joint_label)
    imported_bodies = list(range(body_start, body_end))
    imported_joints = list(range(joint_start, joint_end))
    applied = set_imported_joint_coordinates(builder, imported_joints, base_values, 1.0e5, 1.0e3)

    body_by_link = {body_link_name(builder.body_label[i]): i for i in imported_bodies}
    jaw_re = compile_re(args.jaw_link_regex)
    jaw_cfg = newton.ModelBuilder.ShapeConfig(
        density=1000.0,
        ke=args.contact_stiffness,
        kd=args.contact_damping,
        mu=args.friction,
        margin=radius,
        gap=2.0 * radius,
        is_visible=bool(args.show_jaw_colliders),
        has_shape_collision=True,
        has_particle_collision=True,
    )
    jaw_shapes = []
    jaw_body_labels = []
    for link_name, visuals in links.items():
        if jaw_re is not None and jaw_re.search(link_name) is None:
            continue
        if link_name not in body_by_link:
            continue
        points = link_visual_points(visuals)
        if len(points) == 0:
            continue
        center, axis, half_height = capsule_from_link_visuals(
            points,
            radius=args.jaw_collision_radius,
            min_half_height=args.jaw_min_half_height,
            span_scale=args.jaw_span_scale,
        )
        shape = add_capsule_shape(
            builder,
            wp,
            body_by_link[link_name],
            center,
            axis,
            args.jaw_collision_radius,
            half_height,
            jaw_cfg,
            args.show_jaw_colliders,
            f"jaw_link_collider_{link_name}",
        )
        jaw_shapes.append(shape)
        jaw_body_labels.append(str(builder.body_label[body_by_link[link_name]]))
    if not jaw_shapes:
        raise RuntimeError("No jaw-link collision shapes were created")
    for i, a in enumerate(jaw_shapes):
        for b in jaw_shapes[i + 1 :]:
            builder.add_shape_collision_filter_pair(int(a), int(b))

    wp_points = [wp.vec3(float(x), float(y), float(z)) for x, y, z in thread_newton]
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
        label="reconstructed_thread",
    )
    if "body_frame_origin" in inspect.signature(builder.add_rod).parameters:
        rod_kwargs["body_frame_origin"] = "com"
    rod_bodies, _rod_joints = builder.add_rod(**rod_kwargs)

    builder.color()
    model = builder.finalize()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    q0 = np.asarray(model.joint_q.numpy(), dtype=np.float32)
    qd0 = np.asarray(model.joint_qd.numpy(), dtype=np.float32)
    collision_pipeline = newton.CollisionPipeline(model, contact_matching="latest")
    contacts = collision_pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=args.iterations,
        rigid_body_contact_buffer_size=args.contact_buffer_size,
        rigid_contact_history=True,
    )
    viewer = None
    if not args.benchmark_no_exports:
        viewer = make_usd_viewer(out_dir / "psm_robot_thread_contact.usd", args.fps)
        viewer.set_model(model)

    states = []
    times = []
    frame_reports = []
    loop_update_seconds = 0.0
    export_seconds = 0.0
    def apply_robot_values(frame_f):
        frame = int(np.clip(round(frame_f), 0, args.frames - 1))
        values = frame_value_sequence[frame]
        q, qd = values_to_q(builder, imported_joints, values, q0, qd0)
        # Newton eval_fk updates every articulated body in the model. Since the
        # cable rod is also represented by joints, preserve its current body
        # state while refreshing the robot articulation.
        rod_state_0 = save_body_subset(state_0, rod_bodies)
        rod_state_1 = save_body_subset(state_1, rod_bodies)
        model.joint_q.assign(q)
        model.joint_qd.assign(qd)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_1)
        restore_body_subset(state_0, rod_state_0)
        restore_body_subset(state_1, rod_state_1)
        return values

    def apply_hard_grasp(frame_f):
        if not args.hard_grasp or frame_f < float(args.grasp_frame):
            return None
        target = interpolate_frame_targets(frame_target_sequence, frame_f)
        if args.kinematic_drag:
            positions = kinematic_drag_thread(
                thread_newton,
                int(reach_report["target_thread_idx"]),
                target,
                args.attachment_span,
                args.drag_falloff_nodes,
            )
            force_thread_positions(state_0, rod_bodies, positions)
            force_thread_positions(state_1, rod_bodies, positions)
            return target
        positions = attachment_positions(
            thread_newton,
            attach_indices,
            int(reach_report["target_thread_idx"]),
            target,
        )
        force_thread_attachment(state_0, rod_bodies, attach_indices, positions)
        force_thread_attachment(state_1, rod_bodies, attach_indices, positions)
        return target

    def save_frame(frame, t, values):
        nonlocal export_seconds
        export_t0 = time.perf_counter()
        thread_state = body_positions(state_0, rod_bodies)
        states.append(thread_state)
        times.append(float(t))
        fk = forward_kinematics(links, joints, values)
        grasp_point = jaw_grasp_point_newton(selected_grasp_points, fk, cam_to_base)
        grasp_dist = min_point_thread_distance(grasp_point, thread_state)
        target_disp = float(
            np.linalg.norm(
                thread_state[int(reach_report["target_thread_idx"])] - thread_newton[int(reach_report["target_thread_idx"])]
            )
        )
        frame_reports.append((frame, float(t), grasp_dist, target_disp, grasp_point.copy()))
        if not args.benchmark_no_exports:
            write_thread_robot_obj(
                obj_dir / f"frame_{frame:06d}_psm_thread.obj",
                links,
                fk,
                cam_to_base,
                thread_state,
                radius,
                args,
            )
            viewer.begin_frame(float(t))
            viewer.log_state(state_0)
            try:
                viewer.log_contacts(contacts, state_0)
            except Exception:
                pass
            viewer.end_frame()
        export_seconds += time.perf_counter() - export_t0
        print(
            f"[{frame}] t={t:.4f}s "
            f"jaw_grasp_to_thread={grasp_dist:.6f}m "
            f"target_thread_disp={target_disp:.6f}m "
            f"thread_center={thread_state.mean(axis=0)}"
        )

    values = apply_robot_values(0.0)
    apply_hard_grasp(0.0)
    save_frame(0, 0.0, values)
    total_steps = (args.frames - 1) * args.substeps
    loop_t0 = time.perf_counter()
    for step in range(1, total_steps + 1):
        step_t0 = time.perf_counter()
        frame_f = step / float(args.substeps)
        values = apply_robot_values(frame_f)
        apply_hard_grasp(frame_f)
        if frame_f >= float(physics_start_frame) and not args.kinematic_drag:
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, args.dt)
            state_0, state_1 = state_1, state_0
            apply_hard_grasp(frame_f)
        loop_update_seconds += time.perf_counter() - step_t0
        if step % args.substeps == 0:
            save_frame(step // args.substeps, step * args.dt, values)
    loop_wall_seconds = time.perf_counter() - loop_t0

    if viewer is not None:
        viewer.close()
    states = np.asarray(states, dtype=np.float64)
    disp = np.linalg.norm(states[-1] - states[0], axis=1)
    np.savez(
        out_dir / "psm_robot_thread_contact.npz",
        times=np.asarray(times, dtype=np.float64),
        states_newton=states,
        thread_radius=np.asarray(radius),
        jaw_body_labels=np.asarray(jaw_body_labels),
    )
    report = [
        f"script_version: {SCRIPT_VERSION}",
        f"scene_npz: {args.scene_npz}",
        f"urdf: {args.urdf}",
        f"frames: {args.frames}",
        f"total_steps: {total_steps}",
        f"dt: {args.dt:.12g}",
        f"substeps: {args.substeps}",
        f"simulated_seconds: {total_steps * args.dt:.9g}",
        f"benchmark_no_exports: {args.benchmark_no_exports}",
        f"loop_wall_seconds_including_frame_diagnostics: {loop_wall_seconds:.9g}",
        f"loop_update_seconds_excluding_exports: {loop_update_seconds:.9g}",
        f"export_seconds_inside_loop: {export_seconds:.9g}",
        f"realtime_factor_update_only: {(total_steps * args.dt) / max(loop_update_seconds, 1.0e-12):.9g}",
        f"physics_start_frame: {physics_start_frame}",
        f"target_thread_mode: {args.target_thread}",
        f"hard_grasp: {args.hard_grasp}",
        f"kinematic_drag: {args.kinematic_drag}",
        f"grasp_frame: {args.grasp_frame}",
        f"attachment_span: {args.attachment_span}",
        f"drag_falloff_nodes: {args.drag_falloff_nodes:.9g}",
        f"attachment_indices: {' '.join(str(int(i)) for i in attach_indices)}",
        f"drag_delta_newton_m: [{args.drag_x:.9g} {args.drag_y:.9g} {args.drag_z:.9g}]",
        f"thread_radius_m: {radius:.9g}",
        f"thread_displacement_m_min_median_max: {disp.min():.9g}, {np.median(disp):.9g}, {disp.max():.9g}",
        f"reach_start_surface_dist_m: {reach_report['start_surface_dist_m']:.9g}",
        f"reach_start_dist_m: {reach_report['start_dist_m']:.9g}",
        f"reach_final_dist_m: {reach_report['final_dist_m']:.9g}",
        f"reach_target_thread_idx: {reach_report['target_thread_idx']}",
        f"reach_target_newton: {reach_report['target_newton']}",
        f"reach_start_jaw_grasp_newton: {reach_report['start_jaw_grasp_newton']}",
        f"reach_final_jaw_grasp_newton: {reach_report['final_jaw_grasp_newton']}",
        f"ik_joint_names: {' '.join(reach_report['ik_joint_names'])}",
        f"jaw_shapes: {' '.join(str(int(s)) for s in jaw_shapes)}",
        "jaw_bodies:",
        *[f"  {label}" for label in jaw_body_labels],
        "frame_diagnostics:",
        *[
            (
                f"  frame={frame:03d} t={t:.6f} "
                f"jaw_grasp_to_thread_m={grasp_dist:.9g} "
                f"target_thread_disp_m={target_disp:.9g} "
                f"jaw_grasp_newton={grasp_point}"
            )
            for frame, t, grasp_dist, target_disp, grasp_point in frame_reports
        ],
        "applied_recorded_joints:",
        *[f"  {name}: {value:.9g}" for _idx, name, value, *_rest in applied],
    ]
    (out_dir / "psm_robot_thread_contact_report.txt").write_text("\n".join(report) + "\n")
    print(f"USD: {out_dir / 'psm_robot_thread_contact.usd'}")
    print(f"OBJ frames: {obj_dir}")
    print(f"report: {out_dir / 'psm_robot_thread_contact_report.txt'}")


if __name__ == "__main__":
    main()
