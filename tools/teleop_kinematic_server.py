#!/usr/bin/env python3
"""UDP teleop runtime for the reconstructed PSM/thread scene.

This is a real-time transport/runtime skeleton for VR control. It intentionally
does not write USD/OBJ frames. It receives target gripper commands over UDP and
streams compact JSON state messages containing thread nodes and gripper state.

Current behavior is kinematic grasp/drag: when grip is true, the selected
thread point follows the commanded gripper target with a smooth falloff along
the cable. This is the stable v0 path for Unity/Quest integration; later it can
be swapped for a true Newton contact/constraint grasp.
"""

from __future__ import annotations

import argparse
import json
import select
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from animate_psm_robot_with_thread_contact import (
    choose_grasp_points,
    jaw_grasp_point_newton,
    jaw_visual_clouds_newton,
    kinematic_drag_thread,
    solve_ik_to_target,
)
from render_frame_scene_with_psm_urdf import (
    choose_ee_link,
    forward_kinematics,
    joint_values_by_name,
    parse_urdf,
)
from render_thread_robot_newton_gl import cam_to_newton_view


ONE_SHOT_COMMAND_KEYS = {
    "reset",
    "attempt_grasp",
    "snap_grip",
    "snap_selected",
    "target_thread_idx",
    "select_index_delta",
}


def load_scene(args):
    scene = np.load(args.scene_npz, allow_pickle=True)
    thread_cam = np.asarray(scene["states"][args.state_index], dtype=np.float64)
    thread_newton = cam_to_newton_view(thread_cam)

    links, joints = parse_urdf(args.urdf, args.package_root)
    q_recorded = np.load(args.joints)
    jaw_recorded = np.load(args.jaw) if args.jaw else None
    base_values = joint_values_by_name(joints, q_recorded, jaw_recorded)
    fk0 = forward_kinematics(links, joints, base_values)
    ee_link = choose_ee_link(links, args.ee_link)
    cam_to_ee = np.asarray(scene["tool_cam_to_ee"], dtype=np.float64)
    cam_to_base = cam_to_ee @ np.linalg.inv(fk0[ee_link])

    selected_grasp_points, target_idx, target, start_dist = choose_grasp_points(
        links,
        fk0,
        cam_to_base,
        thread_newton,
        args.jaw_link_regex,
        args.grasp_points_per_link,
        args.target_thread,
    )

    ik_args = SimpleNamespace(
        ik_joints=args.ik_joints,
        ik_iters=args.ik_iters,
        ik_damping=args.ik_damping,
        ik_fd_step=args.ik_fd_step,
        ik_max_step=args.ik_max_step,
        ik_tol=args.ik_tol,
    )

    return {
        "scene": scene,
        "thread_initial": thread_newton,
        "thread_state": thread_newton.copy(),
        "drag_reference": thread_newton.copy(),
        "links": links,
        "joints": joints,
        "base_values": base_values,
        "current_values": dict(base_values),
        "cam_to_base": cam_to_base,
        "selected_grasp_points": selected_grasp_points,
        "target_idx": int(target_idx),
        "home_target": np.asarray(target, dtype=np.float64),
        "target": np.asarray(target, dtype=np.float64),
        "attached": False,
        "attach_target_reference": np.asarray(target, dtype=np.float64),
        "attach_jaw_reference": np.asarray(target, dtype=np.float64),
        "attach_thread_reference": np.asarray(target, dtype=np.float64),
        "attach_driver_newton": np.asarray(target, dtype=np.float64),
        "snap_distance": 0.0,
        "grasp_candidate_distance": float("inf"),
        "grasp_candidate_aperture_t": 0.0,
        "start_surface_dist": float(start_dist),
        "ik_args": ik_args,
        "ee_link": ee_link,
    }


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def jaw_joint_names(values, args):
    markers = tuple(k.strip().lower() for k in str(args.jaw_joint_markers).split(",") if k.strip())
    out = []
    for name in values:
        lname = name.lower()
        if any(marker in lname for marker in markers):
            out.append(name)
    return out


def jaw_joint_signs(names, base_values, args):
    mode = str(args.jaw_sign_mode).lower()
    if mode == "same":
        return [1.0 for _name in names]
    if mode == "explicit":
        signs = [float(v.strip()) for v in str(args.jaw_signs).split(",") if v.strip()]
        if len(signs) != len(names):
            raise ValueError(f"--jaw-signs needs {len(names)} values for joints {names}; got {signs}")
        return signs

    signs = []
    side = 1.0
    for name in names:
        base = float(base_values.get(name, 0.0))
        if mode == "base" and abs(base) > 1.0e-6:
            signs.append(1.0 if base >= 0.0 else -1.0)
        else:
            signs.append(side)
        side *= -1.0
    return signs


def apply_jaw(values, base_values, jaw_open_fraction, args):
    out = dict(values)
    jaw_open_fraction = clamp01(jaw_open_fraction)
    open_angle = float(args.jaw_open_angle)
    closed_angle = float(args.jaw_closed_angle)
    angle = closed_angle + jaw_open_fraction * (open_angle - closed_angle)
    names = jaw_joint_names(out, args)
    for name, sign in zip(names, jaw_joint_signs(names, base_values, args)):
        out[name] = float(sign * angle)
    return out


def parse_command(data):
    try:
        msg = json.loads(data.decode("utf-8"))
    except Exception as exc:
        return None, f"bad_json: {exc}"
    if not isinstance(msg, dict):
        return None, "command must be a JSON object"
    return msg, None


def nearest_thread_index(thread_state, query):
    thread_state = np.asarray(thread_state, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64).reshape(1, 3)
    return int(np.argmin(np.linalg.norm(thread_state - query, axis=1)))


def closest_point_on_segment(point, a, b):
    point = np.asarray(point, dtype=np.float64).reshape(3)
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1.0e-14:
        return a.copy(), 0.0
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    return a + t * ab, t


def segment_parameter(point, a, b):
    point = np.asarray(point, dtype=np.float64).reshape(3)
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1.0e-14:
        return 0.0
    return float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))


def closest_segment_points(a0, a1, b0, b1):
    """Return closest points between two 3-D segments and their distance."""
    a0 = np.asarray(a0, dtype=np.float64).reshape(3)
    a1 = np.asarray(a1, dtype=np.float64).reshape(3)
    b0 = np.asarray(b0, dtype=np.float64).reshape(3)
    b1 = np.asarray(b1, dtype=np.float64).reshape(3)
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    aa = float(np.dot(u, u))
    bb = float(np.dot(v, v))
    ab = float(np.dot(u, v))
    aw = float(np.dot(u, w))
    bw = float(np.dot(v, w))
    denom = aa * bb - ab * ab

    if aa < 1.0e-14 and bb < 1.0e-14:
        pa, pb = a0, b0
    elif aa < 1.0e-14:
        pb, _ = closest_point_on_segment(a0, b0, b1)
        pa = a0
    elif bb < 1.0e-14:
        pa, _ = closest_point_on_segment(b0, a0, a1)
        pb = b0
    else:
        s = 0.0 if abs(denom) < 1.0e-14 else np.clip((ab * bw - bb * aw) / denom, 0.0, 1.0)
        t = np.clip((ab * s + bw) / bb, 0.0, 1.0)
        s = np.clip((ab * t - aw) / aa, 0.0, 1.0)
        pa = a0 + s * u
        pb = b0 + t * v
    return pa, pb, float(np.linalg.norm(pa - pb))


def jaw_aperture_segment(runtime, values):
    fk = forward_kinematics(runtime["links"], runtime["joints"], values)
    clouds = jaw_visual_clouds_newton(
        runtime["links"],
        fk,
        runtime["cam_to_base"],
        runtime["jaw_link_regex"],
    )
    target = np.asarray(runtime["target"], dtype=np.float64).reshape(1, 3)
    jaw_points = []
    for link, _local_pts, world_pts in clouds:
        world_pts = np.asarray(world_pts, dtype=np.float64)
        if len(world_pts) == 0:
            continue
        idx = int(np.argmin(np.linalg.norm(world_pts - target, axis=1)))
        jaw_points.append((link, world_pts[idx]))
    if len(jaw_points) < 2:
        point = jaw_grasp_point_newton(runtime["selected_grasp_points"], fk, runtime["cam_to_base"])
        return point, point
    jaw_points = sorted(jaw_points, key=lambda item: item[0])[:2]
    return np.asarray(jaw_points[0][1], dtype=np.float64), np.asarray(jaw_points[1][1], dtype=np.float64)


def jaw_driver_point(runtime, values):
    aperture_a, aperture_b = jaw_aperture_segment(runtime, values)
    return 0.5 * (np.asarray(aperture_a, dtype=np.float64) + np.asarray(aperture_b, dtype=np.float64))


def nearest_thread_segment_to_aperture(thread_state, aperture_a, aperture_b):
    thread_state = np.asarray(thread_state, dtype=np.float64)
    best = {
        "distance": float("inf"),
        "raw_distance": float("inf"),
        "index": 0,
        "aperture_t": 0.0,
        "thread_t": 0.0,
        "thread_point": thread_state[0].copy(),
        "jaw_point": np.asarray(aperture_a, dtype=np.float64).copy(),
    }
    for i in range(len(thread_state) - 1):
        thread_point, jaw_point, distance = closest_segment_points(
            thread_state[i],
            thread_state[i + 1],
            aperture_a,
            aperture_b,
        )
        aperture_t = segment_parameter(jaw_point, aperture_a, aperture_b)
        thread_t = segment_parameter(thread_point, thread_state[i], thread_state[i + 1])
        inside_margin = 0.08
        outside_penalty = max(0.0, inside_margin - aperture_t, aperture_t - (1.0 - inside_margin))
        score = float(distance + 0.01 * outside_penalty)
        if score < best["distance"]:
            nearest_node = i if np.linalg.norm(thread_state[i] - thread_point) <= np.linalg.norm(thread_state[i + 1] - thread_point) else i + 1
            best = {
                "distance": score,
                "raw_distance": float(distance),
                "index": int(nearest_node),
                "aperture_t": float(aperture_t),
                "thread_t": float(thread_t),
                "thread_point": thread_state[int(nearest_node)].copy(),
                "jaw_point": np.asarray(jaw_point, dtype=np.float64).copy(),
            }
    return best


def try_attach_thread_inside_open_jaw(runtime, values_gate, values_driver, args):
    aperture_a, aperture_b = jaw_aperture_segment(runtime, values_gate)
    hit = nearest_thread_segment_to_aperture(runtime["thread_state"], aperture_a, aperture_b)
    runtime["grasp_candidate_distance"] = float(hit["raw_distance"])
    runtime["grasp_candidate_aperture_t"] = float(hit["aperture_t"])
    latch_radius = max(float(args.grasp_gate_radius), float(args.thread_radius) + float(args.jaw_collision_radius))
    inside_opening = -float(args.grasp_aperture_margin) <= hit["aperture_t"] <= 1.0 + float(args.grasp_aperture_margin)
    if hit["raw_distance"] > latch_radius or not inside_opening:
        runtime["attached"] = False
        runtime["snap_distance"] = float(hit["raw_distance"])
        return False

    runtime["attached"] = True
    runtime["target_idx"] = int(hit["index"])
    runtime["drag_reference"] = np.asarray(runtime["thread_state"], dtype=np.float64).copy()
    runtime["attach_target_reference"] = np.asarray(runtime["target"], dtype=np.float64).copy()
    runtime["attach_jaw_reference"] = jaw_driver_point(runtime, values_driver)
    runtime["attach_thread_reference"] = runtime["drag_reference"][runtime["target_idx"]].copy()
    runtime["attach_driver_newton"] = runtime["attach_jaw_reference"].copy()
    runtime["snap_distance"] = float(hit["raw_distance"])
    return True


def snap_grasp_to_thread(runtime, query):
    idx = nearest_thread_index(runtime["thread_state"], query)
    runtime["target_idx"] = idx
    runtime["drag_reference"] = np.asarray(runtime["thread_state"], dtype=np.float64).copy()
    runtime["attach_target_reference"] = np.asarray(runtime["target"], dtype=np.float64).copy()
    runtime["attach_jaw_reference"] = np.asarray(runtime["target"], dtype=np.float64).copy()
    runtime["attach_thread_reference"] = runtime["drag_reference"][idx].copy()
    runtime["attach_driver_newton"] = runtime["attach_jaw_reference"].copy()
    runtime["snap_distance"] = float(np.linalg.norm(runtime["target"] - np.asarray(query, dtype=np.float64).reshape(3)))
    return idx


def select_thread_index(runtime, idx):
    n = len(runtime["thread_state"])
    idx = int(np.clip(int(idx), 0, n - 1))
    runtime["target_idx"] = idx
    runtime["drag_reference"] = np.asarray(runtime["thread_state"], dtype=np.float64).copy()
    runtime["home_target"] = runtime["drag_reference"][idx].copy()
    runtime["target"] = runtime["home_target"].copy()
    runtime["attached"] = False
    runtime["snap_distance"] = 0.0
    return idx


def update_runtime(runtime, command, args):
    if command.get("subscribe"):
        return bool(runtime.get("last_grip", False)), float(runtime.get("last_jaw", 1.0))

    if command.get("reset"):
        runtime["thread_state"] = runtime["thread_initial"].copy()
        runtime["drag_reference"] = runtime["thread_initial"].copy()
        runtime["current_values"] = dict(runtime["base_values"])
        runtime["home_target"] = runtime["thread_initial"][runtime["target_idx"]].copy()
        runtime["target"] = runtime["home_target"].copy()
        runtime["attached"] = False
        runtime["attach_target_reference"] = runtime["target"].copy()
        runtime["attach_jaw_reference"] = runtime["target"].copy()
        runtime["attach_thread_reference"] = runtime["thread_initial"][runtime["target_idx"]].copy()
        runtime["attach_driver_newton"] = runtime["attach_jaw_reference"].copy()
        runtime["snap_distance"] = 0.0
        runtime["grasp_candidate_distance"] = float("inf")
        runtime["grasp_candidate_aperture_t"] = 0.0

    previous_grip = bool(runtime.get("last_grip", False))
    if "target_thread_idx" in command:
        select_thread_index(runtime, command["target_thread_idx"])
    elif "select_index_delta" in command:
        select_thread_index(runtime, int(runtime["target_idx"]) + int(command["select_index_delta"]))

    if "target_newton" in command:
        runtime["target"] = np.asarray(command["target_newton"], dtype=np.float64).reshape(3)
    elif "delta_newton" in command:
        runtime["target"] = runtime["home_target"] + np.asarray(command["delta_newton"], dtype=np.float64).reshape(3)

    grip = bool(command.get("grip", False))
    if "jaw" in command:
        jaw = clamp01(command.get("jaw", 1.0))
    else:
        jaw = args.grip_jaw_open_fraction if grip else 1.0

    ik_seed_values = apply_jaw(
        runtime["current_values"],
        runtime["base_values"],
        args.grasp_gate_jaw_open_fraction,
        args,
    )
    solved_values = solve_ik_to_target(
        runtime["links"],
        runtime["joints"],
        ik_seed_values,
        runtime["selected_grasp_points"],
        runtime["cam_to_base"],
        runtime["target"],
        runtime["ik_args"],
    )
    values_gate = apply_jaw(solved_values, runtime["base_values"], args.grasp_gate_jaw_open_fraction, args)
    values_current = apply_jaw(solved_values, runtime["base_values"], jaw, args)

    if not grip:
        runtime["attached"] = False
    elif not runtime.get("attached") and (
        command.get("attempt_grasp")
        or command.get("snap_selected")
        or command.get("snap_grip")
        or (args.snap_on_grip and not previous_grip)
        or args.grasp_retry_while_holding
    ):
        try_attach_thread_inside_open_jaw(runtime, values_gate, values_current, args)

    runtime["current_values"] = values_current

    if grip and runtime.get("attached"):
        current_jaw_driver = jaw_driver_point(runtime, runtime["current_values"])
        runtime["attach_driver_newton"] = current_jaw_driver.copy()
        driver_delta = np.asarray(current_jaw_driver, dtype=np.float64) - np.asarray(
            runtime["attach_jaw_reference"], dtype=np.float64
        )
        attached_thread_target = np.asarray(runtime["attach_thread_reference"], dtype=np.float64) + driver_delta
        runtime["thread_state"] = kinematic_drag_thread(
            runtime["drag_reference"],
            runtime["target_idx"],
            attached_thread_target,
            args.attachment_span,
            args.drag_falloff_nodes,
        )

    runtime["last_grip"] = grip
    runtime["last_jaw"] = jaw
    return grip, jaw


def clear_one_shot_commands(command):
    for key in ONE_SHOT_COMMAND_KEYS:
        command.pop(key, None)


def state_message(runtime, seq, sim_time, grip, jaw, perf):
    fk = forward_kinematics(runtime["links"], runtime["joints"], runtime["current_values"])
    jaw_point = jaw_grasp_point_newton(runtime["selected_grasp_points"], fk, runtime["cam_to_base"])
    target_idx = int(runtime["target_idx"])
    target_disp = float(np.linalg.norm(runtime["thread_state"][target_idx] - runtime["thread_initial"][target_idx]))
    return {
        "type": "thread_teleop_state",
        "version": 1,
        "seq": int(seq),
        "time": float(sim_time),
        "target_thread_idx": target_idx,
        "target_newton": runtime["target"].round(7).tolist(),
        "jaw_grasp_newton": jaw_point.round(7).tolist(),
        "attach_driver_newton": np.asarray(runtime.get("attach_driver_newton", runtime["target"]), dtype=np.float64)
        .round(7)
        .tolist(),
        "attach_distance_m": float(runtime.get("snap_distance", 0.0)),
        "snap_distance_m": float(runtime.get("snap_distance", 0.0)),
        "grasp_candidate_distance_m": float(runtime.get("grasp_candidate_distance", float("inf"))),
        "grasp_candidate_aperture_t": float(runtime.get("grasp_candidate_aperture_t", 0.0)),
        "grasp_gate_radius_m": float(runtime.get("grasp_gate_radius", 0.0)),
        "attached": bool(runtime.get("attached", False)),
        "grip": bool(grip),
        "jaw_open_fraction": float(jaw),
        "jaw_open_angle": float(runtime.get("jaw_open_angle", 0.0)),
        "jaw_closed_angle": float(runtime.get("jaw_closed_angle", 0.0)),
        "jaw_joint_values": {
            name: float(runtime["current_values"].get(name, 0.0)) for name in runtime.get("jaw_joint_names", [])
        },
        "target_displacement_m": target_disp,
        "thread_nodes_newton": np.asarray(runtime["thread_state"], dtype=np.float64).round(7).tolist(),
        "joint_values": {k: float(v) for k, v in runtime["current_values"].items()},
        "perf": perf,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-npz", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--joints", required=True)
    parser.add_argument("--jaw", default=None)
    parser.add_argument("--package-root", default=None)
    parser.add_argument("--ee-link", default="PSM1_tool_wrist_sca_shaft_link")
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--jaw-link-regex", default="sca_ee_link_1|sca_ee_link_2|ee_link_1|ee_link_2")
    parser.add_argument("--target-thread", choices=("nearest", "nearest-end", "end0", "end1"), default="nearest-end")
    parser.add_argument("--snap-on-grip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grasp-points-per-link", type=int, default=16)
    parser.add_argument("--grasp-gate-radius", type=float, default=0.0015)
    parser.add_argument("--jaw-collision-radius", type=float, default=0.0012)
    parser.add_argument("--thread-radius", type=float, default=0.0003)
    parser.add_argument("--grasp-aperture-margin", type=float, default=0.15)
    parser.add_argument("--grasp-retry-while-holding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grasp-gate-jaw-open-fraction", type=float, default=1.0)
    parser.add_argument("--attachment-span", type=int, default=5)
    parser.add_argument("--drag-falloff-nodes", type=float, default=16.0)
    parser.add_argument("--grip-jaw-open-fraction", type=float, default=0.02)
    parser.add_argument("--jaw-open-angle", type=float, default=0.75)
    parser.add_argument("--jaw-closed-angle", type=float, default=0.015)
    parser.add_argument("--jaw-joint-markers", default="jaw_1,jaw_2")
    parser.add_argument("--jaw-sign-mode", choices=("same", "alternating", "base", "explicit"), default="same")
    parser.add_argument("--jaw-signs", default="")
    parser.add_argument("--ik-joints", default="yaw,pitch,insertion,roll,wrist_pitch,wrist_yaw")
    parser.add_argument("--ik-iters", type=int, default=16)
    parser.add_argument("--ik-damping", type=float, default=1.0e-5)
    parser.add_argument("--ik-fd-step", type=float, default=1.0e-4)
    parser.add_argument("--ik-max-step", type=float, default=0.04)
    parser.add_argument("--ik-tol", type=float, default=0.001)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--command-port", type=int, default=8765)
    parser.add_argument("--rate-hz", type=float, default=90.0)
    parser.add_argument("--max-packet-bytes", type=int, default=65507)
    parser.add_argument("--print-every", type=int, default=90)
    args = parser.parse_args()

    runtime = load_scene(args)
    runtime["jaw_link_regex"] = args.jaw_link_regex
    runtime["grasp_gate_radius"] = float(args.grasp_gate_radius)
    runtime["jaw_open_angle"] = float(args.jaw_open_angle)
    runtime["jaw_closed_angle"] = float(args.jaw_closed_angle)
    runtime["jaw_joint_names"] = jaw_joint_names(runtime["base_values"], args)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_host, int(args.command_port)))
    sock.setblocking(False)

    dt = 1.0 / max(float(args.rate_hz), 1.0)
    seq = 0
    grip = False
    jaw = 1.0
    clients = {}
    last_command = {}
    start = time.perf_counter()
    next_tick = start
    print(f"teleop server listening on udp://{args.bind_host}:{args.command_port}")
    print(f"target_thread_idx={runtime['target_idx']} home_target_newton={runtime['home_target']}")
    print(f"jaw_joint_names={runtime['jaw_joint_names']}")
    if not runtime["jaw_joint_names"]:
        print(f"WARNING: no jaw joints matched --jaw-joint-markers={args.jaw_joint_markers!r}")
    else:
        print(f"jaw_sign_mode={args.jaw_sign_mode} jaw_signs={jaw_joint_signs(runtime['jaw_joint_names'], runtime['base_values'], args)}")
        open_values = apply_jaw(runtime["base_values"], runtime["base_values"], 1.0, args)
        closed_values = apply_jaw(runtime["base_values"], runtime["base_values"], 0.0, args)
        print("jaw open/closed values:")
        for name in runtime["jaw_joint_names"]:
            print(f"  {name}: open={open_values[name]:.6f} closed={closed_values[name]:.6f}")
    print("send JSON commands with target_newton or delta_newton, jaw, grip")

    while True:
        now = time.perf_counter()
        timeout = max(0.0, next_tick - now)
        ready, _w, _x = select.select([sock], [], [], timeout)
        for _ in ready:
            data, addr = sock.recvfrom(args.max_packet_bytes)
            msg, error = parse_command(data)
            if error:
                print(f"ignoring command from {addr}: {error}")
                continue
            clients[addr] = time.perf_counter()
            if not msg.get("subscribe"):
                last_command.update(msg)

        now = time.perf_counter()
        if now < next_tick:
            continue

        t0 = time.perf_counter()
        grip, jaw = update_runtime(runtime, last_command, args)
        clear_one_shot_commands(last_command)
        update_seconds = time.perf_counter() - t0
        seq += 1
        sim_time = now - start
        perf = {
            "update_seconds": update_seconds,
            "rate_hz": float(args.rate_hz),
        }
        if clients:
            cutoff = time.perf_counter() - 5.0
            clients = {addr: last_seen for addr, last_seen in clients.items() if last_seen >= cutoff}
            payload = json.dumps(state_message(runtime, seq, sim_time, grip, jaw, perf), separators=(",", ":")).encode(
                "utf-8"
            )
            for addr in clients:
                sock.sendto(payload, addr)
        if args.print_every > 0 and seq % args.print_every == 0:
            print(
                f"[{seq}] clients={len(clients)} grip={grip} jaw={jaw:.3f} "
                f"target_disp={np.linalg.norm(runtime['thread_state'][runtime['target_idx']] - runtime['thread_initial'][runtime['target_idx']]):.4f}m "
                f"update={update_seconds * 1000.0:.2f}ms"
            )
        next_tick += dt
        if next_tick < time.perf_counter() - dt:
            next_tick = time.perf_counter() + dt


if __name__ == "__main__":
    main()
