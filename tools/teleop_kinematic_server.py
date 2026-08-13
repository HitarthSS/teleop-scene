#!/usr/bin/env python3
"""UDP teleop runtime for the reconstructed PSM/thread scene.

This is a real-time transport/runtime skeleton for VR control. It intentionally
does not write USD/OBJ frames. It receives target gripper commands over UDP and
streams compact JSON state messages containing thread nodes and gripper state.

Current behavior is kinematic grasp/drag: when grip is true, the selected
thread endpoint follows the commanded gripper target with a smooth falloff along
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
        "links": links,
        "joints": joints,
        "base_values": base_values,
        "current_values": dict(base_values),
        "cam_to_base": cam_to_base,
        "selected_grasp_points": selected_grasp_points,
        "target_idx": int(target_idx),
        "home_target": np.asarray(target, dtype=np.float64),
        "target": np.asarray(target, dtype=np.float64),
        "start_surface_dist": float(start_dist),
        "ik_args": ik_args,
        "ee_link": ee_link,
    }


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def apply_jaw(values, base_values, jaw_open_fraction):
    out = dict(values)
    jaw_open_fraction = clamp01(jaw_open_fraction)
    for name, base in base_values.items():
        lname = name.lower()
        if any(k in lname for k in ("jaw", "gripper", "finger", "scissor")):
            out[name] = float(base) * jaw_open_fraction
    return out


def parse_command(data):
    try:
        msg = json.loads(data.decode("utf-8"))
    except Exception as exc:
        return None, f"bad_json: {exc}"
    if not isinstance(msg, dict):
        return None, "command must be a JSON object"
    return msg, None


def update_runtime(runtime, command, args):
    if command.get("reset"):
        runtime["thread_state"] = runtime["thread_initial"].copy()
        runtime["current_values"] = dict(runtime["base_values"])
        runtime["target"] = runtime["home_target"].copy()

    if "target_newton" in command:
        runtime["target"] = np.asarray(command["target_newton"], dtype=np.float64).reshape(3)
    elif "delta_newton" in command:
        runtime["target"] = runtime["home_target"] + np.asarray(command["delta_newton"], dtype=np.float64).reshape(3)

    grip = bool(command.get("grip", False))
    jaw = clamp01(command.get("jaw", 1.0))
    if grip:
        jaw = min(jaw, args.grip_jaw_open_fraction)

    runtime["current_values"] = solve_ik_to_target(
        runtime["links"],
        runtime["joints"],
        runtime["current_values"],
        runtime["selected_grasp_points"],
        runtime["cam_to_base"],
        runtime["target"],
        runtime["ik_args"],
    )
    runtime["current_values"] = apply_jaw(runtime["current_values"], runtime["base_values"], jaw)

    if grip:
        runtime["thread_state"] = kinematic_drag_thread(
            runtime["thread_initial"],
            runtime["target_idx"],
            runtime["target"],
            args.attachment_span,
            args.drag_falloff_nodes,
        )

    return grip, jaw


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
        "grip": bool(grip),
        "jaw_open_fraction": float(jaw),
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
    parser.add_argument("--grasp-points-per-link", type=int, default=16)
    parser.add_argument("--attachment-span", type=int, default=5)
    parser.add_argument("--drag-falloff-nodes", type=float, default=16.0)
    parser.add_argument("--grip-jaw-open-fraction", type=float, default=0.02)
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
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_host, int(args.command_port)))
    sock.setblocking(False)

    dt = 1.0 / max(float(args.rate_hz), 1.0)
    seq = 0
    grip = False
    jaw = 1.0
    client_addr = None
    last_command = {}
    start = time.perf_counter()
    next_tick = start
    print(f"teleop server listening on udp://{args.bind_host}:{args.command_port}")
    print(f"target_thread_idx={runtime['target_idx']} home_target_newton={runtime['home_target']}")
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
            client_addr = addr
            last_command.update(msg)

        now = time.perf_counter()
        if now < next_tick:
            continue

        t0 = time.perf_counter()
        grip, jaw = update_runtime(runtime, last_command, args)
        update_seconds = time.perf_counter() - t0
        seq += 1
        sim_time = now - start
        perf = {
            "update_seconds": update_seconds,
            "rate_hz": float(args.rate_hz),
        }
        if client_addr is not None:
            payload = json.dumps(state_message(runtime, seq, sim_time, grip, jaw, perf), separators=(",", ":")).encode(
                "utf-8"
            )
            sock.sendto(payload, client_addr)
        if args.print_every > 0 and seq % args.print_every == 0:
            print(
                f"[{seq}] client={client_addr} grip={grip} jaw={jaw:.3f} "
                f"target_disp={np.linalg.norm(runtime['thread_state'][runtime['target_idx']] - runtime['thread_initial'][runtime['target_idx']]):.4f}m "
                f"update={update_seconds * 1000.0:.2f}ms"
            )
        next_tick += dt
        if next_tick < time.perf_counter() - dt:
            next_tick = time.perf_counter() + dt


if __name__ == "__main__":
    main()
