#!/usr/bin/env python3
"""Bridge rail-berkeley/oculus_reader to the Newton teleop UDP server.

Run this on the host machine, not inside the Newton Docker container. It reads
Quest controller transforms from OculusReader and sends JSON commands to
tools/teleop_kinematic_server.py.
"""

from __future__ import annotations

import argparse
import json
import socket
import time

import numpy as np


def parse_axis_map(axis_map):
    axes = []
    for item in axis_map.split(","):
        item = item.strip().lower()
        if not item:
            continue
        sign = -1.0 if item.startswith("-") else 1.0
        name = item[1:] if item.startswith("-") else item
        if name not in ("x", "y", "z"):
            raise ValueError(f"bad axis map item {item!r}; use x,y,z with optional '-'")
        axes.append((sign, {"x": 0, "y": 1, "z": 2}[name]))
    if len(axes) != 3:
        raise ValueError("--axis-map must contain three axes, e.g. x,z,-y")
    return axes


def remap_vector(vec, axis_map):
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    return np.asarray([sign * vec[idx] for sign, idx in axis_map], dtype=np.float64)


def transform_translation(transform):
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"expected 4x4 transform, got {transform.shape}")
    return transform[:3, 3].copy()


def button_float(buttons, names, default=0.0):
    for name in names:
        if name not in buttons:
            continue
        value = buttons[name]
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (tuple, list)):
            continue
        try:
            return float(value)
        except Exception:
            pass
    return float(default)


def button_bool(buttons, names, threshold=0.5):
    for name in names:
        if name not in buttons:
            continue
        value = buttons[name]
        if isinstance(value, bool):
            return value
        if isinstance(value, (tuple, list)):
            continue
        try:
            return float(value) >= float(threshold)
        except Exception:
            pass
    return False


def hand_keys(hand):
    if hand == "right":
        return {
            "transform": "r",
            "trigger_bool": ["RTr"],
            "trigger_float": ["rightTrig"],
            "grip_bool": ["RG"],
            "grip_float": ["rightGrip"],
            "recenter": ["A", "B", "RJ"],
        }
    return {
        "transform": "l",
        "trigger_bool": ["LTr"],
        "trigger_float": ["leftTrig"],
        "grip_bool": ["LG"],
        "grip_float": ["leftGrip"],
        "recenter": ["X", "Y", "LJ"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8765)
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--oculus-ip", default=None, help="Quest IP for wireless ADB; omit for USB.")
    parser.add_argument("--oculus-port", type=int, default=5555)
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--position-scale", type=float, default=0.10)
    parser.add_argument("--axis-map", default="x,z,-y", help="Map Oculus xyz to Newton xyz, e.g. x,z,-y.")
    parser.add_argument("--trigger-threshold", type=float, default=0.55)
    parser.add_argument("--grip-threshold", type=float, default=0.55)
    parser.add_argument("--move-requires-grip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-every", type=int, default=30)
    args = parser.parse_args()

    try:
        from oculus_reader.reader import OculusReader
    except ImportError as exc:
        raise SystemExit(
            "Could not import OculusReader. Install it on the host with:\n"
            "  python3 -m pip install git+https://github.com/rail-berkeley/oculus_reader.git\n"
            "or, for newer Quest 3 maintenance:\n"
            "  python3 -m pip install git+https://github.com/jborbik/oculus_reader.git"
        ) from exc

    axis_map = parse_axis_map(args.axis_map)
    keys = hand_keys(args.hand)
    reader = OculusReader(ip_address=args.oculus_ip, port=args.oculus_port, print_FPS=False)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.001)
    server = (args.server_host, int(args.server_port))

    dt = 1.0 / max(float(args.rate_hz), 1.0)
    seq = 0
    base_pos = None
    held_delta = np.zeros(3, dtype=np.float64)
    last_recenter = False
    received = 0

    print(f"OculusReader client sending to udp://{args.server_host}:{args.server_port}")
    print(f"hand={args.hand} axis_map={args.axis_map} position_scale={args.position_scale}")
    print("Controls: hold controller grip to move/clutch, trigger to grasp/close, A/B/joystick to recenter.")

    while True:
        loop_t0 = time.perf_counter()
        transforms, buttons = reader.get_transformations_and_buttons()
        if keys["transform"] not in transforms:
            time.sleep(dt)
            continue

        pos = transform_translation(transforms[keys["transform"]])
        recenter = button_bool(buttons, keys["recenter"], 0.5)
        if base_pos is None or (recenter and not last_recenter):
            base_pos = pos.copy()
            held_delta[:] = 0.0
            print("recentered Oculus controller origin")
        last_recenter = recenter

        trigger_value = max(
            button_float(buttons, keys["trigger_float"], 0.0),
            1.0 if button_bool(buttons, keys["trigger_bool"], args.trigger_threshold) else 0.0,
        )
        grip_value = max(
            button_float(buttons, keys["grip_float"], 0.0),
            1.0 if button_bool(buttons, keys["grip_bool"], args.grip_threshold) else 0.0,
        )
        move_enabled = grip_value >= args.grip_threshold if args.move_requires_grip else True
        if move_enabled:
            raw_delta = pos - base_pos
            held_delta = remap_vector(raw_delta, axis_map) * float(args.position_scale)

        grasp = trigger_value >= args.trigger_threshold
        command = {
            "type": "teleop",
            "seq": int(seq),
            "grip": bool(grasp),
            "jaw": 0.02 if grasp else 1.0,
            "delta_newton": held_delta.round(7).tolist(),
        }
        sock.sendto(json.dumps(command, separators=(",", ":")).encode("utf-8"), server)

        state = None
        try:
            data, _addr = sock.recvfrom(65507)
            state = json.loads(data.decode("utf-8"))
            received += 1
        except socket.timeout:
            pass

        if args.print_every > 0 and seq % args.print_every == 0:
            disp = None if state is None else state.get("target_displacement_m")
            update_ms = None if state is None else 1000.0 * state.get("perf", {}).get("update_seconds", 0.0)
            print(
                f"[{seq}] move={move_enabled} grasp={grasp} "
                f"trigger={trigger_value:.2f} grip={grip_value:.2f} "
                f"delta={held_delta.round(4).tolist()} disp={disp} update_ms={update_ms}"
            )

        seq += 1
        sleep_time = dt - (time.perf_counter() - loop_t0)
        if sleep_time > 0.0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
