#!/usr/bin/env python3
"""Small UDP client for testing teleop_kinematic_server.py without VR."""

from __future__ import annotations

import argparse
import json
import socket
import time


def smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--drag-x", type=float, default=0.018)
    parser.add_argument("--drag-y", type=float, default=0.0)
    parser.add_argument("--drag-z", type=float, default=0.012)
    parser.add_argument("--grip-at", type=float, default=1.5)
    parser.add_argument("--release-at", type=float, default=-1.0)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.25)
    server = (args.host, int(args.port))
    dt = 1.0 / max(float(args.rate_hz), 1.0)
    start = time.perf_counter()
    seq = 0
    received = 0
    last_print = start

    while True:
        now = time.perf_counter()
        elapsed = now - start
        if elapsed > args.duration:
            break
        grip = elapsed >= args.grip_at and (args.release_at < 0.0 or elapsed < args.release_at)
        drag_alpha = smoothstep((elapsed - args.grip_at) / max(args.duration - args.grip_at, 1.0e-6)) if grip else 0.0
        msg = {
            "type": "teleop",
            "seq": seq,
            "grip": grip,
            "jaw": 0.02 if grip else 1.0,
            "delta_newton": [
                drag_alpha * args.drag_x,
                drag_alpha * args.drag_y,
                drag_alpha * args.drag_z,
            ],
        }
        sock.sendto(json.dumps(msg, separators=(",", ":")).encode("utf-8"), server)
        seq += 1

        try:
            data, _addr = sock.recvfrom(65507)
            state = json.loads(data.decode("utf-8"))
            received += 1
            if now - last_print > 0.5:
                print(
                    f"t={elapsed:.2f}s grip={state.get('grip')} "
                    f"disp={state.get('target_displacement_m', 0.0):.4f}m "
                    f"update={state.get('perf', {}).get('update_seconds', 0.0) * 1000.0:.2f}ms "
                    f"nodes={len(state.get('thread_nodes_newton', []))}"
                )
                last_print = now
        except socket.timeout:
            print("timeout waiting for server state")

        sleep_until = start + seq * dt
        time.sleep(max(0.0, sleep_until - time.perf_counter()))

    print(f"sent={seq} received={received}")


if __name__ == "__main__":
    main()
