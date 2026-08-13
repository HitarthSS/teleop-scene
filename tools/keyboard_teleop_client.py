#!/usr/bin/env python3
"""Keyboard teleop client for tools/teleop_kinematic_server.py.

Run this on the host computer in a normal terminal. It sends UDP commands to the
Newton teleop server and prints the latest streamed state.
"""

from __future__ import annotations

import argparse
import curses
import json
import socket
import time

import numpy as np


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def draw(stdscr, args, delta, grip, step, state, received):
    stdscr.erase()
    stdscr.addstr(0, 0, "Keyboard Teleop -> Newton Thread Server")
    stdscr.addstr(2, 0, "Controls")
    stdscr.addstr(3, 2, "a/d: x -/+        w/s: y +/-        q/e: z +/-")
    stdscr.addstr(4, 2, "[/]: select thread point      {/}: jump 5 thread points")
    stdscr.addstr(5, 2, "space: close/open jaw; attaches only if thread is between jaws")
    stdscr.addstr(6, 2, "r: reset/recenter delta        +/-: step size        x: zero delta        Esc: quit")
    stdscr.addstr(8, 0, f"server: {args.server_host}:{args.server_port}")
    stdscr.addstr(9, 0, f"delta_newton [m]: [{delta[0]: .5f}, {delta[1]: .5f}, {delta[2]: .5f}]")
    stdscr.addstr(10, 0, f"grip: {grip}   jaw: {0.02 if grip else 1.0:.2f}   step: {step:.5f} m")
    stdscr.addstr(11, 0, f"received packets: {received}")
    if state:
        perf = state.get("perf", {})
        nodes = len(state.get("thread_nodes_newton", []))
        idx = state.get("target_thread_idx")
        stdscr.addstr(13, 0, f"server seq: {state.get('seq')}   selected thread idx: {idx}/{max(nodes - 1, 0)}")
        candidate = state.get("grasp_candidate_distance_m", float("inf"))
        gate = state.get("grasp_gate_radius_m", 0.0)
        stdscr.addstr(14, 0, f"attached: {state.get('attached', False)}   target displacement: {state.get('target_displacement_m', 0.0):.5f} m")
        stdscr.addstr(15, 0, f"grasp candidate distance/gate: {candidate:.5f} / {gate:.5f} m")
        stdscr.addstr(16, 0, f"server update: {1000.0 * perf.get('update_seconds', 0.0):.3f} ms")
        stdscr.addstr(17, 0, f"target_newton: {state.get('target_newton')}")
        stdscr.addstr(18, 0, f"jaw_grasp_newton: {state.get('jaw_grasp_newton')}")
        stdscr.addstr(19, 0, f"last attach distance: {state.get('attach_distance_m', 0.0):.5f} m")
    else:
        stdscr.addstr(12, 0, "No server state received yet.")
    stdscr.refresh()


def run(stdscr, args):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    server = (args.server_host, int(args.server_port))

    delta = np.zeros(3, dtype=np.float64)
    step = float(args.step)
    grip = False
    attempt_grasp = False
    select_index_delta = 0
    seq = 0
    state = None
    received = 0
    dt = 1.0 / max(float(args.rate_hz), 1.0)
    last_draw = 0.0

    while True:
        loop_t0 = time.perf_counter()
        key = stdscr.getch()
        while key != -1:
            if key in (27, ord("Q")):
                return
            if key == ord(" "):
                grip = not grip
                if grip:
                    attempt_grasp = True
            elif key == ord("["):
                select_index_delta -= 1
                delta[:] = 0.0
            elif key == ord("]"):
                select_index_delta += 1
                delta[:] = 0.0
            elif key == ord("{"):
                select_index_delta -= 5
                delta[:] = 0.0
            elif key == ord("}"):
                select_index_delta += 5
                delta[:] = 0.0
            elif key in (ord("+"), ord("=")):
                step = min(step * 2.0, args.max_step)
            elif key in (ord("-"), ord("_")):
                step = max(step * 0.5, args.min_step)
            elif key == ord("x"):
                delta[:] = 0.0
            elif key == ord("r"):
                delta[:] = 0.0
                msg = {"type": "teleop", "seq": int(seq), "reset": True}
                sock.sendto(json.dumps(msg, separators=(",", ":")).encode("utf-8"), server)
            elif key == ord("a"):
                delta[0] -= step
            elif key == ord("d"):
                delta[0] += step
            elif key == ord("w"):
                delta[1] += step
            elif key == ord("s"):
                delta[1] -= step
            elif key == ord("q"):
                delta[2] += step
            elif key == ord("e"):
                delta[2] -= step
            delta[:] = np.clip(delta, -float(args.max_abs_delta), float(args.max_abs_delta))
            key = stdscr.getch()

        command = {
            "type": "teleop",
            "seq": int(seq),
            "grip": bool(grip),
            "jaw": 0.02 if grip else 1.0,
            "delta_newton": delta.round(7).tolist(),
            "attempt_grasp": bool(attempt_grasp),
        }
        if select_index_delta:
            command["select_index_delta"] = int(select_index_delta)
        sock.sendto(json.dumps(command, separators=(",", ":")).encode("utf-8"), server)
        attempt_grasp = False
        select_index_delta = 0
        seq += 1

        while True:
            try:
                data, _addr = sock.recvfrom(65507)
                state = json.loads(data.decode("utf-8"))
                received += 1
            except BlockingIOError:
                break

        now = time.perf_counter()
        if now - last_draw > 1.0 / max(float(args.draw_hz), 1.0):
            draw(stdscr, args, delta, grip, step, state, received)
            last_draw = now

        sleep_time = dt - (time.perf_counter() - loop_t0)
        if sleep_time > 0.0:
            time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8765)
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--draw-hz", type=float, default=15.0)
    parser.add_argument("--step", type=float, default=0.00025)
    parser.add_argument("--min-step", type=float, default=0.00005)
    parser.add_argument("--max-step", type=float, default=0.002)
    parser.add_argument("--max-abs-delta", type=float, default=0.018)
    args = parser.parse_args()
    curses.wrapper(run, args)


if __name__ == "__main__":
    main()
