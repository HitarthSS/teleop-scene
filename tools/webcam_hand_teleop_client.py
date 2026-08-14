#!/usr/bin/env python3
"""Webcam hand-tracking teleop client for tools/teleop_kinematic_server.py.

Run this on the host computer, not inside Docker. It uses MediaPipe Hands and
OpenCV to send the same UDP commands as the keyboard client:

  - hand motion -> delta_newton
  - thumb/index pinch -> close jaw + attempt grasp
  - open hand -> open jaw + release

This is intentionally a lightweight input layer. The Newton/server side remains
unchanged.
"""

from __future__ import annotations

import argparse
import json
import socket
import time

import numpy as np


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def landmark_xyz(hand_landmarks, idx):
    lm = hand_landmarks.landmark[idx]
    return np.asarray([lm.x, lm.y, lm.z], dtype=np.float64)


def hand_center(hand_landmarks):
    # Average wrist, index MCP, pinky MCP, and middle MCP for a stable palm point.
    ids = [0, 5, 9, 17]
    return np.mean([landmark_xyz(hand_landmarks, idx) for idx in ids], axis=0)


def pinch_distance(hand_landmarks):
    thumb = landmark_xyz(hand_landmarks, 4)
    index = landmark_xyz(hand_landmarks, 8)
    return float(np.linalg.norm(thumb[:2] - index[:2]))


def hand_width(hand_landmarks):
    index_mcp = landmark_xyz(hand_landmarks, 5)
    pinky_mcp = landmark_xyz(hand_landmarks, 17)
    return float(max(np.linalg.norm(index_mcp[:2] - pinky_mcp[:2]), 1.0e-6))


def map_hand_delta(current, home, width_now, width_home, args):
    image_delta = np.asarray(current[:2] - home[:2], dtype=np.float64)
    # Camera image y grows downward, so invert it for a more natural up/down.
    x = image_delta[0] * float(args.xy_scale)
    y = -image_delta[1] * float(args.xy_scale)

    if args.depth_mode == "hand-size":
        z = (float(width_now) - float(width_home)) * float(args.z_scale)
    else:
        z = -(float(current[2]) - float(home[2])) * float(args.z_scale)

    delta = np.asarray([x, y, z], dtype=np.float64)
    return np.clip(delta, -float(args.max_abs_delta), float(args.max_abs_delta))


def draw_overlay(cv2, frame, hand_landmarks, status, args):
    h, w = frame.shape[:2]
    lines = [
        "Webcam hand teleop",
        "pinch: close/grasp    open: release",
        "r: recenter    q/esc: quit",
        f"pinch={status['pinch_distance']:.3f} close<{args.pinch_close_threshold:.3f} open>{args.pinch_open_threshold:.3f}",
        f"holding={status['holding']} attached={status.get('attached')}",
        f"delta={np.asarray(status['delta']).round(4).tolist()}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (12, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 240, 30), 2, cv2.LINE_AA)

    if hand_landmarks is not None:
        thumb = landmark_xyz(hand_landmarks, 4)
        index = landmark_xyz(hand_landmarks, 8)
        p0 = (int(thumb[0] * w), int(thumb[1] * h))
        p1 = (int(index[0] * w), int(index[1] * h))
        color = (0, 0, 255) if status["holding"] else (0, 255, 255)
        cv2.circle(frame, p0, 8, color, -1, cv2.LINE_AA)
        cv2.circle(frame, p1, 8, color, -1, cv2.LINE_AA)
        cv2.line(frame, p0, p1, color, 2, cv2.LINE_AA)


def load_hand_tracking_modules():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "Missing OpenCV. Install on the host with:\n"
            "  python3 -m pip install opencv-python"
        ) from exc

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            "Missing MediaPipe. Install on the host with:\n"
            "  python3 -m pip install mediapipe\n"
            "If you are using Python 3.13 and MediaPipe is unavailable/broken, create a Python 3.11 env."
        ) from exc

    hands_module = getattr(getattr(mp, "solutions", None), "hands", None)
    if hands_module is None:
        try:
            from mediapipe.python.solutions import hands as hands_module
        except Exception as exc:
            raise SystemExit(
                "Installed mediapipe does not expose the legacy Hands API.\n"
                "Recommended fix on the host:\n"
                "  conda create -n teleop-hand python=3.11 -y\n"
                "  conda activate teleop-hand\n"
                "  python -m pip install opencv-python mediapipe\n"
                "Then rerun:\n"
                "  bash scripts/run_webcam_hand_teleop_client.sh"
            ) from exc
    return cv2, hands_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8765)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--xy-scale", type=float, default=0.055)
    parser.add_argument("--z-scale", type=float, default=0.10)
    parser.add_argument("--max-abs-delta", type=float, default=0.025)
    parser.add_argument("--depth-mode", choices=("hand-size", "mediapipe-z"), default="hand-size")
    parser.add_argument("--pinch-close-threshold", type=float, default=0.045)
    parser.add_argument("--pinch-open-threshold", type=float, default=0.075)
    parser.add_argument("--min-detection-confidence", type=float, default=0.65)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.65)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--print-every", type=int, default=30)
    args = parser.parse_args()

    cv2, mp_hands = load_hand_tracking_modules()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.001)
    server = (args.server_host, int(args.server_port))

    cap = cv2.VideoCapture(int(args.camera_index))
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam index {args.camera_index}")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=float(args.min_detection_confidence),
        min_tracking_confidence=float(args.min_tracking_confidence),
    )

    dt = 1.0 / max(float(args.rate_hz), 1.0)
    seq = 0
    home = None
    width_home = None
    delta = np.zeros(3, dtype=np.float64)
    holding = False
    attempt_grasp = False
    last_state = None

    print(f"Webcam hand client sending to udp://{args.server_host}:{args.server_port}")
    print("Controls: pinch thumb+index to close/grasp, open to release, r to recenter, q/esc to quit.")

    while True:
        loop_t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            time.sleep(dt)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        hand_landmarks = result.multi_hand_landmarks[0] if result.multi_hand_landmarks else None

        if hand_landmarks is not None:
            center = hand_center(hand_landmarks)
            width_now = hand_width(hand_landmarks)
            pinch = pinch_distance(hand_landmarks)
            if home is None:
                home = center.copy()
                width_home = width_now

            delta = map_hand_delta(center, home, width_now, width_home, args)

            if holding:
                if pinch >= float(args.pinch_open_threshold):
                    holding = False
            elif pinch <= float(args.pinch_close_threshold):
                holding = True
                attempt_grasp = True
        else:
            pinch = float("inf")
            holding = False

        command = {
            "type": "teleop",
            "seq": int(seq),
            "grip": bool(holding),
            "jaw": 0.0 if holding else 1.0,
            "delta_newton": delta.round(7).tolist(),
            "attempt_grasp": bool(attempt_grasp),
        }
        sock.sendto(json.dumps(command, separators=(",", ":")).encode("utf-8"), server)
        attempt_grasp = False

        try:
            data, _addr = sock.recvfrom(65507)
            last_state = json.loads(data.decode("utf-8"))
        except socket.timeout:
            pass

        key = -1
        if not args.no_preview:
            status = {
                "pinch_distance": pinch,
                "holding": holding,
                "attached": None if last_state is None else last_state.get("attached"),
                "delta": delta,
            }
            draw_overlay(cv2, frame, hand_landmarks, status, args)
            cv2.imshow("webcam hand teleop", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                home = None
                width_home = None
                delta[:] = 0.0
                print("recenter requested")

        if args.print_every > 0 and seq % args.print_every == 0:
            attached = None if last_state is None else last_state.get("attached")
            print(f"[{seq}] holding={holding} attached={attached} pinch={pinch:.3f} delta={delta.round(4).tolist()}")

        seq += 1
        sleep_time = dt - (time.perf_counter() - loop_t0)
        if sleep_time > 0.0:
            time.sleep(sleep_time)

    cap.release()
    if not args.no_preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
