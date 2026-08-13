#!/usr/bin/env python3
"""Trim a Newton scene NPZ to frames near first thread-pad contact."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1.0e-12:
        return v
    return v / n


def pad_for_frame(scene, frame_i):
    if "pad_grid_vertices" in scene.files and np.asarray(scene["pad_grid_vertices"]).size:
        pads = np.asarray(scene["pad_grid_vertices"], dtype=np.float64)
        return pads[min(frame_i, len(pads) - 1)] if pads.ndim == 3 else pads
    return np.asarray(scene["pad_corners"], dtype=np.float64)


def signed_distance_stats(scene, frame_i):
    states = np.asarray(scene["states"], dtype=np.float64)
    pad = pad_for_frame(scene, frame_i)
    normal = normalize(np.asarray(scene["camera_to_newton_normal"], dtype=np.float64))
    pad_center = pad.mean(axis=0)
    d = (states[frame_i] - pad_center.reshape(1, 3)) @ normal
    return float(d.min()), float(d.mean()), float(d.max())


def copy_payload(scene, end_frame):
    payload = {}
    n = end_frame + 1
    for key in scene.files:
        val = scene[key]
        if key in {"states", "velocities", "newton_states", "newton_velocities", "times"}:
            payload[key] = val[:n]
        elif key == "pad_grid_vertices" and val.ndim == 3:
            payload[key] = val[:n]
        else:
            payload[key] = val
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-mean", type=float, default=0.0012)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=20)
    args = parser.parse_args()

    scene_path = Path(args.scene_npz)
    scene = np.load(scene_path, allow_pickle=True)
    states = np.asarray(scene["states"], dtype=np.float64)
    rows = []
    for i in range(len(states)):
        dmin, dmean, dmax = signed_distance_stats(scene, i)
        rows.append((i, dmin, dmean, dmax))

    candidates = [r for r in rows if r[0] + 1 >= args.min_frames]
    if args.max_frames > 0:
        candidates = [r for r in candidates if r[0] + 1 <= args.max_frames]
    if not candidates:
        candidates = rows
    best = min(candidates, key=lambda r: abs(r[2] - args.target_mean))
    end_frame = best[0]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **copy_payload(scene, end_frame))

    print(f"input: {scene_path}")
    print(f"output: {out}")
    print("frame,dmin,dmean,dmax")
    for i, dmin, dmean, dmax in rows:
        mark = " <-- selected" if i == end_frame else ""
        print(f"{i},{dmin:.9f},{dmean:.9f},{dmax:.9f}{mark}")
    print(f"selected frames: {end_frame + 1}")
    print(f"target_mean: {args.target_mean:.9f}")
    print(f"selected_mean: {best[2]:.9f}")


if __name__ == "__main__":
    main()
