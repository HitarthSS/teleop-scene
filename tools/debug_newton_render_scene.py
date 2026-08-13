#!/usr/bin/env python3
"""Print frame-consistency diagnostics for Newton/render NPZ scenes."""

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


def signed_dist(points, plane_point, normal):
    points = np.asarray(points, dtype=np.float64)
    return (points - plane_point.reshape(1, 3)) @ normal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-npz", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    scene_path = Path(args.scene_npz)
    scene = np.load(scene_path)
    states = np.asarray(scene["states"], dtype=np.float64)
    if "pad_grid_vertices" in scene.files and np.asarray(scene["pad_grid_vertices"]).size:
        pad_grid = np.asarray(scene["pad_grid_vertices"], dtype=np.float64)
        pad = pad_grid[-1] if pad_grid.ndim == 3 else pad_grid
    else:
        pad = np.asarray(scene["pad_corners"], dtype=np.float64)
    normal = normalize(np.asarray(scene["camera_to_newton_normal"], dtype=np.float64))
    pad_center = pad.mean(axis=0)

    d0 = signed_dist(states[0], pad_center, normal)
    d1 = signed_dist(states[-1], pad_center, normal)
    thread_center0 = states[0].mean(axis=0)
    thread_center1 = states[-1].mean(axis=0)
    pad_span = pad.max(axis=0) - pad.min(axis=0)

    lines = [
        f"scene: {scene_path}",
        f"saved frames: {len(states)}",
        f"pad center camera: {pad_center}",
        f"pad normal camera: {normal}",
        f"pad span xyz camera: {pad_span}",
        f"thread center start camera: {thread_center0}",
        f"thread center end camera:   {thread_center1}",
        f"thread center delta camera: {thread_center1 - thread_center0}",
        f"signed distance start min/mean/max: {d0.min():.9f}, {d0.mean():.9f}, {d0.max():.9f}",
        f"signed distance end min/mean/max:   {d1.min():.9f}, {d1.mean():.9f}, {d1.max():.9f}",
    ]

    if "newton_states" in scene.files:
        ns = np.asarray(scene["newton_states"], dtype=np.float64)
        lines.extend(
            [
                f"newton z start min/mean/max: {ns[0,:,2].min():.9f}, {ns[0,:,2].mean():.9f}, {ns[0,:,2].max():.9f}",
                f"newton z end min/mean/max:   {ns[-1,:,2].min():.9f}, {ns[-1,:,2].mean():.9f}, {ns[-1,:,2].max():.9f}",
                f"newton center dz: {ns[-1,:,2].mean() - ns[0,:,2].mean():.9f}",
            ]
        )

    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)


if __name__ == "__main__":
    main()
