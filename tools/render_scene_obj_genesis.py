#!/usr/bin/env python3
"""Render an exported Newton/tool OBJ scene with Genesis.

The OBJ exported by ``export_frame_scene_obj.py`` is in OpenCV camera
coordinates: x right, y down, z forward. Genesis/OpenGL rendering uses y up and
forward along -z, so this script writes a temporary converted OBJ before loading
it into Genesis.
"""

import argparse
import math
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np


def convert_obj_opencv_to_genesis(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r") as fin, dst.open("w") as fout:
        for line in fin:
            if line.startswith("mtllib "):
                fout.write(line)
            elif line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = map(float, parts[1:4])
                    fout.write(f"v {x:.9g} {-y:.9g} {-z:.9g}\n")
                else:
                    fout.write(line)
            else:
                fout.write(line)


def obj_bounds(path):
    verts = []
    with Path(path).open("r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        return np.zeros(3), np.asarray([0.1, 0.1, 0.1])
    verts = np.asarray(verts, dtype=float)
    return verts.min(axis=0), verts.max(axis=0)


def auto_camera_from_bounds(obj_path, view, distance_scale):
    low, high = obj_bounds(obj_path)
    center = 0.5 * (low + high)
    extent = high - low
    radius = max(float(np.linalg.norm(extent)) * 0.5, 0.04)
    if view == "camera":
        return np.asarray([0.0, 0.0, 0.0]), np.asarray([0.0, 0.0, -0.24])
    if view == "side":
        direction = np.asarray([1.0, -0.25, 0.35], dtype=float)
    else:
        direction = np.asarray([0.85, -1.25, 0.75], dtype=float)
    direction = direction / np.linalg.norm(direction)
    pos = center + direction * radius * distance_scale
    return pos, center


def load_genesis():
    try:
        import genesis as gs
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Genesis is not installed in this Python environment. Run this "
            "inside the genesis-tools Docker image."
        ) from exc
    return gs


def init_genesis(gs, backend):
    backend_obj = getattr(gs, backend)
    gs.init(backend=backend_obj)


def add_mesh_entity(scene, gs, obj_path):
    morph = gs.morphs.Mesh(file=str(obj_path), fixed=True)
    try:
        return scene.add_entity(morph=morph)
    except TypeError:
        return scene.add_entity(morph)


def make_scene(gs, camera_pos, camera_lookat, fov):
    sim_options = gs.options.SimOptions(dt=1.0 / 60.0)
    viewer_options = gs.options.ViewerOptions(
        camera_pos=tuple(camera_pos),
        camera_lookat=tuple(camera_lookat),
        camera_fov=float(fov),
    )
    try:
        return gs.Scene(
            sim_options=sim_options,
            viewer_options=viewer_options,
            show_viewer=False,
        )
    except TypeError:
        return gs.Scene(show_viewer=False)


def add_camera(scene, width, height, camera_pos, camera_lookat, fov):
    try:
        return scene.add_camera(
            res=(int(width), int(height)),
            pos=tuple(camera_pos),
            lookat=tuple(camera_lookat),
            fov=float(fov),
            GUI=False,
        )
    except TypeError:
        return scene.add_camera(
            res=(int(width), int(height)),
            pos=tuple(camera_pos),
            lookat=tuple(camera_lookat),
            fov=float(fov),
        )


def render_camera(camera):
    out = camera.render(rgb=True)
    if isinstance(out, tuple):
        rgb = out[0]
    else:
        rgb = out
    return np.asarray(rgb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=1025.8822021484375)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--camera-z", type=float, default=0.0)
    parser.add_argument("--lookat-z", type=float, default=-0.24)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--lookat-y", type=float, default=0.0)
    parser.add_argument("--view", choices=("oblique", "side", "camera"), default="oblique")
    parser.add_argument("--distance-scale", type=float, default=2.4)
    args = parser.parse_args()

    obj = Path(args.obj).resolve()
    if not obj.exists():
        raise FileNotFoundError(obj)

    tmp_dir = Path(tempfile.mkdtemp(prefix="genesis_scene_obj_"))
    converted = tmp_dir / obj.name
    convert_obj_opencv_to_genesis(obj, converted)
    mtl = obj.with_suffix(".mtl")
    if mtl.exists():
        shutil.copy2(mtl, tmp_dir / mtl.name)

    gs = load_genesis()
    init_genesis(gs, args.backend)

    fov = math.degrees(2.0 * math.atan(args.width / (2.0 * args.fx)))
    if args.view == "camera":
        camera_pos = np.asarray([0.0, args.camera_y, args.camera_z], dtype=float)
        camera_lookat = np.asarray([0.0, args.lookat_y, args.lookat_z], dtype=float)
    else:
        camera_pos, camera_lookat = auto_camera_from_bounds(converted, args.view, args.distance_scale)
    scene = make_scene(gs, camera_pos, camera_lookat, fov)
    add_mesh_entity(scene, gs, converted)
    camera = add_camera(scene, args.width, args.height, camera_pos, camera_lookat, fov)
    scene.build()

    rgb = render_camera(camera)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"Genesis render: {out}")
    print(f"Converted OBJ: {converted}")
    print(f"FOV degrees: {fov:.3f}")
    print(f"camera_pos: {camera_pos}")
    print(f"camera_lookat: {camera_lookat}")


if __name__ == "__main__":
    main()
