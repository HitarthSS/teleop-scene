#!/usr/bin/env python3
"""Render a directory of OBJ frames with Genesis and optionally write a GIF."""

import argparse
import math
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from render_scene_obj_genesis import (
    add_camera,
    add_mesh_entity,
    auto_camera_from_bounds,
    convert_obj_opencv_to_genesis,
    init_genesis,
    load_genesis,
    make_scene,
    render_camera,
)


def list_obj_frames(obj_dir):
    frames = sorted(Path(obj_dir).glob("frame_*.obj"))
    if not frames:
        raise FileNotFoundError(f"no frame_*.obj files found under {obj_dir}")
    return frames


def write_gif(png_paths, gif_path, fps):
    try:
        from PIL import Image
    except Exception as exc:
        print(f"GIF skipped: Pillow unavailable ({exc})")
        return
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in png_paths]
    if not images:
        return
    duration_ms = int(round(1000.0 / max(fps, 1.0)))
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"GIF: {gif_path}")


def converted_obj_bounds_from_source(obj_path):
    verts = []
    with Path(obj_path).open("r") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, z = map(float, parts[1:4])
            verts.append([x, -y, -z])
    if not verts:
        return np.zeros(3), np.zeros(3)
    verts = np.asarray(verts, dtype=float)
    return verts.min(axis=0), verts.max(axis=0)


def converted_object_vertices_from_source(obj_path):
    objects = {}
    current = None
    with Path(obj_path).open("r") as f:
        for line in f:
            if line.startswith("o "):
                current = line.split(maxsplit=1)[1].strip()
                objects.setdefault(current, [])
            elif line.startswith("v ") and current is not None:
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = map(float, parts[1:4])
                    objects[current].append([x, -y, -z])
    return {
        name: np.asarray(verts, dtype=float)
        for name, verts in objects.items()
        if len(verts)
    }


def union_bounds(frames):
    lows = []
    highs = []
    for frame in frames:
        low, high = converted_obj_bounds_from_source(frame)
        lows.append(low)
        highs.append(high)
    return np.min(np.asarray(lows), axis=0), np.max(np.asarray(highs), axis=0)


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def infer_pad_bird_camera(frames, low, high, distance_scale):
    if not frames:
        return auto_camera_from_union_bounds(low, high, "oblique", distance_scale)

    # Use only the first frame to decide which side of the pad is "above".
    # Later cloth deformation/bounce should not flip the camera underneath.
    objects = converted_object_vertices_from_source(frames[0])
    if "background_pad" not in objects or "newton_thread" not in objects:
        return auto_camera_from_union_bounds(low, high, "oblique", distance_scale)

    pad = objects["background_pad"]
    thread = objects["newton_thread"]
    pad_center = pad.mean(axis=0)
    thread_center = thread.mean(axis=0)
    _, _, vh = np.linalg.svd(pad - pad_center, full_matrices=False)
    normal = normalize(vh[-1])
    if np.dot(normal, thread_center - pad_center) < 0.0:
        normal = -normal

    target = 0.65 * pad_center + 0.35 * thread_center
    extent = high - low
    radius = max(float(np.linalg.norm(extent)) * 0.5, 0.04)
    tangent = normalize(vh[0] - normal * np.dot(vh[0], normal))
    if np.linalg.norm(tangent) < 1e-8:
        tangent = normalize(vh[1] - normal * np.dot(vh[1], normal))
    tangent = normalize(tangent)

    # Bird's-eye from the thread side of the pad, with a small pad-axis offset.
    # This keeps "falling onto the pad" visually consistent and avoids viewing
    # the robot/tool from the underside.
    direction = normalize(3.0 * normal + 0.35 * tangent)
    return target + direction * radius * distance_scale, target


def auto_camera_from_union_bounds(low, high, view, distance_scale):
    center = 0.5 * (low + high)
    extent = high - low
    radius = max(float(np.linalg.norm(extent)) * 0.5, 0.04)
    if view == "camera":
        return np.asarray([0.0, 0.0, 0.0]), np.asarray([0.0, 0.0, -0.24])
    if view == "side":
        direction = np.asarray([1.0, -0.25, 0.35], dtype=float)
    elif view in ("thread-side", "pad-bird"):
        return center + np.asarray([0.0, 0.0, 1.0]) * radius * distance_scale, center
    else:
        direction = np.asarray([0.85, -1.25, 0.75], dtype=float)
    direction = direction / np.linalg.norm(direction)
    return center + direction * radius * distance_scale, center


def render_one(gs, obj_path, out_path, args, fov, fixed_camera=None):
    tmp_dir = Path(tempfile.mkdtemp(prefix="genesis_scene_obj_frame_"))
    converted = tmp_dir / obj_path.name
    convert_obj_opencv_to_genesis(obj_path, converted)
    mtl = obj_path.with_suffix(".mtl")
    if mtl.exists():
        shutil.copy2(mtl, tmp_dir / mtl.name)

    if fixed_camera is not None:
        camera_pos, camera_lookat = fixed_camera
    elif args.view == "camera":
        camera_pos = np.asarray([0.0, args.camera_y, args.camera_z], dtype=float)
        camera_lookat = np.asarray([0.0, args.lookat_y, args.lookat_z], dtype=float)
    else:
        camera_pos, camera_lookat = auto_camera_from_bounds(converted, args.view, args.distance_scale)
    scene = make_scene(gs, camera_pos, camera_lookat, fov)
    add_mesh_entity(scene, gs, converted)
    camera = add_camera(scene, args.width, args.height, camera_pos, camera_lookat, fov)
    scene.build()
    rgb = render_camera(camera)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gif", default=None)
    parser.add_argument("--gif-fps", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=1025.8822021484375)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--camera-z", type=float, default=0.0)
    parser.add_argument("--lookat-z", type=float, default=-0.24)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--lookat-y", type=float, default=0.0)
    parser.add_argument("--view", choices=("oblique", "side", "camera", "thread-side", "pad-bird"), default="oblique")
    parser.add_argument("--distance-scale", type=float, default=2.4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--camera-fit",
        choices=("union", "per-frame"),
        default="union",
        help="use one fixed camera for the whole sequence, or refit every frame",
    )
    args = parser.parse_args()

    frames = list_obj_frames(args.obj_dir)
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gs = load_genesis()
    init_genesis(gs, args.backend)
    fov = math.degrees(2.0 * math.atan(args.width / (2.0 * args.fx)))
    fixed_camera = None
    if args.camera_fit == "union":
        low, high = union_bounds(frames)
        if args.view in ("thread-side", "pad-bird"):
            fixed_camera = infer_pad_bird_camera(frames, low, high, args.distance_scale)
        else:
            fixed_camera = auto_camera_from_union_bounds(low, high, args.view, args.distance_scale)
        print(f"union_bounds_low: {low}")
        print(f"union_bounds_high: {high}")
        print(f"fixed_camera_pos: {fixed_camera[0]}")
        print(f"fixed_camera_lookat: {fixed_camera[1]}")
    png_paths = []
    for i, obj_path in enumerate(frames):
        out_path = out_dir / f"frame_{i:06d}.png"
        render_one(gs, obj_path, out_path, args, fov, fixed_camera=fixed_camera)
        png_paths.append(out_path)
        print(f"[{i}] {out_path}")

    if args.gif:
        write_gif(png_paths, Path(args.gif), args.gif_fps)
    print(f"rendered {len(png_paths)} Genesis frame(s) to {out_dir}")


if __name__ == "__main__":
    main()
