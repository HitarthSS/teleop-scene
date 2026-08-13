#!/usr/bin/env python3
"""Blender script for rendering an exported Newton/tool OBJ scene.

Run with:
  blender -b --python tools/render_scene_obj_blender.py -- --obj scene.obj --out render.png
"""

import argparse
import math
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    import sys

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=1025.8822021484375)
    parser.add_argument("--fy", type=float, default=1025.8822021484375)
    parser.add_argument("--cx", type=float, default=167.919017)
    parser.add_argument("--cy", type=float, default=234.152707)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_obj(path):
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        bpy.ops.import_scene.obj(filepath=str(path))
    return list(bpy.context.selected_objects)


def set_materials():
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        name = obj.name.lower()
        if "thread" in name:
            color = (0.88, 0.84, 0.76, 1.0)
            roughness = 0.38
            metallic = 0.0
        elif "psm" in name:
            color = (0.50, 0.53, 0.55, 1.0)
            roughness = 0.22
            metallic = 0.35
        else:
            color = (0.66, 0.34, 0.33, 1.0)
            roughness = 0.72
            metallic = 0.0
        mat = bpy.data.materials.new(obj.name + "_mat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = color
            bsdf.inputs["Roughness"].default_value = roughness
            bsdf.inputs["Metallic"].default_value = metallic
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        try:
            for poly in obj.data.polygons:
                poly.use_smooth = True
        except Exception:
            pass


def look_at(obj, target):
    loc = obj.location
    direction = Vector(target) - loc
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def scene_bounds():
    points = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    if not points:
        return Vector((0, 0, 0)), 0.1
    low = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    high = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    center = 0.5 * (low + high)
    radius = max((high - low).length * 0.55, 0.03)
    return center, radius


def setup_camera(args):
    center, radius = scene_bounds()
    camera_data = bpy.data.cameras.new("camera")
    camera = bpy.data.objects.new("camera", camera_data)
    bpy.context.collection.objects.link(camera)

    # Render in camera coordinates: x right, y down, z forward. Blender camera
    # looks along local -Z, so place it in front of the scene and look at center.
    camera.location = (center.x, center.y - 0.18, max(center.z + 0.12, 0.05))
    look_at(camera, center)
    camera_data.lens_unit = "FOV"
    camera_data.angle = 2.0 * math.atan(args.width / (2.0 * args.fx))
    camera_data.clip_start = 0.001
    camera_data.clip_end = 10.0
    bpy.context.scene.camera = camera


def setup_lighting():
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = (0.025, 0.025, 0.028)

    for name, loc, power, size in [
        ("key", (-0.12, -0.25, 0.32), 500, 0.12),
        ("fill", (0.18, -0.15, 0.16), 90, 0.25),
        ("rim", (0.0, 0.15, 0.26), 180, 0.10),
    ]:
        light_data = bpy.data.lights.new(name, type="AREA")
        light_data.energy = power
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        light.location = loc
        bpy.context.collection.objects.link(light)


def setup_render(args):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(Path(args.out))


def main():
    args = parse_args()
    clear_scene()
    import_obj(Path(args.obj))
    set_materials()
    setup_lighting()
    setup_camera(args)
    setup_render(args)
    bpy.ops.render.render(write_still=True)
    print(f"rendered {args.out}")


if __name__ == "__main__":
    main()
