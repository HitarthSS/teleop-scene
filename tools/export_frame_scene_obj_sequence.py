#!/usr/bin/env python3
"""Export every saved Newton state in a scene NPZ as an OBJ sequence."""

import argparse
from pathlib import Path

import numpy as np

from export_frame_scene_obj import (
    choose_ee_link,
    forward_kinematics,
    joint_values_by_name,
    load_mesh_instances,
    pad_mesh_from_scene,
    parse_urdf,
    thicken_surface_mesh,
    transform_points,
    tube_mesh,
    write_obj,
)


def pad_mesh_for_frame(scene, frame_index, thickness):
    if "pad_grid_vertices" not in scene.files or "pad_grid_faces" not in scene.files:
        return pad_mesh_from_scene(scene, thickness)
    verts = np.asarray(scene["pad_grid_vertices"], dtype=float)
    faces = np.asarray(scene["pad_grid_faces"], dtype=np.int32)
    if verts.size == 0 or faces.size == 0:
        return pad_mesh_from_scene(scene, thickness)
    if verts.ndim == 3:
        verts = verts[min(frame_index, verts.shape[0] - 1)]
    normal = None
    if "camera_to_newton_normal" in scene.files:
        normal = np.asarray(scene["camera_to_newton_normal"], dtype=float)
    return thicken_surface_mesh(verts, faces, thickness, normal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-npz", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--joints", required=True)
    parser.add_argument("--jaw", default=None)
    parser.add_argument("--package-root", default=None)
    parser.add_argument("--ee-link", default="PSM1_tool_wrist_sca_shaft_link")
    parser.add_argument("--include-link-regex", default=None)
    parser.add_argument("--exclude-link-regex", default=None)
    parser.add_argument("--include-mesh-regex", default="tool_wrist|tool_wrist_sca")
    parser.add_argument("--exclude-mesh-regex", default="tool_main_link")
    parser.add_argument("--thread-radius", type=float, default=None)
    parser.add_argument(
        "--thread-visual-offset",
        type=float,
        default=0.0,
        help="render-only offset along camera_to_newton_normal to keep thread visible",
    )
    parser.add_argument("--thread-sides", type=int, default=24)
    parser.add_argument("--pad-thickness", type=float, default=0.0025)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    scene = np.load(args.scene_npz)
    states = np.asarray(scene["states"], dtype=float)
    if states.ndim != 3 or states.shape[-1] != 3:
        raise ValueError(f"expected states with shape (F, N, 3), got {states.shape}")

    links, joints = parse_urdf(args.urdf, args.package_root)
    q = np.load(args.joints)
    jaw = np.load(args.jaw) if args.jaw else None
    values = joint_values_by_name(joints, q, jaw)
    fk = forward_kinematics(links, joints, values)
    ee_link = choose_ee_link(links, args.ee_link)
    if ee_link not in fk:
        raise ValueError(f"EE link {ee_link!r} was not reachable in URDF FK")

    cam_to_ee = np.asarray(scene["tool_cam_to_ee"], dtype=float)
    cam_to_base = cam_to_ee @ np.linalg.inv(fk[ee_link])
    tool_instances = load_mesh_instances(
        links,
        fk,
        cam_to_base,
        include_link=args.include_link_regex,
        exclude_link=args.exclude_link_regex,
        include_mesh=args.include_mesh_regex,
        exclude_mesh=args.exclude_mesh_regex,
    )

    radius = args.thread_radius
    if radius is None:
        radius = float(np.asarray(scene["radius"]).reshape(-1)[0]) if "radius" in scene.files else 0.001
    tool_meshes = [
        (f"psm_{i}_{link}", "tool", transform_points(verts, transform), faces)
        for i, (verts, faces, transform, link, _mesh_path) in enumerate(tool_instances)
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = list(range(0, len(states), max(args.stride, 1)))
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]

    manifest = out_dir / "manifest.csv"
    with manifest.open("w") as f:
        f.write("frame,source_state,obj\n")
        for out_i, state_i in enumerate(frame_indices):
            thread_points = np.asarray(states[state_i], dtype=float)
            if args.thread_visual_offset and "camera_to_newton_normal" in scene.files:
                normal = np.asarray(scene["camera_to_newton_normal"], dtype=float).reshape(3)
                normal = normal / max(np.linalg.norm(normal), 1e-12)
                thread_points = thread_points + normal.reshape(1, 3) * args.thread_visual_offset
            thread_v, thread_f = tube_mesh(thread_points, radius, args.thread_sides)
            pad_v, pad_f = pad_mesh_for_frame(scene, state_i, args.pad_thickness)
            obj_path = out_dir / f"frame_{out_i:06d}.obj"
            meshes = [
                ("background_pad", "pad", pad_v, pad_f),
                ("newton_thread", "thread", thread_v, thread_f),
                *tool_meshes,
            ]
            write_obj(obj_path, meshes)
            f.write(f"{out_i},{state_i},{obj_path.name}\n")
            print(f"[{out_i}] {obj_path.name} from state {state_i}")

    print(f"wrote {len(frame_indices)} OBJ frame(s) to {out_dir}")
    print(f"tool mesh parts exported per frame: {len(tool_instances)}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
