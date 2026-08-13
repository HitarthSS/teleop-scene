#!/usr/bin/env python3
"""Export a no-pad thread + PSM scene as Newton-view OBJ and USD.

The real-frame reprojection is the accuracy check. This script writes 3D
inspection artifacts in Newton's z-up viewer coordinate frame:

  - thread_robot_newton.obj/.mtl: simple mesh scene
  - thread_robot_newton.usd: Newton ViewerUSD scene
  - report.txt: camera-frame projected scale and pose diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from export_frame_scene_obj import write_obj
from render_thread_robot_newton_gl import (
    cam_to_newton_view,
    compute_camera,
    load_scene_geometry,
    write_report,
)


def write_newton_obj(path: Path, meshes):
    obj_meshes = []
    for mesh in meshes:
        material = "thread" if mesh["label"] == "thread" else "tool"
        obj_meshes.append(
            (
                mesh["name"].strip("/").replace("/", "_") or "mesh",
                material,
                cam_to_newton_view(mesh["verts_cam"]),
                mesh["faces"],
            )
        )
    write_obj(Path(path), obj_meshes)


def make_empty_newton_model():
    import newton

    builder = newton.ModelBuilder()
    model = builder.finalize()
    return model, model.state()


def make_usd_viewer(output_path: Path, fps: float):
    import newton.viewer

    try:
        return newton.viewer.ViewerUSD(output_path=str(output_path), fps=float(fps), up_axis="Z")
    except TypeError:
        try:
            return newton.viewer.ViewerUSD(output_path=str(output_path), num_frames=1)
        except TypeError:
            return newton.viewer.ViewerUSD(str(output_path))


def write_newton_usd(path: Path, meshes, fps: float, device=None):
    import warp as wp

    model, state = make_empty_newton_model()
    viewer = make_usd_viewer(Path(path), fps)
    viewer.set_model(model)
    viewer.begin_frame(0.0)
    viewer.log_state(state)
    for mesh in meshes:
        if "verts_newton" in mesh:
            verts = np.asarray(mesh["verts_newton"], dtype=np.float32)
        else:
            verts = cam_to_newton_view(mesh["verts_cam"]).astype(np.float32)
        faces = np.asarray(mesh["faces"], dtype=np.int32).reshape(-1)
        points_wp = wp.array(verts, dtype=wp.vec3, device=device)
        indices_wp = wp.array(faces, dtype=wp.int32, device=device)
        try:
            viewer.log_mesh(
                mesh["name"],
                points_wp,
                indices_wp,
                color=mesh["color"],
                roughness=mesh["roughness"],
                metallic=mesh["metallic"],
                backface_culling=False,
            )
        except TypeError:
            viewer.log_mesh(mesh["name"], points_wp, indices_wp, backface_culling=False)
    viewer.end_frame()
    viewer.close()


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
    parser.add_argument("--include-mesh-regex", default="instruments/420006")
    parser.add_argument("--exclude-mesh-regex", default=None)
    parser.add_argument("--thread-radius-m", type=float, default=None)
    parser.add_argument("--thread-diameter-m", type=float, default=None)
    parser.add_argument("--thread-sides", type=int, default=32)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fx", type=float, default=1025.8822021484375)
    parser.add_argument("--fy", type=float, default=1025.8822021484375)
    parser.add_argument("--cx", type=float, default=167.919017)
    parser.add_argument("--cy", type=float, default=234.152707)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene, thread_cam, thread_radius, ee_link, tool_instances, meshes = load_scene_geometry(args)
    if not tool_instances:
        raise RuntimeError("URDF loaded no renderable PSM STL/OBJ meshes")

    camera_info = compute_camera(meshes, "oblique", 2.2)
    report = out_dir / "thread_robot_scene_report.txt"
    write_report(report, args, scene, thread_cam, thread_radius, ee_link, tool_instances, meshes, camera_info)

    obj_path = out_dir / "thread_robot_newton.obj"
    usd_path = out_dir / "thread_robot_newton.usd"
    write_newton_obj(obj_path, meshes)
    write_newton_usd(usd_path, meshes, args.fps, args.device)

    print(f"OBJ: {obj_path}")
    print(f"USD: {usd_path}")
    print(f"report: {report}")
    print(f"tool mesh parts: {len(tool_instances)}")
    print(f"thread_diameter_m: {2.0 * thread_radius:.9g}")


if __name__ == "__main__":
    main()
