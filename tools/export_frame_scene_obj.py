#!/usr/bin/env python3
"""Export the one-frame Newton/tool scene as renderable OBJ geometry.

This is a rendering handoff artifact: the output OBJ contains real triangle
geometry for the thread, selected PSM URDF meshes, and a background pad. Use it
with Blender, Genesis, or another renderer for actual lighting/materials.
"""

import argparse
from pathlib import Path

import numpy as np

from render_frame_scene_with_psm_urdf import (
    choose_ee_link,
    forward_kinematics,
    joint_values_by_name,
    load_mesh_instances,
    parse_urdf,
    transform_points,
)


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1.0e-12:
        return v
    return v / n


def tube_mesh(points, radius, sides):
    points = np.asarray(points, dtype=float)
    tangents = np.empty_like(points)
    tangents[0] = normalize(points[1] - points[0])
    tangents[-1] = normalize(points[-1] - points[-2])
    for i in range(1, len(points) - 1):
        tangents[i] = normalize(points[i + 1] - points[i - 1])

    ref = np.asarray([0.0, 1.0, 0.0])
    if abs(np.dot(ref, tangents[0])) > 0.9:
        ref = np.asarray([1.0, 0.0, 0.0])
    normal = normalize(ref - tangents[0] * np.dot(ref, tangents[0]))
    binormal = normalize(np.cross(tangents[0], normal))

    rings = []
    prev_normal = normal
    for tangent, center in zip(tangents, points):
        prev_normal = normalize(prev_normal - tangent * np.dot(prev_normal, tangent))
        if np.linalg.norm(prev_normal) < 1.0e-9:
            prev_normal = normalize(ref - tangent * np.dot(ref, tangent))
        binormal = normalize(np.cross(tangent, prev_normal))
        angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
        ring = center + radius * (
            np.cos(angles)[:, None] * prev_normal[None, :]
            + np.sin(angles)[:, None] * binormal[None, :]
        )
        rings.append(ring)
    verts = np.vstack(rings)

    faces = []
    for i in range(len(points) - 1):
        base = i * sides
        nxt = (i + 1) * sides
        for j in range(sides):
            a = base + j
            b = base + ((j + 1) % sides)
            c = nxt + ((j + 1) % sides)
            d = nxt + j
            faces.append([a, b, c])
            faces.append([a, c, d])

    # Cap ends.
    start_center = len(verts)
    end_center = len(verts) + 1
    verts = np.vstack([verts, points[0], points[-1]])
    for j in range(sides):
        faces.append([start_center, (j + 1) % sides, j])
        last = (len(points) - 1) * sides
        faces.append([end_center, last + j, last + ((j + 1) % sides)])
    return verts, np.asarray(faces, dtype=np.int32)


def mesh_normal(verts, faces):
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)
    normal = np.zeros(3, dtype=float)
    for f in faces:
        a, b, c = verts[f]
        normal += np.cross(b - a, c - a)
    norm = np.linalg.norm(normal)
    if norm < 1.0e-12:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return normal / norm


def thicken_surface_mesh(verts, faces, thickness, normal_override=None):
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)
    if thickness <= 0.0:
        return verts, np.vstack([faces, faces[:, ::-1]])

    if normal_override is None:
        normal = mesh_normal(verts, faces)
    else:
        normal = np.asarray(normal_override, dtype=float).reshape(3)
        normal = normal / max(np.linalg.norm(normal), 1.0e-12)
    bottom = verts - normal.reshape(1, 3) * thickness
    out_verts = np.vstack([verts, bottom])
    n = len(verts)

    out_faces = [*faces.tolist(), *((faces[:, ::-1] + n).tolist())]
    edge_counts = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((int(u), int(v))))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    for (a, b), count in edge_counts.items():
        if count != 1:
            continue
        out_faces.append([a, b, b + n])
        out_faces.append([a, b + n, a + n])
    return out_verts, np.asarray(out_faces, dtype=np.int32)


def pad_mesh(corners, thickness=0.006, normal_override=None):
    corners = np.asarray(corners, dtype=float)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return thicken_surface_mesh(corners, faces, thickness, normal_override)


def pad_mesh_from_scene(scene, thickness=0.006):
    normal = None
    if "camera_to_newton_normal" in scene.files:
        # Positive direction is the side where the thread lives in the Newton
        # plane frame. The pad volume must extend in the opposite direction.
        normal = np.asarray(scene["camera_to_newton_normal"], dtype=float)
    if "pad_grid_vertices" in scene.files and "pad_grid_faces" in scene.files:
        verts = np.asarray(scene["pad_grid_vertices"], dtype=float)
        faces = np.asarray(scene["pad_grid_faces"], dtype=np.int32)
        return thicken_surface_mesh(verts, faces, thickness, normal)
    if "pad_corners" in scene.files:
        return pad_mesh(scene["pad_corners"], thickness, normal)
    return pad_mesh(
        np.asarray(
            [
                [-0.05, -0.04, 0.18],
                [0.05, -0.04, 0.18],
                [0.05, 0.06, 0.18],
                [-0.05, 0.06, 0.18],
            ],
            dtype=float,
        ),
        thickness,
        normal,
    )


def write_mtl(path):
    path.write_text(
        "\n".join(
            [
                "newmtl thread",
                "Kd 0.86 0.82 0.74",
                "Ks 0.18 0.16 0.12",
                "Ns 32",
                "",
                "newmtl tool",
                "Kd 0.46 0.48 0.50",
                "Ks 0.72 0.72 0.70",
                "Ns 96",
                "",
                "newmtl pad",
                "Kd 0.88 0.48 0.43",
                "Ks 0.18 0.10 0.08",
                "Ns 12",
                "",
            ]
        )
        + "\n"
    )


def append_obj_mesh(lines, name, material, verts, faces, vertex_offset):
    lines.append(f"o {name}")
    lines.append(f"usemtl {material}")
    for v in verts:
        lines.append(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}")
    for f in faces:
        a, b, c = f + vertex_offset + 1
        lines.append(f"f {a} {b} {c}")
    return vertex_offset + len(verts)


def write_obj(path, meshes):
    mtl_path = path.with_suffix(".mtl")
    write_mtl(mtl_path)
    lines = [f"mtllib {mtl_path.name}"]
    offset = 0
    for name, material, verts, faces in meshes:
        offset = append_obj_mesh(lines, name, material, verts, faces, offset)
    path.write_text("\n".join(lines) + "\n")


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
    parser.add_argument("--thread-sides", type=int, default=18)
    parser.add_argument("--pad-thickness", type=float, default=0.0025)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scene = np.load(args.scene_npz)
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
    thread_v, thread_f = tube_mesh(np.asarray(scene["states"][-1], dtype=float), radius, args.thread_sides)
    pad_v, pad_f = pad_mesh_from_scene(scene, args.pad_thickness)

    meshes = [("background_pad", "pad", pad_v, pad_f), ("newton_thread", "thread", thread_v, thread_f)]
    for i, (verts, faces, transform, link, mesh_path) in enumerate(tool_instances):
        meshes.append((f"psm_{i}_{link}", "tool", transform_points(verts, transform), faces))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_obj(out, meshes)

    print(f"wrote OBJ scene: {out}")
    print(f"wrote material file: {out.with_suffix('.mtl')}")
    print(f"tool mesh parts exported: {len(tool_instances)}")
    print(f"ee_link: {ee_link}")


if __name__ == "__main__":
    main()
