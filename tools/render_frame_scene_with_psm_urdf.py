#!/usr/bin/env python3
"""Render a frame scene with Newton thread state and a PSM URDF mesh.

The script consumes ``newton_frame_scene.npz`` from
``generate_newton_frame_scene.py`` and renders:
  - synthetic or real-frame background
  - Newton thread as a camera-projected tube
  - PSM visual meshes from a URDF, placed by the pose-estimator cam_to_ee

It intentionally has no trimesh/urdfpy dependency. STL and OBJ meshes are
loaded directly; DAE files are skipped unless they are converted externally.
"""

import argparse
import math
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from generate_newton_frame_scene import draw_polyline_camera, load_rgb_image, make_synthetic_pad


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def transform_from_xyz_rpy(xyz=None, rpy=None):
    out = np.eye(4, dtype=float)
    if rpy is not None:
        out[:3, :3] = rpy_matrix(rpy)
    if xyz is not None:
        out[:3, 3] = np.asarray(xyz, dtype=float)
    return out


def axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1.0e-12)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    c1 = 1.0 - c
    return np.asarray(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def parse_vec(text, n, default):
    if text is None:
        return np.asarray(default, dtype=float)
    vals = [float(v) for v in text.split()]
    if len(vals) != n:
        raise ValueError(f"expected {n} values, got {text!r}")
    return np.asarray(vals, dtype=float)


def origin_transform(elem):
    origin = elem.find("origin")
    if origin is None:
        return np.eye(4, dtype=float)
    return transform_from_xyz_rpy(
        parse_vec(origin.attrib.get("xyz"), 3, (0, 0, 0)),
        parse_vec(origin.attrib.get("rpy"), 3, (0, 0, 0)),
    )


def resolve_mesh_path(filename, urdf_dir, package_root):
    raw = filename
    if raw.startswith("package://"):
        rel = raw[len("package://") :]
        candidates = []
        if package_root:
            root = Path(package_root)
            candidates.append(root / rel)
            parts = Path(rel).parts
            if len(parts) > 1:
                candidates.append(root / Path(*parts[1:]))
            candidates.extend(root.rglob(Path(rel).name))
        candidates.append(urdf_dir / Path(rel).name)
    else:
        p = Path(raw)
        candidates = [p if p.is_absolute() else urdf_dir / p]
        if package_root:
            root = Path(package_root)
            candidates.append(root / p.name)
            if p.is_absolute():
                parts = p.parts
                for marker in ("external", "workspace"):
                    if marker in parts:
                        idx = parts.index(marker)
                        if idx + 1 < len(parts):
                            candidates.append(root / Path(*parts[idx + 1 :]))
                if "dvrk_model" in parts:
                    idx = parts.index("dvrk_model")
                    candidates.append(root / Path(*parts[idx:]))
            else:
                candidates.append(root / p)

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def load_ascii_stl(path):
    verts = []
    faces = []
    tri = []
    with Path(path).open("r", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                tri.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(tri) == 3:
                    i = len(verts)
                    verts.extend(tri)
                    faces.append([i, i + 1, i + 2])
                    tri = []
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int32)


def load_binary_stl(path):
    data = Path(path).read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL too small: {path}")
    n_tri = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + n_tri * 50
    if expected > len(data):
        raise ValueError(f"binary STL is truncated: {path}")
    verts = np.empty((n_tri * 3, 3), dtype=float)
    faces = np.empty((n_tri, 3), dtype=np.int32)
    off = 84
    for i in range(n_tri):
        off += 12
        for j in range(3):
            verts[i * 3 + j] = struct.unpack_from("<fff", data, off)
            off += 12
        faces[i] = [i * 3, i * 3 + 1, i * 3 + 2]
        off += 2
    return verts, faces


def load_stl(path):
    header = Path(path).read_bytes()[:256].lstrip().lower()
    if header.startswith(b"solid"):
        try:
            verts, faces = load_ascii_stl(path)
            if len(faces):
                return verts, faces
        except Exception:
            pass
    return load_binary_stl(path)


def load_obj(path):
    verts = []
    faces = []
    with Path(path).open("r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for part in line.split()[1:]:
                    idx.append(int(part.split("/")[0]) - 1)
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int32)


def load_mesh(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".stl":
        return load_stl(path)
    if suffix == ".obj":
        return load_obj(path)
    raise ValueError(f"unsupported mesh format {suffix}; convert to STL/OBJ or install a richer renderer")


def parse_urdf(path, package_root):
    urdf_path = Path(path)
    root = ET.parse(urdf_path).getroot()
    links = {link.attrib["name"]: [] for link in root.findall("link")}
    joints = []

    for link in root.findall("link"):
        lname = link.attrib["name"]
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            if mesh is None or "filename" not in mesh.attrib:
                continue
            mesh_path = resolve_mesh_path(mesh.attrib["filename"], urdf_path.parent, package_root)
            if mesh_path is None:
                print(f"skipping missing mesh: {mesh.attrib['filename']}")
                continue
            scale = parse_vec(mesh.attrib.get("scale"), 3, (1, 1, 1))
            links[lname].append(
                {
                    "origin": origin_transform(visual),
                    "mesh": mesh_path,
                    "scale": scale,
                }
            )

    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        joints.append(
            {
                "name": joint.attrib["name"],
                "type": joint.attrib.get("type", "fixed"),
                "parent": parent.attrib["link"],
                "child": child.attrib["link"],
                "origin": origin_transform(joint),
                "axis": parse_vec(joint.find("axis").attrib.get("xyz"), 3, (1, 0, 0))
                if joint.find("axis") is not None
                else np.asarray([1.0, 0.0, 0.0]),
            }
        )
    return links, joints


def choose_base_link(links, joints):
    children = {j["child"] for j in joints}
    bases = [name for name in links if name not in children]
    return bases[0] if bases else next(iter(links))


def choose_ee_link(links, preferred):
    if preferred and preferred in links:
        return preferred
    for name in links:
        if "tool_wrist_sca_shaft_link" in name:
            return name
    for name in links:
        if "shaft" in name.lower() or "tool" in name.lower():
            return name
    return next(reversed(links))


def joint_values_by_name(joints, q, jaw):
    q = np.asarray(q, dtype=float).reshape(-1)
    labels = [
        ("outer_yaw", "yaw"),
        ("outer_pitch", "pitch"),
        ("insertion", "insertion"),
        ("outer_roll", "roll"),
        ("outer_wrist_pitch", "wrist_pitch"),
        ("outer_wrist_yaw", "wrist_yaw"),
    ]
    nonfixed = [j for j in joints if j["type"] != "fixed"]
    values = {j["name"]: 0.0 for j in joints}

    used = set()
    lname_to_joint = {j["name"].lower(): j["name"] for j in nonfixed}
    for i, (needle_a, needle_b) in enumerate(labels[: len(q)]):
        found = None
        for lname, name in lname_to_joint.items():
            if name in used:
                continue
            if needle_a in lname or needle_b in lname:
                found = name
                break
        if found is not None:
            values[found] = float(q[i])
            used.add(found)

    qi = 0
    for j in nonfixed:
        name = j["name"]
        lname = name.lower()
        if name in used:
            continue
        if any(k in lname for k in ("jaw", "gripper", "finger", "scissor")):
            continue
        if qi < len(q):
            values[name] = float(q[qi])
            qi += 1

    jaw_val = None if jaw is None else float(np.asarray(jaw).reshape(-1)[0])
    if jaw_val is not None:
        side = 1.0
        for j in nonfixed:
            lname = j["name"].lower()
            if any(k in lname for k in ("jaw", "gripper", "finger", "scissor")):
                values[j["name"]] = side * 0.5 * jaw_val
                side *= -1.0
    return values


def joint_motion(joint, value):
    out = np.eye(4, dtype=float)
    if joint["type"] in ("revolute", "continuous"):
        out[:3, :3] = axis_angle(joint["axis"], value)
    elif joint["type"] == "prismatic":
        out[:3, 3] = joint["axis"] * value
    return out


def forward_kinematics(links, joints, values):
    base = choose_base_link(links, joints)
    child_joints = {}
    for joint in joints:
        child_joints.setdefault(joint["parent"], []).append(joint)

    transforms = {base: np.eye(4, dtype=float)}
    stack = [base]
    while stack:
        parent = stack.pop()
        for joint in child_joints.get(parent, []):
            transforms[joint["child"]] = (
                transforms[parent] @ joint["origin"] @ joint_motion(joint, values.get(joint["name"], 0.0))
            )
            stack.append(joint["child"])
    return transforms


def transform_points(points, transform):
    hom = np.c_[points, np.ones((len(points), 1), dtype=float)]
    return (hom @ transform.T)[:, :3]


def project(points, fx, fy, cx, cy):
    z = points[:, 2]
    uv = np.empty((len(points), 2), dtype=float)
    uv[:, 0] = fx * points[:, 0] / z + cx
    uv[:, 1] = fy * points[:, 1] / z + cy
    return uv


def render_meshes(image, mesh_instances, fx, fy, cx, cy, alpha):
    triangles = []
    light = np.asarray([-0.35, -0.55, -0.75], dtype=float)
    light = light / np.linalg.norm(light)

    for verts, faces, transform, _link, _mesh_path in mesh_instances:
        cam = transform_points(verts, transform)
        for face in faces:
            tri = cam[face]
            if not np.all(np.isfinite(tri)) or np.any(tri[:, 2] <= 1.0e-5):
                continue
            v0, v1, v2 = tri
            normal = np.cross(v1 - v0, v2 - v0)
            n_norm = np.linalg.norm(normal)
            if n_norm < 1.0e-12:
                continue
            normal = normal / n_norm
            shade = 0.45 + 0.55 * abs(float(np.dot(normal, light)))
            depth = float(np.mean(tri[:, 2]))
            uv = project(tri, fx, fy, cx, cy)
            triangles.append((depth, uv, shade))

    # Painter's algorithm. For this rigid tool preview it is good enough and
    # keeps the renderer dependency-free.
    triangles.sort(key=lambda item: item[0], reverse=True)
    h, w = image.shape[:2]
    overlay = image.copy()
    for _, uv, shade in triangles:
        if (
            np.max(uv[:, 0]) < -20
            or np.min(uv[:, 0]) > w + 20
            or np.max(uv[:, 1]) < -20
            or np.min(uv[:, 1]) > h + 20
        ):
            continue
        pts = np.round(uv).astype(np.int32)
        color = np.asarray([138, 145, 150], dtype=float) * shade + np.asarray([38, 42, 45], dtype=float) * (1.0 - shade)
        cv2.fillConvexPoly(overlay, pts, tuple(int(c) for c in color), lineType=cv2.LINE_AA)
        cv2.polylines(overlay, [pts], True, (42, 46, 49), 1, lineType=cv2.LINE_AA)
    if alpha >= 1.0:
        image[:] = overlay
    else:
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def compile_regex(pattern):
    return re.compile(pattern) if pattern else None


def regex_match(regex, text):
    return regex is None or regex.search(text) is not None


def load_mesh_instances(links, link_transforms, cam_to_base, include_link, exclude_link, include_mesh, exclude_mesh):
    cache = {}
    instances = []
    include_link_re = compile_regex(include_link)
    exclude_link_re = compile_regex(exclude_link)
    include_mesh_re = compile_regex(include_mesh)
    exclude_mesh_re = compile_regex(exclude_mesh)
    for link, visuals in links.items():
        if link not in link_transforms:
            continue
        if not regex_match(include_link_re, link):
            continue
        if exclude_link_re is not None and exclude_link_re.search(link):
            continue
        for visual in visuals:
            mesh_path = visual["mesh"]
            mesh_text = str(mesh_path)
            if not regex_match(include_mesh_re, mesh_text):
                continue
            if exclude_mesh_re is not None and exclude_mesh_re.search(mesh_text):
                continue
            if mesh_path not in cache:
                try:
                    cache[mesh_path] = load_mesh(mesh_path)
                except Exception as exc:
                    print(f"skipping mesh {mesh_path}: {exc}")
                    continue
            verts, faces = cache[mesh_path]
            if len(verts) == 0 or len(faces) == 0:
                continue
            scaled = verts * visual["scale"].reshape(1, 3)
            transform = cam_to_base @ link_transforms[link] @ visual["origin"]
            instances.append((scaled, faces, transform, link, mesh_path))
    return instances


def scene_thread_radius(scene, radius_override=None, diameter_override=None):
    if diameter_override is not None:
        return 0.5 * float(diameter_override)
    if radius_override is not None:
        return float(radius_override)
    return float(np.asarray(scene["radius"]).reshape(-1)[0]) if "radius" in scene.files else 0.001


def draw_thread(image, scene, fx, fy, cx, cy, radius_override=None, diameter_override=None):
    thread = np.asarray(scene["states"][-1], dtype=float)
    radius = scene_thread_radius(scene, radius_override, diameter_override)
    draw_polyline_camera(
        image,
        thread,
        fx,
        fy,
        cx,
        cy,
        color=(238, 231, 219),
        radius_m=radius,
        min_px=3,
        max_px=12,
        shadow=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-npz", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--joints", required=True)
    parser.add_argument("--jaw", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--package-root", default=None)
    parser.add_argument("--ee-link", default="PSM1_tool_wrist_sca_shaft_link")
    parser.add_argument("--image", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=1025.8822021484375)
    parser.add_argument("--fy", type=float, default=1025.8822021484375)
    parser.add_argument("--cx", type=float, default=167.919017)
    parser.add_argument("--cy", type=float, default=234.152707)
    parser.add_argument("--draw-proxy-if-no-mesh", action="store_true")
    parser.add_argument("--include-link-regex", default=None)
    parser.add_argument("--exclude-link-regex", default=None)
    parser.add_argument("--include-mesh-regex", default=None)
    parser.add_argument("--exclude-mesh-regex", default=None)
    parser.add_argument("--mesh-alpha", type=float, default=0.82)
    parser.add_argument("--thread-radius-m", type=float, default=None)
    parser.add_argument("--thread-diameter-m", type=float, default=None)
    parser.add_argument("--list-rendered-meshes", action="store_true")
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
    mesh_instances = load_mesh_instances(
        links,
        fk,
        cam_to_base,
        include_link=args.include_link_regex,
        exclude_link=args.exclude_link_regex,
        include_mesh=args.include_mesh_regex,
        exclude_mesh=args.exclude_mesh_regex,
    )

    if args.image:
        image = load_rgb_image(args.image, args.width, args.height)
        image = (0.88 * image + 0.12 * make_synthetic_pad(args.width, args.height)).astype(np.uint8)
    else:
        image = make_synthetic_pad(args.width, args.height)

    thread_radius = scene_thread_radius(scene, args.thread_radius_m, args.thread_diameter_m)
    draw_thread(
        image,
        scene,
        args.fx,
        args.fy,
        args.cx,
        args.cy,
        radius_override=thread_radius,
    )
    if mesh_instances:
        render_meshes(image, mesh_instances, args.fx, args.fy, args.cx, args.cy, np.clip(args.mesh_alpha, 0.0, 1.0))
    elif args.draw_proxy_if_no_mesh and "tool_proxy_segments" in scene:
        for seg in scene["tool_proxy_segments"]:
            draw_polyline_camera(
                image,
                seg,
                args.fx,
                args.fy,
                args.cx,
                args.cy,
                color=(46, 50, 54),
                radius_m=0.002,
                min_px=3,
                max_px=14,
                shadow=True,
            )
    else:
        raise RuntimeError("URDF loaded no renderable STL/OBJ meshes")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"rendered PSM scene: {out}")
    print(f"meshes rendered: {len(mesh_instances)}")
    print(f"ee_link: {ee_link}")
    print(f"thread_radius_m: {thread_radius:.9g}")
    print(f"thread_diameter_m: {2.0 * thread_radius:.9g}")
    if args.list_rendered_meshes:
        for _verts, _faces, _transform, link, mesh_path in mesh_instances:
            print(f"  {link}: {mesh_path}")


if __name__ == "__main__":
    main()
