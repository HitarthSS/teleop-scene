#!/usr/bin/env python3
"""Newton PSM-URDF jaw contact against the reconstructed thread.

This is the first non-proxy robot-contact path:

  - imports the PSM URDF into Newton as an articulated robot
  - places the URDF base from the pose-estimated camera-to-EE transform
  - attaches hidden capsule collision shapes to the imported jaw link bodies
  - drives the URDF joint coordinates and closes the jaw joints
  - lets the Newton rod/thread collide with those jaw-link collision shapes

The collision capsules are not separate rendered gripper prongs. They are shapes
attached to the actual imported jaw link bodies. The full STL jaw mesh remains
visual geometry.
"""

from __future__ import annotations

import argparse
import inspect
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from export_frame_scene_obj import tube_mesh, write_obj
from export_thread_pad_resting_scene import local_coordinates
from export_thread_robot_newton_scene import make_usd_viewer
from render_frame_scene_with_psm_urdf import (
    choose_ee_link,
    forward_kinematics,
    joint_values_by_name,
    load_mesh,
    load_mesh_instances,
    parse_urdf,
    resolve_mesh_path,
    scene_thread_radius,
    transform_points,
)


def normalize(v, fallback):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1.0e-12:
        return np.asarray(fallback, dtype=np.float64)
    return v / n


def matrix_to_quat_xyzw(r):
    r = np.asarray(r, dtype=np.float64)
    tr = float(np.trace(r))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return np.asarray(
            [
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    i = int(np.argmax(np.diag(r)))
    if i == 0:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        q = [(0.25 * s), (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif i == 1:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        q = [(r[0, 1] + r[1, 0]) / s, (0.25 * s), (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, (0.25 * s), (r[1, 0] - r[0, 1]) / s]
    q = np.asarray(q, dtype=np.float64)
    return q / max(np.linalg.norm(q), 1.0e-12)


def transform_from_rotation_translation(rotation, translation):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = np.asarray(rotation, dtype=np.float64)
    t[:3, 3] = np.asarray(translation, dtype=np.float64)
    return t


def local_from_camera_transform(center, x_axis, y_axis, normal):
    r = np.vstack([x_axis, y_axis, normal]).astype(np.float64)
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = r
    t[:3, 3] = -r @ np.asarray(center, dtype=np.float64)
    return t


def axis_frame_from_z(axis):
    z = normalize(axis, (0.0, 0.0, 1.0))
    helper = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, z))) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    x = normalize(np.cross(helper, z), (1.0, 0.0, 0.0))
    y = normalize(np.cross(z, x), (0.0, 1.0, 0.0))
    return np.column_stack([x, y, z])


def make_importable_urdf(urdf_path, package_root, out_path, remove_collisions=True):
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        resolved = resolve_mesh_path(filename, urdf_path.parent, package_root)
        if resolved is not None:
            mesh.set("filename", str(Path(resolved).resolve()))
    if remove_collisions:
        for link in root.findall("link"):
            for collision in list(link.findall("collision")):
                link.remove(collision)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def link_visual_points(link_visuals):
    pts = []
    for visual in link_visuals:
        try:
            verts, _faces = load_mesh(visual["mesh"])
        except Exception:
            continue
        verts = np.asarray(verts, dtype=np.float64) * np.asarray(visual["scale"], dtype=np.float64).reshape(1, 3)
        pts.append(transform_points(verts, np.asarray(visual["origin"], dtype=np.float64)))
    if not pts:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(pts)


def capsule_from_link_visuals(points, radius, min_half_height, span_scale):
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    if len(points) < 3:
        return center, np.asarray([0.0, 0.0, 1.0], dtype=np.float64), min_half_height
    _u, _s, vh = np.linalg.svd(points - center.reshape(1, 3), full_matrices=False)
    axis = normalize(vh[0], (0.0, 0.0, 1.0))
    proj = (points - center.reshape(1, 3)) @ axis
    lo = float(np.percentile(proj, 2.0))
    hi = float(np.percentile(proj, 98.0))
    center = center + 0.5 * (lo + hi) * axis
    total_len = max((hi - lo) * span_scale, 2.0 * min_half_height + 2.0 * radius)
    half_height = max(0.5 * total_len - radius, min_half_height)
    return center, axis, half_height


def body_positions(state, body_indices):
    body_q = state.body_q.numpy()
    return np.asarray([body_q[int(i), :3] for i in body_indices], dtype=np.float64)


def compile_re(pattern):
    return re.compile(pattern) if pattern else None


def body_link_name(label):
    return str(label).split("/")[-1]


def joint_short_name(label):
    return str(label).split("/")[-1]


def coord_span(builder, joint_index):
    start = int(builder.joint_q_start[joint_index])
    if joint_index + 1 < len(builder.joint_q_start):
        end = int(builder.joint_q_start[joint_index + 1])
    else:
        end = len(builder.joint_q)
    return start, end


def dof_span(builder, joint_index):
    start = int(builder.joint_qd_start[joint_index])
    if joint_index + 1 < len(builder.joint_qd_start):
        end = int(builder.joint_qd_start[joint_index + 1])
    else:
        end = len(builder.joint_qd)
    return start, end


def set_imported_joint_coordinates(builder, imported_joint_indices, values, target_ke, target_kd):
    applied = []
    for joint_idx in imported_joint_indices:
        name = joint_short_name(builder.joint_label[joint_idx])
        if name not in values:
            continue
        q0, q1 = coord_span(builder, joint_idx)
        d0, d1 = dof_span(builder, joint_idx)
        if q1 <= q0:
            continue
        value = float(values[name])
        builder.joint_q[q0] = value
        builder.joint_target_q[q0] = value
        for dof in range(d0, d1):
            builder.joint_target_ke[dof] = float(target_ke)
            builder.joint_target_kd[dof] = float(target_kd)
            builder.joint_damping[dof] = max(float(builder.joint_damping[dof]), float(target_kd))
        applied.append((joint_idx, name, value, q0, q1, d0, d1))
    return applied


def find_jaw_joint_indices(builder, imported_joint_indices, jaw_joint_regex):
    regex = compile_re(jaw_joint_regex)
    out = []
    for joint_idx in imported_joint_indices:
        name = joint_short_name(builder.joint_label[joint_idx])
        q0, q1 = coord_span(builder, joint_idx)
        if q1 <= q0:
            continue
        lname = name.lower()
        if regex is not None and regex.search(name):
            out.append(joint_idx)
        elif any(k in lname for k in ("jaw", "gripper", "finger", "scissor", "ee_link", "sca_ee")):
            out.append(joint_idx)
    return out


def add_capsule_shape(builder, wp, body, center, axis, radius, half_height, cfg, visible, label):
    rot = axis_frame_from_z(axis)
    quat = matrix_to_quat_xyzw(rot)
    xform = wp.transform(
        wp.vec3(float(center[0]), float(center[1]), float(center[2])),
        wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
    )
    kwargs = {
        "body": int(body),
        "xform": xform,
        "radius": float(radius),
        "half_height": float(half_height),
        "cfg": cfg,
        "color": (0.1, 0.8, 1.0) if visible else None,
        "label": label,
    }
    fn = builder.add_shape_capsule
    accepted = inspect.signature(fn).parameters
    return fn(**{k: v for k, v in kwargs.items() if k in accepted})


def set_robot_body_mass(builder, wp, body_indices, mass, inertia_diag):
    inertia = wp.mat33(
        float(inertia_diag),
        0.0,
        0.0,
        0.0,
        float(inertia_diag),
        0.0,
        0.0,
        0.0,
        float(inertia_diag),
    )
    inv_inertia = wp.mat33(
        1.0 / float(inertia_diag),
        0.0,
        0.0,
        0.0,
        1.0 / float(inertia_diag),
        0.0,
        0.0,
        0.0,
        1.0 / float(inertia_diag),
    )
    for body in body_indices:
        builder.body_mass[int(body)] = float(mass)
        builder.body_inv_mass[int(body)] = 1.0 / float(mass) if mass > 0.0 else 0.0
        builder.body_inertia[int(body)] = inertia
        builder.body_inv_inertia[int(body)] = inv_inertia if mass > 0.0 else wp.mat33(0.0)


def set_robot_body_kinematic(builder, wp, body_indices):
    for body in body_indices:
        builder.body_mass[int(body)] = 0.0
        builder.body_inv_mass[int(body)] = 0.0
        builder.body_inertia[int(body)] = wp.mat33(0.0)
        builder.body_inv_inertia[int(body)] = wp.mat33(0.0)


def drive_robot_values(base_values, jaw_names, frame_f, close_frames, close_scale):
    out = dict(base_values)
    if not jaw_names:
        return out
    if frame_f < close_frames:
        alpha = frame_f / max(close_frames - 1, 1)
    else:
        alpha = 1.0
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    for name in jaw_names:
        initial = float(base_values.get(name, 0.0))
        out[name] = float((1.0 - alpha) * initial + alpha * close_scale * initial)
    return out


def set_imported_robot_state_from_fk(state, links, joints, values_frame, local_to_base, body_by_link):
    fk_frame = forward_kinematics(links, joints, values_frame)
    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy() if getattr(state, "body_qd", None) is not None else None
    for link_name, body in body_by_link.items():
        if link_name not in fk_frame:
            continue
        transform = local_to_base @ fk_frame[link_name]
        quat = matrix_to_quat_xyzw(transform[:3, :3])
        body_q[int(body), :3] = transform[:3, 3].astype(body_q.dtype)
        body_q[int(body), 3:7] = quat.astype(body_q.dtype)
        if body_qd is not None:
            body_qd[int(body), :] = 0.0
    state.body_q.assign(body_q)
    if body_qd is not None:
        state.body_qd.assign(body_qd)


def tool_meshes_local_for_obj(links, fk, cam_to_base, center, x_axis, y_axis, normal, include_mesh, exclude_mesh):
    instances = load_mesh_instances(
        links,
        fk,
        cam_to_base,
        include_link=None,
        exclude_link=None,
        include_mesh=include_mesh,
        exclude_mesh=exclude_mesh,
    )
    out = []
    for i, (verts, faces, transform, link, _mesh_path) in enumerate(instances):
        verts_cam = transform_points(verts, transform)
        out.append(
            (
                f"psm_mesh_{i:03d}_{link}",
                "tool",
                local_coordinates(verts_cam, center, x_axis, y_axis, normal),
                np.asarray(faces, dtype=np.int32),
            )
        )
    return out


def write_thread_obj(path, thread_state, pad_vertices, pad_faces, tool_meshes_local, radius, args):
    thread_v, thread_f = tube_mesh(thread_state, radius, args.thread_sides)
    meshes = [
        ("deformable_pad", "pad", pad_vertices, pad_faces),
        ("newton_thread", "thread", thread_v, thread_f),
        *tool_meshes_local,
    ]
    write_obj(Path(path), meshes)


def write_report(
    path,
    args,
    radius,
    imported_body_labels,
    imported_joint_labels,
    jaw_body_labels,
    jaw_joint_labels,
    jaw_shape_indices,
    applied_joints,
    states,
    jaw_targets_start,
    jaw_targets_end,
    temp_urdf,
):
    disp = np.linalg.norm(states[-1] - states[0], axis=1)
    lines = [
        f"rest_scene_npz: {args.rest_scene_npz}",
        f"temp_newton_urdf: {temp_urdf}",
        "contact_model: PSM URDF imported with hidden jaw-link capsule colliders",
        "detached_proxy_prongs: false",
        f"robot_kinematic: {args.robot_kinematic}",
        "obj_frames_include_robot_mesh: true",
        f"frames: {len(states)}",
        f"thread_nodes: {states.shape[1]}",
        f"thread_radius_m: {radius:.9g}",
        f"thread_diameter_m: {2.0 * radius:.9g}",
        f"thread_displacement_m_min_median_max: {disp.min():.9g}, {np.median(disp):.9g}, {disp.max():.9g}",
        f"jaw_collision_shapes: {' '.join(str(int(i)) for i in jaw_shape_indices)}",
        f"jaw_targets_start: {jaw_targets_start}",
        f"jaw_targets_end: {jaw_targets_end}",
        "jaw_bodies:",
        *[f"  {label}" for label in jaw_body_labels],
        "jaw_joints:",
        *[f"  {label}" for label in jaw_joint_labels],
        "applied_recorded_joints:",
        *[f"  {name}: {value:.9g}" for _idx, name, value, *_rest in applied_joints],
        "imported_body_labels:",
        *[f"  {label}" for label in imported_body_labels],
        "imported_joint_labels:",
        *[f"  {label}" for label in imported_joint_labels],
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rest-scene-npz", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--joints", required=True)
    parser.add_argument("--jaw", default=None)
    parser.add_argument("--package-root", default=None)
    parser.add_argument("--ee-link", default="PSM1_tool_wrist_sca_shaft_link")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", type=int, default=18)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--dt", type=float, default=1.0 / 1200.0)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--device", default=None)
    parser.add_argument("--thread-sides", type=int, default=32)
    parser.add_argument("--jaw-link-regex", default="sca_ee_link_1|sca_ee_link_2|ee_link_1|ee_link_2")
    parser.add_argument("--jaw-joint-regex", default="jaw|gripper|finger|scissor|sca_ee|ee_link")
    parser.add_argument("--jaw-collision-radius", type=float, default=0.00065)
    parser.add_argument("--jaw-min-half-height", type=float, default=0.0014)
    parser.add_argument("--jaw-span-scale", type=float, default=0.85)
    parser.add_argument("--show-jaw-colliders", action="store_true")
    parser.add_argument("--close-frames", type=int, default=8)
    parser.add_argument("--jaw-close-scale", type=float, default=0.0)
    parser.add_argument("--robot-target-ke", type=float, default=1.0e5)
    parser.add_argument("--robot-target-kd", type=float, default=1.0e3)
    parser.add_argument("--robot-kinematic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robot-body-mass", type=float, default=2.0)
    parser.add_argument("--robot-body-inertia", type=float, default=1.0e-3)
    parser.add_argument("--obj-include-mesh-regex", default="tool_wrist|tool_wrist_sca")
    parser.add_argument("--obj-exclude-mesh-regex", default="tool_main_link")
    parser.add_argument("--gravity", type=float, default=0.0)
    parser.add_argument("--stretch-stiffness", type=float, default=5.0e3)
    parser.add_argument("--stretch-damping", type=float, default=2.0e1)
    parser.add_argument("--bend-stiffness", type=float, default=1.0e0)
    parser.add_argument("--bend-damping", type=float, default=5.0e-2)
    parser.add_argument("--contact-stiffness", type=float, default=2.0e5)
    parser.add_argument("--contact-damping", type=float, default=10.0)
    parser.add_argument("--friction", type=float, default=2.0)
    parser.add_argument("--contact-buffer-size", type=int, default=8192)
    args = parser.parse_args()

    import newton
    import warp as wp

    if args.device:
        wp.set_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rest = np.load(args.rest_scene_npz, allow_pickle=True)
    thread = np.asarray(rest["thread_resting_newton"], dtype=np.float64)
    radius = float(np.asarray(rest["thread_radius"]).reshape(()))
    pad_vertices = np.asarray(rest["pad_grid_vertices_newton"], dtype=np.float64)
    pad_faces = np.asarray(rest["pad_grid_faces"], dtype=np.int32)
    center = np.asarray(rest["pad_center_camera"], dtype=np.float64)
    x_axis = np.asarray(rest["pad_x_axis_camera"], dtype=np.float64)
    y_axis = np.asarray(rest["pad_y_axis_camera"], dtype=np.float64)
    normal = np.asarray(rest["pad_normal_camera"], dtype=np.float64)

    links, joints = parse_urdf(args.urdf, args.package_root)
    q = np.load(args.joints)
    jaw = np.load(args.jaw) if args.jaw else None
    values = joint_values_by_name(joints, q, jaw)
    fk = forward_kinematics(links, joints, values)
    ee_link = choose_ee_link(links, args.ee_link)
    cam_to_ee = np.asarray(rest["tool_cam_to_ee"], dtype=np.float64)
    cam_to_base = cam_to_ee @ np.linalg.inv(fk[ee_link])
    local_from_cam = local_from_camera_transform(center, x_axis, y_axis, normal)
    local_to_base = local_from_cam @ cam_to_base
    tool_obj_meshes = tool_meshes_local_for_obj(
        links,
        fk,
        cam_to_base,
        center,
        x_axis,
        y_axis,
        normal,
        args.obj_include_mesh_regex,
        args.obj_exclude_mesh_regex,
    )
    base_pos = local_to_base[:3, 3]
    base_quat = matrix_to_quat_xyzw(local_to_base[:3, :3])

    temp_urdf = make_importable_urdf(
        args.urdf,
        args.package_root,
        out_dir / "psm1_si_newton_importable_visual_only.urdf",
        remove_collisions=True,
    )

    builder = newton.ModelBuilder(gravity=float(args.gravity))
    builder.default_shape_cfg.ke = args.contact_stiffness
    builder.default_shape_cfg.kd = args.contact_damping
    builder.default_shape_cfg.mu = args.friction

    body_start = len(builder.body_label)
    joint_start = len(builder.joint_label)
    shape_start = len(builder.shape_type)
    builder.add_urdf(
        str(temp_urdf),
        xform=wp.transform(
            wp.vec3(float(base_pos[0]), float(base_pos[1]), float(base_pos[2])),
            wp.quat(float(base_quat[0]), float(base_quat[1]), float(base_quat[2]), float(base_quat[3])),
        ),
        floating=False,
        parse_visuals_as_colliders=False,
        enable_self_collisions=False,
        ignore_inertial_definitions=True,
        joint_ordering="dfs",
        bodies_follow_joint_ordering=True,
    )
    body_end = len(builder.body_label)
    joint_end = len(builder.joint_label)
    imported_bodies = list(range(body_start, body_end))
    imported_joints = list(range(joint_start, joint_end))
    imported_body_labels = [str(builder.body_label[i]) for i in imported_bodies]
    imported_joint_labels = [str(builder.joint_label[i]) for i in imported_joints]
    applied_joints = set_imported_joint_coordinates(
        builder,
        imported_joints,
        values,
        args.robot_target_ke,
        args.robot_target_kd,
    )

    body_by_link = {body_link_name(builder.body_label[i]): i for i in imported_bodies}
    jaw_link_re = compile_re(args.jaw_link_regex)
    jaw_shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=1000.0,
        ke=args.contact_stiffness,
        kd=args.contact_damping,
        mu=args.friction,
        margin=0.0,
        gap=0.0,
        is_visible=bool(args.show_jaw_colliders),
        has_shape_collision=True,
        has_particle_collision=True,
    )
    jaw_shape_indices = []
    jaw_body_labels = []
    for link_name, visuals in links.items():
        if jaw_link_re is not None and jaw_link_re.search(link_name) is None:
            continue
        if link_name not in body_by_link:
            continue
        points = link_visual_points(visuals)
        if len(points) == 0:
            continue
        cap_center, cap_axis, cap_half = capsule_from_link_visuals(
            points,
            radius=args.jaw_collision_radius,
            min_half_height=args.jaw_min_half_height,
            span_scale=args.jaw_span_scale,
        )
        body = body_by_link[link_name]
        shape_idx = add_capsule_shape(
            builder,
            wp,
            body,
            cap_center,
            cap_axis,
            args.jaw_collision_radius,
            cap_half,
            jaw_shape_cfg,
            args.show_jaw_colliders,
            label=f"real_psm_jaw_collision_{link_name}",
        )
        jaw_shape_indices.append(shape_idx)
        jaw_body_labels.append(str(builder.body_label[body]))

    if not jaw_shape_indices:
        available = "\n  ".join(imported_body_labels)
        raise RuntimeError(
            "No imported URDF jaw link bodies matched --jaw-link-regex "
            f"{args.jaw_link_regex!r}. Available imported body labels:\n  {available}"
        )

    for i, shape_a in enumerate(jaw_shape_indices):
        for shape_b in jaw_shape_indices[i + 1 :]:
            builder.add_shape_collision_filter_pair(int(shape_a), int(shape_b))

    if args.robot_kinematic:
        set_robot_body_kinematic(builder, wp, imported_bodies)
    else:
        set_robot_body_mass(builder, wp, imported_bodies, args.robot_body_mass, args.robot_body_inertia)

    wp_points = [wp.vec3(float(x), float(y), float(z)) for x, y, z in thread]
    quats = newton.utils.create_parallel_transport_cable_quaternions(wp_points)
    rod_kwargs = dict(
        positions=wp_points,
        quaternions=quats,
        radius=radius,
        stretch_stiffness=args.stretch_stiffness,
        stretch_damping=args.stretch_damping,
        bend_stiffness=args.bend_stiffness,
        bend_damping=args.bend_damping,
        closed=False,
        label="real_psm_contact_thread",
    )
    if "body_frame_origin" in inspect.signature(builder.add_rod).parameters:
        rod_kwargs["body_frame_origin"] = "com"
    rod_bodies, rod_joints = builder.add_rod(**rod_kwargs)

    builder.color()
    model = builder.finalize()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model, contact_matching="latest")
    contacts = collision_pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=args.iterations,
        rigid_body_contact_buffer_size=args.contact_buffer_size,
        rigid_contact_history=True,
    )

    jaw_joint_indices = find_jaw_joint_indices(builder, imported_joints, args.jaw_joint_regex)
    jaw_joint_labels = [str(builder.joint_label[i]) for i in jaw_joint_indices]
    jaw_names = []
    for joint_idx in jaw_joint_indices:
        name = joint_short_name(builder.joint_label[joint_idx])
        q0, q1 = coord_span(builder, joint_idx)
        if q1 > q0:
            jaw_names.append(name)
    if not jaw_names:
        available = "\n  ".join(imported_joint_labels)
        raise RuntimeError(
            "No imported URDF jaw joints with coordinates matched --jaw-joint-regex "
            f"{args.jaw_joint_regex!r}. Available imported joint labels:\n  {available}"
        )

    viewer = make_usd_viewer(out_dir / "psm_urdf_thread_contact.usd", args.fps)
    viewer.set_model(model)
    obj_dir = out_dir / "obj_frames"
    obj_dir.mkdir(parents=True, exist_ok=True)

    states = []
    times = []
    jaw_targets_start = [float(values.get(name, 0.0)) for name in jaw_names]
    jaw_targets_end = None
    sim_time = 0.0

    def apply_robot_target(frame_f):
        target_values = drive_robot_values(
            values,
            jaw_names,
            frame_f,
            args.close_frames,
            args.jaw_close_scale,
        )
        set_imported_robot_state_from_fk(state_0, links, joints, target_values, local_to_base, body_by_link)
        set_imported_robot_state_from_fk(state_1, links, joints, target_values, local_to_base, body_by_link)
        return target_values

    def save_frame(frame_idx, frame_time):
        thread_state = body_positions(state_0, rod_bodies)
        states.append(thread_state)
        times.append(float(frame_time))
        write_thread_obj(
            obj_dir / f"frame_{frame_idx:06d}_thread_pad_robot.obj",
            thread_state,
            pad_vertices,
            pad_faces,
            tool_obj_meshes,
            radius,
            args,
        )
        viewer.begin_frame(float(frame_time))
        viewer.log_state(state_0)
        try:
            points_wp = wp.array(pad_vertices.astype(np.float32), dtype=wp.vec3, device=args.device)
            indices_wp = wp.array(pad_faces.astype(np.int32).reshape(-1), dtype=wp.int32, device=args.device)
            viewer.log_mesh("/deformable_pad", points_wp, indices_wp, color=(0.84, 0.43, 0.38), backface_culling=False)
        except TypeError:
            pass
        viewer.end_frame()
        print(f"[{frame_idx}] saved t={frame_time:.6f}s")

    apply_robot_target(0.0)
    save_frame(0, 0.0)
    total_steps = (args.frames - 1) * args.substeps
    for step in range(1, total_steps + 1):
        frame_f = step / float(args.substeps)
        target = apply_robot_target(frame_f)
        jaw_targets_end = [float(target.get(name, 0.0)) for name in jaw_names]
        state_0.clear_forces()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, args.dt)
        state_0, state_1 = state_1, state_0
        sim_time = step * args.dt
        if step % args.substeps == 0:
            save_frame(step // args.substeps, sim_time)

    viewer.close()
    states = np.asarray(states, dtype=np.float64)
    np.savez(
        out_dir / "psm_urdf_thread_contact.npz",
        times=np.asarray(times, dtype=np.float64),
        states_newton=states,
        initial_thread_newton=thread,
        thread_radius=np.asarray(radius, dtype=np.float64),
        rod_bodies=np.asarray(rod_bodies, dtype=np.int32),
        rod_joints=np.asarray(rod_joints, dtype=np.int32),
        jaw_shape_indices=np.asarray(jaw_shape_indices, dtype=np.int32),
        jaw_joint_labels=np.asarray(jaw_joint_labels),
        jaw_body_labels=np.asarray(jaw_body_labels),
        jaw_targets_start=np.asarray(jaw_targets_start, dtype=np.float64),
        jaw_targets_end=np.asarray(jaw_targets_end if jaw_targets_end is not None else jaw_targets_start, dtype=np.float64),
    )
    write_report(
        out_dir / "psm_urdf_thread_contact_report.txt",
        args,
        radius,
        imported_body_labels,
        imported_joint_labels,
        jaw_body_labels,
        jaw_joint_labels,
        jaw_shape_indices,
        applied_joints,
        states,
        jaw_targets_start,
        jaw_targets_end if jaw_targets_end is not None else jaw_targets_start,
        temp_urdf,
    )
    print(f"USD: {out_dir / 'psm_urdf_thread_contact.usd'}")
    print(f"NPZ: {out_dir / 'psm_urdf_thread_contact.npz'}")
    print(f"OBJ thread/pad frames: {obj_dir}")
    print(f"report: {out_dir / 'psm_urdf_thread_contact_report.txt'}")
    print(f"jaw bodies: {jaw_body_labels}")
    print(f"jaw joints: {jaw_joint_labels}")


if __name__ == "__main__":
    main()
