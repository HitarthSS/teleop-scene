#!/usr/bin/env python3
"""Export nearest PSM joint/jaw state arrays for pose-estimation frames.

This pairs the frame timestamps produced by export_rosbag_frames_for_pose.py
with nearest sensor_msgs/JointState messages from a ROS2 bag and writes:

    episode_0000_inputs/joint_000000.npy
    episode_0000_inputs/jaw_000000.npy

The darthandvader/Surgical_Instrument_Pose_Estimation inference script expects
a six-value arm joint vector in --joints:

    [yaw, pitch, insertion, roll, wrist_pitch, wrist_yaw]

and a one-value jaw vector in --jaw.  dVRK /PSM*/joint_states often contains
extra mimic joints, so this exporter selects by JointState.name rather than
blindly saving the whole position vector.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def stamp_ns(msg):
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def nearest(sorted_stamps, stamp):
    pos = np.searchsorted(sorted_stamps, stamp)
    candidates = []
    if pos < len(sorted_stamps):
        candidates.append(sorted_stamps[pos])
    if pos > 0:
        candidates.append(sorted_stamps[pos - 1])
    if not candidates:
        return None
    return min(candidates, key=lambda s: abs(s - stamp))


def read_joint_states(bag_dir, topics):
    stores = {name: {} for name in topics}
    names = {name: {} for name in topics}
    wanted = set(topics.values())
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([Path(bag_dir)], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in wanted]
        found = sorted({c.topic for c in conns})
        missing = sorted(wanted - set(found))
        if missing:
            raise RuntimeError(f"Bag is missing required topics: {missing}")

        topic_to_name = {topic: name for name, topic in topics.items()}
        for conn, _, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, conn.msgtype)
            name = topic_to_name[conn.topic]
            ts = stamp_ns(msg)
            stores[name][ts] = np.asarray(msg.position, dtype=np.float64)
            names[name][ts] = list(msg.name)
    return stores, names


def select_named_positions(values, names, wanted):
    lookup = {name: i for i, name in enumerate(names)}
    missing = [name for name in wanted if name not in lookup]
    if missing:
        raise RuntimeError(
            f"JointState is missing requested name(s) {missing}. "
            f"Available names: {names}"
        )
    return np.asarray([values[lookup[name]] for name in wanted], dtype=np.float64)


def read_manifest(path):
    rows = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found in manifest: {path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="ROS2 bag directory")
    parser.add_argument("--manifest", required=True, help="frame_manifest.csv")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--joint-topic", default="/PSM1/joint_states")
    parser.add_argument("--jaw-topic", default="/PSM1/jaw/measured_js")
    parser.add_argument(
        "--joint-output-names",
        default="yaw,pitch,insertion,roll,wrist_pitch,wrist_yaw",
        help="Comma-separated JointState names to save into joint_*.npy.",
    )
    parser.add_argument(
        "--jaw-source",
        choices=("joint", "topic"),
        default="joint",
        help=(
            "Use the jaw entry embedded in --joint-topic, or the nearest "
            "message on --jaw-topic."
        ),
    )
    parser.add_argument("--jaw-name", default="jaw")
    parser.add_argument("--stamp-column", default="left_stamp_ns")
    parser.add_argument("--max-delta-ms", type=float, default=250.0)
    parser.add_argument("--episode", default="episode_0000")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    topic_map = {"joint": args.joint_topic}
    if args.jaw_source == "topic":
        topic_map["jaw"] = args.jaw_topic
    stores, names = read_joint_states(args.bag, topic_map)
    joint_stamps = sorted(stores["joint"])
    jaw_stamps = sorted(stores["jaw"]) if args.jaw_source == "topic" else []
    joint_output_names = [name.strip() for name in args.joint_output_names.split(",")]
    out_root = Path(args.out)
    out_dir = out_root / f"{args.episode}_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_root / "tool_state_manifest.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "target_stamp_ns",
                "joint_stamp_ns",
                "jaw_stamp_ns",
                "joint_delta_ms",
                "jaw_delta_ms",
                "joint_file",
                "jaw_file",
                "joint_names",
                "jaw_names",
            ],
        )
        writer.writeheader()

        for row in rows:
            frame_i = int(row["frame"])
            target = int(row[args.stamp_column])
            js = nearest(joint_stamps, target)
            if js is None:
                raise RuntimeError(f"No joint match for frame {frame_i}")
            jw = nearest(jaw_stamps, target) if args.jaw_source == "topic" else js
            if jw is None:
                raise RuntimeError(f"No jaw match for frame {frame_i}")

            joint_delta_ms = abs(js - target) / 1e6
            jaw_delta_ms = abs(jw - target) / 1e6
            if joint_delta_ms > args.max_delta_ms or jaw_delta_ms > args.max_delta_ms:
                raise RuntimeError(
                    f"Frame {frame_i} state match too far: "
                    f"joint {joint_delta_ms:.1f} ms, jaw {jaw_delta_ms:.1f} ms"
                )

            joint_name = f"joint_{frame_i:06d}.npy"
            jaw_name = f"jaw_{frame_i:06d}.npy"
            joint_values = select_named_positions(
                stores["joint"][js], names["joint"][js], joint_output_names
            )
            if args.jaw_source == "topic":
                jaw_values = select_named_positions(
                    stores["jaw"][jw], names["jaw"][jw], [args.jaw_name]
                )
            else:
                jaw_values = select_named_positions(
                    stores["joint"][js], names["joint"][js], [args.jaw_name]
                )
            np.save(out_dir / joint_name, joint_values)
            np.save(out_dir / jaw_name, jaw_values)

            writer.writerow(
                {
                    "frame": frame_i,
                    "target_stamp_ns": target,
                    "joint_stamp_ns": js,
                    "jaw_stamp_ns": jw,
                    "joint_delta_ms": f"{joint_delta_ms:.3f}",
                    "jaw_delta_ms": f"{jaw_delta_ms:.3f}",
                    "joint_file": str((out_dir / joint_name).relative_to(out_root)),
                    "jaw_file": str((out_dir / jaw_name).relative_to(out_root)),
                    "joint_names": "|".join(joint_output_names),
                    "jaw_names": args.jaw_name,
                }
            )
            print(
                f"[{frame_i}] joint_{frame_i:06d}.npy "
                f"dt_joint={joint_delta_ms:.1f}ms dt_jaw={jaw_delta_ms:.1f}ms"
            )

    print(f"Wrote {len(rows)} joint/jaw pair(s) under {out_dir}")
    print(f"Manifest: {summary_path}")


if __name__ == "__main__":
    main()
