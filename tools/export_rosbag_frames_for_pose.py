#!/usr/bin/env python3
"""Export synchronized stereo ROS bag frames for tool pose estimation.

This creates a simple image-file dataset from the same ROS2 bag used by the
thread reconstruction.  The output layout is intentionally close to the
Genesis/tool-pose scripts that expect files such as:

    episode_0000/colors/left_image_000212.jpg

It also writes a manifest mapping frame indices to ROS timestamps so later
thread samples and tool poses can be synchronized.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from offline_rosbag_reconstruct import (
    DEFAULT_LEFT,
    DEFAULT_LEFT_MASK,
    DEFAULT_RIGHT,
    DEFAULT_RIGHT_MASK,
    as_mask,
    as_rgb,
    read_synced_frames,
)


def write_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write image: {path}")


def write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8)):
        raise RuntimeError(f"Failed to write mask: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="ROS2 bag directory")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--left-topic", default=DEFAULT_LEFT)
    parser.add_argument("--right-topic", default=DEFAULT_RIGHT)
    parser.add_argument("--left-mask-topic", default=DEFAULT_LEFT_MASK)
    parser.add_argument("--right-mask-topic", default=DEFAULT_RIGHT_MASK)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=75.0)
    parser.add_argument("--episode", default="episode_0000")
    args = parser.parse_args()

    topics = {
        "left": args.left_topic,
        "right": args.right_topic,
        "left_mask": args.left_mask_topic,
        "right_mask": args.right_mask_topic,
    }
    stores, stamps = read_synced_frames(
        args.bag, topics, tolerance_ns=int(args.sync_tolerance_ms * 1e6)
    )
    selected = stamps[args.start_index :: max(args.stride, 1)]
    if args.max_frames > 0:
        selected = selected[: args.max_frames]

    out = Path(args.out)
    colors = out / args.episode / "colors"
    masks = out / args.episode / "masks"
    manifest_path = out / "frame_manifest.csv"
    out.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "left_stamp_ns",
                "right_stamp_ns",
                "left_mask_stamp_ns",
                "right_mask_stamp_ns",
                "left_image",
                "right_image",
                "left_mask",
                "right_mask",
            ],
        )
        writer.writeheader()

        for frame_i, stamp_tuple in enumerate(selected):
            left_stamp, right_stamp, left_mask_stamp, right_mask_stamp = stamp_tuple
            left = as_rgb(stores["left"][left_stamp])
            right = as_rgb(stores["right"][right_stamp])
            left_mask = as_mask(stores["left_mask"][left_mask_stamp])
            right_mask = as_mask(stores["right_mask"][right_mask_stamp])

            left_name = f"left_image_{frame_i:06d}.jpg"
            right_name = f"right_image_{frame_i:06d}.jpg"
            left_mask_name = f"left_mask_{frame_i:06d}.png"
            right_mask_name = f"right_mask_{frame_i:06d}.png"

            write_rgb(colors / left_name, left)
            write_rgb(colors / right_name, right)
            write_mask(masks / left_mask_name, left_mask)
            write_mask(masks / right_mask_name, right_mask)

            writer.writerow(
                {
                    "frame": frame_i,
                    "left_stamp_ns": left_stamp,
                    "right_stamp_ns": right_stamp,
                    "left_mask_stamp_ns": left_mask_stamp,
                    "right_mask_stamp_ns": right_mask_stamp,
                    "left_image": str((colors / left_name).relative_to(out)),
                    "right_image": str((colors / right_name).relative_to(out)),
                    "left_mask": str((masks / left_mask_name).relative_to(out)),
                    "right_mask": str((masks / right_mask_name).relative_to(out)),
                }
            )
            print(f"[{frame_i}] {left_name} stamp={left_stamp}")

    print(f"Wrote {len(selected)} frame(s) under {out}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
