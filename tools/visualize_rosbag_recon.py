#!/usr/bin/env python3
"""Render saved offline reconstructions over their source ROS bag images."""

import argparse
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


DEFAULT_LEFT = "/stereo/left/rectified_downscaled_image"
DEFAULT_LEFT_MASK = "/stereo/left/sam3_image"


def stamp_ns(msg):
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def image_msg_to_array(msg):
    enc = msg.encoding.lower()
    channels_by_enc = {
        "mono8": 1,
        "8uc1": 1,
        "rgb8": 3,
        "bgr8": 3,
        "8uc3": 3,
    }
    if enc not in channels_by_enc:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    channels = channels_by_enc[enc]
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    rows = data[: height * step].reshape(height, step)
    useful = rows[:, : width * channels]
    if channels == 1:
        return useful.reshape(height, width)
    arr = useful.reshape(height, width, channels)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if enc == "bgr8" else arr


def load_camera(calib):
    cv_file = cv2.FileStorage(str(calib), cv2.FILE_STORAGE_READ)
    if not cv_file.isOpened():
        raise FileNotFoundError(f"Could not open calibration file: {calib}")

    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    image_size = cv_file.getNode("ImageSize").mat()
    img_size = (int(image_size[0][1]), int(image_size[0][0]))
    _, _, P1, _, _, _, _ = cv2.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        img_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=(640, 480),
    )
    return P1


def nearest(sorted_stamps, stamp):
    pos = np.searchsorted(sorted_stamps, stamp)
    candidates = []
    if pos < len(sorted_stamps):
        candidates.append(sorted_stamps[pos])
    if pos > 0:
        candidates.append(sorted_stamps[pos - 1])
    return min(candidates, key=lambda s: abs(s - stamp))


def read_topic_messages(bag_dir, topic_names):
    data = {topic: {} for topic in topic_names}
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([Path(bag_dir)], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in topic_names]
        for conn, _, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, conn.msgtype)
            data[conn.topic][stamp_ns(msg)] = msg
    return data


def stamp_from_samples(path):
    match = re.search(r"frame_\d+_(\d+)_samples\.npz$", path.name)
    if not match:
        raise ValueError(f"Could not parse timestamp from {path.name}")
    return int(match.group(1))


def project(points, P):
    aug = np.column_stack([points, np.ones(len(points))])
    uvw = (P @ aug.T).T
    uvw /= uvw[:, 2:3] + 1e-9
    return uvw[:, :2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--recon-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--left-topic", default=DEFAULT_LEFT)
    parser.add_argument("--left-mask-topic", default=DEFAULT_LEFT_MASK)
    parser.add_argument("--max-frames", type=int, default=12)
    args = parser.parse_args()

    recon_dir = Path(args.recon_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    P1 = load_camera(args.calib)

    samples = sorted(recon_dir.glob("*_samples.npz"))
    if args.max_frames > 0:
        samples = samples[: args.max_frames]
    if not samples:
        raise RuntimeError(f"No *_samples.npz files found in {recon_dir}")

    topics = read_topic_messages(args.bag, [args.left_topic, args.left_mask_topic])
    img_stamps = sorted(topics[args.left_topic])
    mask_stamps = sorted(topics[args.left_mask_topic])

    for sample_path in samples:
        target = stamp_from_samples(sample_path)
        img_stamp = nearest(img_stamps, target)
        mask_stamp = nearest(mask_stamps, target)
        image = image_msg_to_array(topics[args.left_topic][img_stamp])
        mask = image_msg_to_array(topics[args.left_mask_topic][mask_stamp])
        pts = np.load(sample_path)["points"]
        uv = project(pts, P1)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].imshow(image)
        axes[0].set_title("left image")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("left mask")
        axes[2].imshow(image)
        axes[2].imshow(np.where(mask > 0, 1.0, np.nan), cmap="Greens", alpha=0.35)
        axes[2].plot(uv[:, 0], uv[:, 1], "r-", lw=2)
        axes[2].scatter(uv[0, 0], uv[0, 1], c="cyan", s=40, label="start")
        axes[2].scatter(uv[-1, 0], uv[-1, 1], c="yellow", s=40, label="end")
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].set_title("spline projection")
        for ax in axes:
            ax.set_axis_off()
        fig.tight_layout()
        out_path = out_dir / sample_path.name.replace("_samples.npz", "_overlay.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(out_path)


if __name__ == "__main__":
    main()
