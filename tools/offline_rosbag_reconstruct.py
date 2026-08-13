#!/usr/bin/env python3
"""Offline reconstruction from a ROS2 bag that already contains stereo masks.

This bypasses the live ROS node and custom trigger messages. It expects a bag
with rectified stereo RGB images plus mono thread masks, then runs:

    Select.keypt_selection -> Order.keypt_ordering -> Optim.optim

Each accepted frame is saved as a pickled BSpline and as sampled XYZ points.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("THREAD_RECON_BREAK", "0")

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from thread_reconstruction.keypt_selection import Select
from thread_reconstruction.keypt_ordering import Order
from thread_reconstruction.optim import Optim


DEFAULT_LEFT = "/stereo/left/rectified_downscaled_image"
DEFAULT_RIGHT = "/stereo/right/rectified_downscaled_image"
DEFAULT_LEFT_MASK = "/stereo/left/sam3_image"
DEFAULT_RIGHT_MASK = "/stereo/right/sam3_image"


def stamp_ns(msg):
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


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
    if any(x is None for x in (K1, D1, K2, D2, R, T, image_size)):
        raise ValueError(f"Calibration file is missing stereo keys: {calib}")

    img_size = (int(image_size[0][1]), int(image_size[0][0]))
    new_size = (640, 480)
    _, _, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        img_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=new_size,
    )
    return P1, P2, Q


def read_synced_frames(bag_dir, topics, tolerance_ns):
    wanted = set(topics.values())
    stores = {name: {} for name in topics}
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
            stores[name][stamp_ns(msg)] = msg

    counts = {k: len(v) for k, v in stores.items()}
    exact = sorted(set.intersection(*(set(s.keys()) for s in stores.values())))
    if exact:
        print(f"Using {len(exact)} exact timestamp matches.")
        return stores, [(s, s, s, s) for s in exact]

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

    left_stamps = sorted(stores["left"])
    right_stamps = sorted(stores["right"])
    left_mask_stamps = sorted(stores["left_mask"])
    right_mask_stamps = sorted(stores["right_mask"])

    # Anchor on left-mask frames because masks are the sparse stream: this yields
    # one reconstruction per available segmentation rather than mostly duplicates.
    matches = []
    for lm in left_mask_stamps:
        l = nearest(left_stamps, lm)
        r = nearest(right_stamps, lm)
        rm = nearest(right_mask_stamps, lm)
        if l is None or r is None or rm is None:
            continue
        spread = max(l, r, lm, rm) - min(l, r, lm, rm)
        if spread <= tolerance_ns:
            matches.append((l, r, lm, rm))

    if not matches:
        raise RuntimeError(
            "No approximate timestamp match across image/mask topics. "
            f"Per-topic counts: {counts}. "
            f"Try increasing --sync-tolerance-ms above {tolerance_ns / 1e6:.1f}."
        )

    print(
        f"Using {len(matches)} approximate timestamp matches "
        f"(tolerance {tolerance_ns / 1e6:.1f} ms)."
    )
    return stores, matches


def as_rgb(msg):
    arr = image_msg_to_array(msg)
    enc = msg.encoding.lower()
    if enc in ("rgb8", "8uc3"):
        return arr
    if enc == "bgr8":
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if enc in ("mono8", "8uc1"):
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    raise ValueError(f"Unsupported RGB image encoding: {msg.encoding}")


def as_mask(msg):
    arr = image_msg_to_array(msg)
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask = np.asarray(arr)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def image_msg_to_array(msg):
    """Decode common uncompressed sensor_msgs/Image encodings.

    This avoids depending on cv_bridge or rosbags.image, neither of which is
    consistently present in lightweight Docker installs.
    """
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
    if data.size < height * step:
        raise ValueError(
            f"Image data too small for {height=} {width=} {step=}: "
            f"{data.size} bytes"
        )
    rows = data[: height * step].reshape(height, step)
    useful = rows[:, : width * channels]
    if channels == 1:
        return useful.reshape(height, width)
    return useful.reshape(height, width, channels)


def save_result(out_dir, frame_i, stamp, thread, specs):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{frame_i:05d}_{stamp}"
    with (out_dir / f"{stem}_spline.pkl").open("wb") as f:
        pickle.dump(thread, f)
    with (out_dir / f"{stem}_specs.pkl").open("wb") as f:
        pickle.dump(specs, f)

    spline = thread["thread"]
    u = np.linspace(0.0, 1.0, 200)
    np.savez(
        out_dir / f"{stem}_samples.npz",
        u=u,
        points=spline(u),
        reliability=specs.get("reliability"),
        lower_constr=specs.get("lower_constr"),
        upper_constr=specs.get("upper_constr"),
        keypt_s=specs.get("keypt_s"),
    )


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_calib = repo_root / "assets" / "camera_calibration_fei.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="ROS2 bag directory")
    parser.add_argument("--out", default="offline_recon_out")
    parser.add_argument("--calib", default=str(default_calib))
    parser.add_argument("--left-topic", default=DEFAULT_LEFT)
    parser.add_argument("--right-topic", default=DEFAULT_RIGHT)
    parser.add_argument("--left-mask-topic", default=DEFAULT_LEFT_MASK)
    parser.add_argument("--right-mask-topic", default=DEFAULT_RIGHT_MASK)
    parser.add_argument("--max-frames", type=int, default=25)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=75.0)
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--debug-plots", action="store_true")
    args = parser.parse_args()

    P1, P2, Q = load_camera(args.calib)
    worker_args = SimpleNamespace(time=args.time)
    selector = Select(worker_args)
    orderer = Order(worker_args)
    optimizer = Optim(worker_args)

    topics = {
        "left": args.left_topic,
        "right": args.right_topic,
        "left_mask": args.left_mask_topic,
        "right_mask": args.right_mask_topic,
    }
    matches = read_synced_frames(
        args.bag, topics, tolerance_ns=int(args.sync_tolerance_ms * 1e6)
    )
    stores, stamps = matches
    selected = stamps[args.start_index :: max(args.stride, 1)]
    if args.max_frames > 0:
        selected = selected[: args.max_frames]

    print(f"Found {len(stamps)} synchronized image/mask sets.")
    print(f"Processing {len(selected)} frame(s) into {args.out}")

    accepted = 0
    prev_thread = None
    for frame_i, stamp_tuple in enumerate(selected):
        try:
            left_stamp, right_stamp, left_mask_stamp, right_mask_stamp = stamp_tuple
            img1 = as_rgb(stores["left"][left_stamp])
            img2 = as_rgb(stores["right"][right_stamp])
            mask1 = as_mask(stores["left_mask"][left_mask_stamp])
            mask2 = as_mask(stores["right_mask"][right_mask_stamp])

            img1 = np.where(mask1[..., None] > 0, img1, 0).astype(np.uint32)
            img2 = np.where(mask2[..., None] > 0, img2, 0).astype(np.uint32)

            select_out = selector.keypt_selection(
                img1=img1,
                img2=img2,
                mask1=mask1,
                Q=Q,
                speedy=not args.debug_plots,
            )
            (
                img_3d,
                _,
                cluster_map,
                keypoints,
                grow_paths,
                adjacents,
                intersection_segments,
                _dense_pts,
                *rest,
            ) = select_out
            reliable_flag = rest[0] if rest else None

            _, keypoints, _, order = orderer.keypt_ordering(
                img1,
                img_3d,
                cluster_map,
                keypoints,
                grow_paths,
                adjacents,
                intersection_segments=intersection_segments,
                speedy=not args.debug_plots,
                keypt_conf=getattr(selector, "last_keypt_conf", None),
            )

            if reliable_flag is not None and order is not None:
                order = [idx for idx in order if reliable_flag[idx]]

            keypt_conf = getattr(selector, "last_keypt_conf", None)
            thread, specs = optimizer.optim(
                mask1,
                keypoints,
                order,
                P1[:, :-1],
                P1,
                P2,
                speedy=not args.debug_plots,
                x_prior_thread=prev_thread,
                keypt_conf=keypt_conf,
            )
            if thread is None:
                print(f"[{frame_i}] skipped: optimizer returned no thread")
                continue

            save_result(Path(args.out), frame_i, left_mask_stamp, thread, specs)
            prev_thread = thread["thread"]
            accepted += 1
            spread_ms = (
                max(stamp_tuple) - min(stamp_tuple)
            ) / 1e6
            print(
                f"[{frame_i}] accepted stamp={left_mask_stamp} "
                f"spread={spread_ms:.1f}ms keypoints={len(order)}"
            )
        except Exception as exc:
            print(
                f"[{frame_i}] failed stamps={stamp_tuple}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    print(f"Accepted {accepted}/{len(selected)} frame(s).")


if __name__ == "__main__":
    main()
