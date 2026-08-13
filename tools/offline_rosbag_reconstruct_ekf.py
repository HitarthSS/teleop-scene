#!/usr/bin/env python3
"""Offline ROS bag reconstruction through the warm-ordering + EKF path.

The original offline runner is intentionally cold per frame:

    Select.keypt_selection -> Order.keypt_ordering -> Optim.optim

That is useful for quick bag checks, but it bypasses the EKF/warm-ordering
logic used by the patched tracker.  This script uses the cold optimizer only
to seed frame 0, then processes later frames with:

    Select.keypt_selection -> Order.run_warm_ordering_with_ekf

and saves the EKF posterior spline samples for overlay/debug rendering.
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("THREAD_RECON_BREAK", "0")

import numpy as np

from offline_rosbag_reconstruct import (
    DEFAULT_LEFT,
    DEFAULT_LEFT_MASK,
    DEFAULT_RIGHT,
    DEFAULT_RIGHT_MASK,
    as_mask,
    as_rgb,
    load_camera,
    read_synced_frames,
    save_result,
)
from thread_reconstruction import ekf_params
from thread_reconstruction.keypt_ordering import Order
from thread_reconstruction.keypt_selection import Select
from thread_reconstruction.optim import Optim


IDENTITY_T = np.eye(4)


def run_selection(selector, img1, img2, mask1, mask2, Q, speedy=True):
    """Run keypoint selection with the same image masking as the cold runner."""
    img1_masked = np.where(mask1[..., None] > 0, img1, 0).astype(np.uint32)
    img2_masked = np.where(mask2[..., None] > 0, img2, 0).astype(np.uint32)
    select_out = selector.keypt_selection(
        img1=img1_masked,
        img2=img2_masked,
        mask1=mask1,
        Q=Q,
        speedy=speedy,
    )
    (
        img_3d,
        _,
        cluster_map,
        keypoints,
        grow_paths,
        adjacents,
        intersection_segments,
        dense_pts,
        *rest,
    ) = select_out
    reliable_flag = rest[0] if rest else None
    return {
        "img1": img1_masked,
        "img_3d": img_3d,
        "cluster_map": cluster_map,
        "keypoints": keypoints,
        "grow_paths": grow_paths,
        "adjacents": adjacents,
        "intersection_segments": intersection_segments,
        "dense_pts": dense_pts,
        "reliable_flag": reliable_flag,
        "keypt_conf": getattr(selector, "last_keypt_conf", None),
    }


def cold_seed(mask1, P1, P2, selector, orderer, optimizer, selected, speedy=True):
    _, keypoints, _, order = orderer.keypt_ordering(
        selected["img1"],
        selected["img_3d"],
        selected["cluster_map"],
        selected["keypoints"],
        selected["grow_paths"],
        selected["adjacents"],
        intersection_segments=selected["intersection_segments"],
        speedy=speedy,
        keypt_conf=selected["keypt_conf"],
    )
    reliable_flag = selected["reliable_flag"]
    if reliable_flag is not None and order is not None:
        order = [idx for idx in order if reliable_flag[idx]]

    thread, specs = optimizer.optim(
        mask1,
        keypoints,
        order,
        P1[:, :-1],
        P1,
        P2,
        speedy=speedy,
        x_prior_thread=None,
        keypt_conf=getattr(selector, "last_keypt_conf", None),
    )
    return thread, specs, order


def save_ekf_result(out_dir, frame_i, stamp, orderer, nwsk_full=None):
    spline = orderer.ekf.get_spline()
    specs = {
        "reliability": np.ones(200, dtype=float),
        "keypt_s": None if nwsk_full is None else np.asarray(nwsk_full, dtype=float),
        "lower_constr": None,
        "upper_constr": None,
        "source": "offline_warm_ordering_ekf",
        "ekf_output_mode": ekf_params.EKF_OUTPUT_MODE,
    }
    save_result(Path(out_dir), frame_i, stamp, {"thread": spline}, specs)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_calib = repo_root / "assets" / "camera_calibration_fei.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="ROS2 bag directory")
    parser.add_argument("--out", default="offline_recon_ekf_out")
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
    parser.add_argument(
        "--fallback-cold",
        action="store_true",
        help="Re-seed from the cold optimizer when EKF warm ordering fails.",
    )
    parser.add_argument(
        "--projection-order",
        choices=("on", "off"),
        default="on",
        help=(
            "Use KF projection t-sort ordering. Turn off to run the older "
            "warm segment assembly path, which can recover unmatched mask "
            "segments in offline/no-gripper bags."
        ),
    )
    args = parser.parse_args()

    P1, P2, Q = load_camera(args.calib)
    worker_args = SimpleNamespace(time=args.time)
    selector = Select(worker_args)
    orderer = Order(worker_args)
    optimizer = Optim(worker_args)
    speedy = not args.debug_plots
    orderer.KF_PROJECTION_ORDER = args.projection_order == "on"

    topics = {
        "left": args.left_topic,
        "right": args.right_topic,
        "left_mask": args.left_mask_topic,
        "right_mask": args.right_mask_topic,
    }
    stores, stamps = read_synced_frames(
        args.bag, topics, tolerance_ns=int(args.sync_tolerance_ms * 1e6)
    )
    selected_stamps = stamps[args.start_index :: max(args.stride, 1)]
    if args.max_frames > 0:
        selected_stamps = selected_stamps[: args.max_frames]

    print(f"EKF_OUTPUT_MODE={ekf_params.EKF_OUTPUT_MODE}")
    print(f"KF_PROJECTION_ORDER={orderer.KF_PROJECTION_ORDER}")
    print(f"Found {len(stamps)} synchronized image/mask sets.")
    print(f"Processing {len(selected_stamps)} frame(s) into {args.out}")

    accepted = 0
    prev_thread = None
    for frame_i, stamp_tuple in enumerate(selected_stamps):
        try:
            left_stamp, right_stamp, left_mask_stamp, right_mask_stamp = stamp_tuple
            img1 = as_rgb(stores["left"][left_stamp])
            img2 = as_rgb(stores["right"][right_stamp])
            mask1 = as_mask(stores["left_mask"][left_mask_stamp])
            mask2 = as_mask(stores["right_mask"][right_mask_stamp])

            selected = run_selection(
                selector,
                img1,
                img2,
                mask1,
                mask2,
                Q,
                speedy=speedy,
            )

            if prev_thread is None:
                thread, specs, order = cold_seed(
                    mask1,
                    P1,
                    P2,
                    selector,
                    orderer,
                    optimizer,
                    selected,
                    speedy=speedy,
                )
                if thread is None:
                    print(f"[{frame_i}] skipped: cold seed returned no thread")
                    continue
                prev_thread = thread["thread"]
                orderer.ekf.initialize(prev_thread)
                orderer._ekf_initialized = True
                save_result(Path(args.out), frame_i, left_mask_stamp, thread, specs)
                accepted += 1
                print(
                    f"[{frame_i}] accepted cold seed stamp={left_mask_stamp} "
                    f"keypoints={0 if order is None else len(order)}"
                )
                continue

            new_keypoints, order, nwsk_full = orderer.run_warm_ordering_with_ekf(
                mask1,
                selected["keypoints"],
                P1,
                warm_thread=prev_thread,
                curr_T=IDENTITY_T,
                prev_T=IDENTITY_T,
                speedy=speedy,
                update_ekf=True,
                adjacents=selected["adjacents"],
                intersection_segments=selected["intersection_segments"],
                dense_pts=selected["dense_pts"],
                keypt_conf=selected["keypt_conf"],
            )

            if new_keypoints is None or order is None:
                if not args.fallback_cold:
                    print(f"[{frame_i}] skipped: EKF warm ordering returned no order")
                    continue
                thread, specs, order = cold_seed(
                    mask1,
                    P1,
                    P2,
                    selector,
                    orderer,
                    optimizer,
                    selected,
                    speedy=speedy,
                )
                if thread is None:
                    print(f"[{frame_i}] skipped: fallback cold seed returned no thread")
                    continue
                prev_thread = thread["thread"]
                orderer.ekf.initialize(prev_thread)
                orderer._ekf_initialized = True
                save_result(Path(args.out), frame_i, left_mask_stamp, thread, specs)
                accepted += 1
                print(
                    f"[{frame_i}] accepted fallback cold stamp={left_mask_stamp} "
                    f"keypoints={0 if order is None else len(order)}"
                )
                continue

            save_ekf_result(args.out, frame_i, left_mask_stamp, orderer, nwsk_full)
            prev_thread = orderer.ekf.get_spline()
            accepted += 1
            spread_ms = (max(stamp_tuple) - min(stamp_tuple)) / 1e6
            print(
                f"[{frame_i}] accepted ekf stamp={left_mask_stamp} "
                f"spread={spread_ms:.1f}ms keypoints={len(order)}"
            )
        except Exception as exc:
            print(
                f"[{frame_i}] failed stamps={stamp_tuple}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    print(f"Accepted {accepted}/{len(selected_stamps)} frame(s).")


if __name__ == "__main__":
    main()
