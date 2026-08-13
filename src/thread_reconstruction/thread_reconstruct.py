import numpy as np
import cv2
import os
import pickle
import argparse
import traceback
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from thread_reconstruction.warm_start import WarmStart
from thread_reconstruction.keypt_selection import Select
from thread_reconstruction.keypt_ordering import Order
from thread_reconstruction.optim import Optim

import pdb
import time

# Drop into pdb at every error / degenerate return in this module, so a
# silently-skipped frame stops instead of being swallowed by the caller's
# fallback path.  Export THREAD_RECON_BREAK=0 to disable every breakpoint
# in the package at once (unattended runs, live ROS sessions).
DEBUG_BREAK_ON_ERROR = os.environ.get("THREAD_RECON_BREAK", "1") != "0"


# os.environ['ROS_DOMAIN_ID'] = '100'

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self, name=""):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"\033[94m --- {name} function took: {elapsed_time:0.4f} seconds\033[0m")

class FitEvalClass():
    def __init__(self, args):
        # print('python version in fit script')
        # print(sys.version)

        self.thread_file       = getattr(args, 'thread_pkl', None)
        self.thread_specs_file = getattr(args, 'thread_specs_pkl', None)
        self.cam2img1           = None
        self.cam2img2          = None
        self.trial_path        = None
        self.trial_name        = None
        self.speedy            = None
        self.ros_enable        = None
        self.hand_order        = None

        self.camera_init(args)

        self.trial_path  = getattr(args, 'trial_path', None)
        self.trial_name  = getattr(args, 'trial_name', None)
        self.speedy      = getattr(args, 'speedy', None)
        self.ros_enable  = getattr(args, 'ros_enable', None)
        self.hand_order  = getattr(args, 'hand_order', None)

        # ── Endpoint reliability taper ────────────────────────────────────────
        # The thread's ends are the least certain part of the reconstruction
        # (under-observed, prone to over/under-extension), so down-weight the
        # published reliability near each tip AND widen the depth uncertainty
        # bounds there, consistently.  Over the first/last ENDPOINT_TAPER_FRAC of
        # the thread the reliability multiplier ramps ENDPOINT_MIN_FACTOR → 1.0
        # (and the bound half-width scales by 1/that); the middle is unchanged.
        # Set FRAC = 0 to disable.
        # DISABLED (0.0): the taper widened the tip depth bounds by 1/MIN_FACTOR
        # over the outer 30% of EACH end and scaled reliability down to match.
        # It is redundant against an envelope derived from the filter's own
        # posterior covariance, which already grows where the ends are
        # under-observed — applying both compounded the same effect twice.
        # Restore by setting FRAC back to 0.30.
        self.ENDPOINT_TAPER_FRAC = 0.0    # fraction of each end that is tapered
        self.ENDPOINT_MIN_FACTOR = 0.30   # reliability multiplier at the tip
            # ------------------------------------------------------------------
    def camera_init(self, args):
        calib = args.calib or (
            os.path.dirname(__file__) + "/../../assets/camera_calibration_fei.yaml")

        cv_file   = cv2.FileStorage(calib, cv2.FILE_STORAGE_READ)
        K1        = cv_file.getNode("K1").mat()
        D1        = cv_file.getNode("D1").mat()
        K2        = cv_file.getNode("K2").mat()
        D2        = cv_file.getNode("D2").mat()
        R         = cv_file.getNode("R").mat()
        T         = cv_file.getNode("T").mat()
        ImageSize = cv_file.getNode("ImageSize").mat()

        img_size = (int(ImageSize[0][1]), int(ImageSize[0][0]))
        print("cv_file img_size", img_size)
        new_size = (640, 480)

        _, _, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=new_size)

        self.cam2img1 = self.P1[:, :-1]
        self.cam2img2 = self.P2[:, :-1]

    # ------------------------------------------------------------------
    def seek_warm_start(self, frame):
        """Iterative descent: find the latest existing warm-start file ≤ frame."""
        for f in range(frame, -1, -1):
            path = (self.trial_path
                    + self.trial_name + "_{:03d}_".format(f) + "spline.pkl")
            if Path(path).exists():
                specs_path = (self.trial_path
                              + self.trial_name + "_{:03d}_".format(f)
                              + "spline_specs.pkl")
                print(f"warm start frame: {f}\n")
                return path, specs_path
        print("Frame 0 reached and no warm start found\n")
        if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
        return None, None

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _endpoint_taper_factor(self, n):
        """Per-sample reliability multiplier in [ENDPOINT_MIN_FACTOR, 1]: it is
        ENDPOINT_MIN_FACTOR at each tip and ramps to 1.0 by ENDPOINT_TAPER_FRAC
        of the length in from each end; the interior is 1.0.  Returns None when
        the taper is disabled or the array is too short."""
        if n < 2 or self.ENDPOINT_TAPER_FRAC <= 0.0:
            return None
        u    = np.linspace(0.0, 1.0, n)
        d    = np.minimum(u, 1.0 - u) / self.ENDPOINT_TAPER_FRAC
        ramp = np.clip(d, 0.0, 1.0)
        return self.ENDPOINT_MIN_FACTOR + (1.0 - self.ENDPOINT_MIN_FACTOR) * ramp

    def taper_endpoints(self, reliability, lower_constr, upper_constr):
        """Down-weight reliability AND widen the depth bounds near the two tips,
        consistently: reliability *= factor, and the z uncertainty half-width is
        divided by the same factor (lower reliability ⇒ wider bound).  Only the
        z column (index 2) of the bounds is the uncertainty envelope — the x,y
        curve position is left untouched.  Returns (reliability, lower, upper)."""
        r = np.asarray(reliability, dtype=np.float64)
        factor = self._endpoint_taper_factor(r.size)
        if factor is None:
            return r, lower_constr, upper_constr
        r = r * factor

        lc = np.asarray(lower_constr, dtype=np.float64).copy()
        uc = np.asarray(upper_constr, dtype=np.float64).copy()
        fb = self._endpoint_taper_factor(lc.shape[0]) if lc.ndim == 2 else None
        if (fb is not None and lc.shape == uc.shape and lc.shape[1] >= 3):
            center = 0.5 * (uc[:, 2] + lc[:, 2])       # thread z
            hw     = 0.5 * (uc[:, 2] - lc[:, 2])       # z uncertainty half-width
            scale  = 1.0 / fb                          # tip: /MIN_FACTOR → wider
            lc[:, 2] = center - hw * scale
            uc[:, 2] = center + hw * scale
        return r, lc, uc

    def _apply_endpoint_taper(self, thread_specs_dict):
        """Apply taper_endpoints in place on a thread_specs dict (no-op if it
        lacks the keys)."""
        if thread_specs_dict is None:
            return thread_specs_dict
        r  = thread_specs_dict.get("reliability")
        lc = thread_specs_dict.get("lower_constr")
        uc = thread_specs_dict.get("upper_constr")
        if r is None or lc is None or uc is None:
            return thread_specs_dict
        r, lc, uc = self.taper_endpoints(r, lc, uc)
        thread_specs_dict["reliability"]  = r
        thread_specs_dict["lower_constr"] = lc
        thread_specs_dict["upper_constr"] = uc
        return thread_specs_dict

    def fit_eval(self, img1, img2, mask1, warm_thread, warm_keypts, curr_T, prev_T):
        # img1 = img1 / 255.0 if np.max(img1) == 255.0 else img1
        # img1 = self.img1 if img1 is None else img1 # to work with the py-tree-pipeline as well as through main
        # img2 = self.img2 if img2 is None else img2
        # mask1 = self.mask1_t if mask1_t is None else mask1_t

        needle_pos_file = None  # needle position pipeline currently unused
        t = Timer()
        t_full = Timer()
        t_full.start()
        t.start()
        (img_3D, __, cluster_map, keypoints,
         grow_paths, adjacents, intersection_segments, dense_pts) = \
            Select.keypt_selection(img1, img2, mask1, self.Q, speedy=self.speedy)
        t.stop("keypt_selection")
        # warm_keypoints, order, new_warm_keypts = Order.warm_ordering(
        #     mask1, keypoints,
        #     P1=self.P1,
        #     warm_thread=Warmstart.trans_thread,
        #     prev_keypts=Warmstart.prev_keypts,
        #     curr_T=Warmstart.curr_T,
        #     speedy=self.speedy,
        #     adjacents=adjacents,
        #     intersection_segments=intersection_segments,
        #     dense_pts=dense_pts,
        # )
        # print(f"previous keypts \n  {Warmstart.prev_keypts}")
        t.start()
        warm_keypoints, order, new_warm_keypts = \
            Order.run_warm_ordering_with_ekf(mask1, keypoints,
                                             P1=self.P1,
                                             warm_thread=warm_thread,
                                             prev_keypts=warm_keypts,
                                             curr_T=curr_T,
                                             prev_T=prev_T, 
                                             speedy=self.speedy,
                                             adjacents=adjacents,
                                             intersection_segments=intersection_segments,
                                             dense_pts=dense_pts,
                                             keypt_conf=getattr(
                                                 Select, 'last_keypt_conf', None),
                                            )
        t.stop("warm_ordering_with_ekf")

        if warm_keypoints is None or self.hand_order:
            if self.hand_order:
                keypoints, order = Order.hand_ordering(
                    img1, img_3D, keypoints, needle_pos_file, self.P1)
            else:
                t.start()
                __, keypoints, __, order = Order.keypt_ordering(
                    img1, img_3D, cluster_map, keypoints, grow_paths,
                    adjacents, intersection_segments=intersection_segments, speedy=self.speedy)
                t.stop("keypt_ordering")
            t.start()
            # warm_thread is the previous reconstruction ALREADY motion-compensated
            # to this frame (WarmStart.refresh_warm_start applies the same warp as
            # SplineEKF.predict()), so it is used directly as the temporal prior —
            # re-applying trans here would double-count the tool motion.
            thread_dict, thread_specs_dict = Optim.optim(
                mask1, keypoints, order,
                self.cam2img1, self.P1, self.P2, needle_pos_file, speedy=self.speedy,
                x_prior_thread=warm_thread)
            t.stop("optim (no warm start)")
        else:
            t.start()
            thread_dict, thread_specs_dict = Optim.optim_warm_start(
                mask1, warm_keypoints, order, self.cam2img1, self.P1,
                warm_thread=warm_thread,
                warm_keypts=new_warm_keypts,
                speedy=self.speedy,
                ros_enable=self.ros_enable,
                needle_pos_file=needle_pos_file,
            )
            t.stop("optim_warm_start")
        t_full.stop("full fit eval")
        if not self.hand_order:
            thread_dict, thread_specs_dict = Optim.match_warm_order(img1,
                thread_dict, thread_specs_dict,
                warm_thread=warm_thread, P=self.P1)

        # Down-weight reliability + widen depth bounds near the ends.
        thread_specs_dict = self._apply_endpoint_taper(thread_specs_dict)

        return thread_dict, thread_specs_dict

    # ------------------------------------------------------------------
    def prep_masks(self, frame, curr_T, trans_thread, dist_thresh=20, speedy=False):
        lr   = ["left_rgb/", "right_rgb/"]
        ext  = ".png"
        stem = self.trial_name + "_{:03d}".format(frame)

        imfile1 = self.trial_path + lr[0] + stem + ext
        imfile2 = self.trial_path + lr[1] + stem + ext
        # print(f"img1 path {imfile1}")

        t_mask_1 = Path(self.trial_path + lr[0] + stem + "_mask" + ext)
        t_mask_2 = Path(self.trial_path + lr[0] + "binary_masks/" + stem + ext)

        if t_mask_1.exists():
            m1_path = str(t_mask_1)
            m2_path = self.trial_path + lr[1] + stem + "_mask" + ext
        elif t_mask_2.exists():
            m1_path = str(t_mask_2)
            m2_path = self.trial_path + lr[1] + "binary_masks/" + stem + ext
        else:   
            raise FileNotFoundError(
                f"Mask not found:\n  {t_mask_1}\n  {t_mask_2}")

        img1 = cv2.cvtColor(cv2.imread(imfile1), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(imfile2), cv2.COLOR_BGR2RGB)
        mask1   = cv2.cvtColor(cv2.imread(m1_path),  cv2.COLOR_BGR2GRAY)
        mask2   = cv2.cvtColor(cv2.imread(m2_path),  cv2.COLOR_BGR2GRAY)
        
        if trans_thread is not None:
            mask1, mask2 = self.clip_mask(mask1, mask2, self.P1, self.P2, curr_T, trans_thread, dist_thresh=dist_thresh, speedy=speedy)

        # Apply segmentation masks and convert to float32 in one step
        img1 = np.where(np.stack([mask1]*3, axis=-1) > 0, img1, 0).astype(np.uint32)
        img2 = np.where(np.stack([mask2]*3, axis=-1) > 0, img2, 0).astype(np.uint32)

        return img1, img2, mask1, mask2
    
    def clip_mask(self, mask1, mask2, P1, P2, curr_T, trans_thread, dist_thresh=20, clip_radius=80, speedy=False,
                  grasp_window_px=None):
        """
        grasp_window_px : None → original behaviour (keep the needle→gripper
                          span of the warm spline).  A number → EXPERIMENTAL
                          grasp-local mode: keep only the section of the thread
                          within ±grasp_window_px of ARC LENGTH (measured along
                          the projected warm spline, in image pixels, per
                          camera) around the point the gripper is grasping.  If
                          the thread is not grasped (no spline point within
                          dist_thresh of the gripper) the masks are returned
                          unclipped.
        """
        lin_keypts     = np.linspace(0, 1, 100)
        warm_keypoints = trans_thread(lin_keypts)
        dists = np.linalg.norm(warm_keypoints - curr_T[:3, 3], axis=1)
        min_dist = np.min(dists)

        # keep points before the gripper (ordered from needle)
        if min_dist < dist_thresh:
            idx_gripper = np.argmin(dists)
            if grasp_window_px is not None:
                # grasp-local mode: project the FULL spline, window per camera
                # below (after the projection block) by arc length around the
                # grasp point.
                thread_kept = warm_keypoints
            else:
                thread_kept = warm_keypoints[:idx_gripper+1]
                # If the gripper matches near t=0 the "before gripper" slice is a
                # handful of points — that means the warm spline's direction is
                # flipped or stale, and clipping to it would erase the mask.  Fall
                # back to the whole warm spline as the clip region.
                if idx_gripper < len(warm_keypoints) // 10:
                    print(f"clip_mask: gripper matched at t~{idx_gripper/len(warm_keypoints):.2f} "
                          "(near t=0 — warm direction flipped/stale?); "
                          "clipping to the FULL warm spline instead.")
                    thread_kept = warm_keypoints
        else:
            print(f"no valid gripper pose found, closest point is {min_dist}"
                  + (" — thread not grasped, not clipping"
                     if grasp_window_px is not None else ""))
            return mask1, mask2

        # gripper projections
        aug_curr_T  = np.append(curr_T[:3, 3], 1.0)
        left_proj_curr_T = P1 @ aug_curr_T
        left_proj_curr_T /= left_proj_curr_T[2] + 1e-7
        left_proj_curr_T  = left_proj_curr_T[[0, 1, 2]]

        right_proj_curr_T = P2 @ aug_curr_T
        right_proj_curr_T /= right_proj_curr_T[2] + 1e-7
        right_proj_curr_T  = right_proj_curr_T[[0, 1, 2]]

        # thread projections
        aug_pts  = np.concatenate(
            (thread_kept, np.ones((len(thread_kept), 1))), axis=1)
        left_proj_pts = (P1 @ aug_pts.T).T
        left_proj_pts /= left_proj_pts[:, 2:] + 1e-7
        left_proj_pts[:, 2] = thread_kept[:, 2]
        left_proj_pts        = np.asarray(left_proj_pts[:, [1, 0, 2]])

        right_proj_pts = (P2 @ aug_pts.T).T
        right_proj_pts /= right_proj_pts[:, 2:] + 1e-7
        right_proj_pts[:, 2] = thread_kept[:, 2]
        right_proj_pts        = np.asarray(right_proj_pts[:, [1, 0, 2]])

        if grasp_window_px is not None:
            # ── EXPERIMENTAL grasp-local window ────────────────────────────────
            # Keep only the spline points within ±grasp_window_px of arc length
            # (along the PROJECTED thread, so "px" means image pixels) of the
            # grasp point, independently per camera.
            def arc_window(proj_pts):
                seg = np.linalg.norm(np.diff(proj_pts[:, :2], axis=0), axis=1)
                cum = np.concatenate([[0.0], np.cumsum(seg)])
                keep = np.abs(cum - cum[idx_gripper]) <= grasp_window_px
                keep[idx_gripper] = True          # grasp point always kept
                return proj_pts[keep]
            left_proj_pts  = arc_window(left_proj_pts)
            right_proj_pts = arc_window(right_proj_pts)
            print(f"clip_mask[grasp]: keeping ±{grasp_window_px}px of thread "
                  f"around grasp (t~{idx_gripper/len(warm_keypoints):.2f}): "
                  f"{len(left_proj_pts)}/{len(warm_keypoints)} left, "
                  f"{len(right_proj_pts)}/{len(warm_keypoints)} right spline pts.")

        # draw clip mask — union of disks of clip_radius around the projected
        # spline points, rasterised with cv2.circle (C-speed) instead of a
        # full-image distance field per point (was ~100 × H×W float ops).
        def draw_mask(image, points):
            h, w = image.shape[:2]
            clip_mask = np.zeros((h, w), np.uint8)
            pts_xy = np.round(points[:, [1, 0]]).astype(np.int32)  # (col, row)
            for x_c, y_c in pts_xy:
                cv2.circle(clip_mask, (int(x_c), int(y_c)), int(clip_radius), 1, -1)
            return clip_mask
        left_clip = draw_mask(mask1, left_proj_pts)
        right_clip = draw_mask(mask2, right_proj_pts)

        # Keep the uint8 0/255 mask convention: downstream allocates arrays
        # with np.zeros_like(mask) (e.g. cluster_map in keypt_selection) — a
        # boolean mask there silently collapses all cluster ids to True/1 and
        # destroys the adjacency graph / cold ordering.
        new_mask1 = (np.logical_and(mask1==255, left_clip > 0)
                     .astype(np.uint8) * 255)
        new_mask2 = (np.logical_and(mask2==255, right_clip > 0)
                     .astype(np.uint8) * 255)

        # ── Retention guard ────────────────────────────────────────────────────
        # A stale/flipped warm spline can clip away nearly the whole mask, which
        # starves selection (a handful of keypoints), produces a garbage
        # reconstruction, and feeds an even worse warm spline next frame — a
        # death spiral.  If clipping keeps too little of the original mask,
        # skip it this frame and let the pipeline see the full mask.
        # In grasp-local mode keeping only a small section IS the intent, so the
        # fraction guard is replaced by an absolute floor: enough pixels must
        # survive for keypt_selection to work at all.
        orig1 = max(int(np.count_nonzero(mask1 == 255)), 1)
        orig2 = max(int(np.count_nonzero(mask2 == 255)), 1)
        frac1 = np.count_nonzero(new_mask1) / orig1
        frac2 = np.count_nonzero(new_mask2) / orig2
        if grasp_window_px is not None:
            MIN_KEEP_PX = 100
            kept1 = int(np.count_nonzero(new_mask1))
            kept2 = int(np.count_nonzero(new_mask2))
            if kept1 < MIN_KEEP_PX or kept2 < MIN_KEEP_PX:
                print(f"clip_mask[grasp]: only {kept1}/{kept2} mask px inside "
                      f"the grasp window (< {MIN_KEEP_PX}); warm spline likely "
                      "off the thread — skipping clip this frame.")
                return mask1, mask2
            print(f"clip_mask[grasp]: kept {frac1:.2f}/{frac2:.2f} of the "
                  f"masks ({kept1}/{kept2} px).")
        else:
            MIN_KEEP_FRAC = 0.30
            if frac1 < MIN_KEEP_FRAC or frac2 < MIN_KEEP_FRAC:
                print(f"clip_mask: clipping would keep only "
                      f"{frac1:.2f}/{frac2:.2f} of the masks "
                      f"(< {MIN_KEEP_FRAC}); warm spline likely stale — "
                      "skipping clip this frame.")
                return mask1, mask2

        if not speedy:
            # uint8 0/1 → 1.0/NaN so the overlay stays transparent off-disk
            left_clip  = np.where(left_clip  > 0, 1.0, np.nan)
            right_clip = np.where(right_clip > 0, 1.0, np.nan)
            fig = plt.figure(figsize=(12, 6))
            ax_l = fig.add_subplot(1, 2, 1)
            ax_l.imshow(mask1, cmap="gray")
            ax_l.imshow(left_clip, cmap='jet', alpha=0.7)
            # Warm spline in 2-D projection
            ax_l.plot(left_proj_pts[:, 1], left_proj_pts[:, 0],
                    c='red', lw=1.2, alpha=0.5, label='warm spline (2D)')
            # Mark t=0 and t=1 ends
            ax_l.scatter(left_proj_pts[0,  1], left_proj_pts[0,  0], c='red', s=80, marker='^', zorder=6, label='t=0')
            ax_l.scatter(left_proj_pts[-1, 1], left_proj_pts[-1, 0], c='red', s=80, marker='v', zorder=6, label='t=1')

            ax_l.set_title(f'left clipped mask, r={clip_radius}'); ax_l.legend(fontsize=7)

            ax_l.scatter(left_proj_curr_T[0], left_proj_curr_T[1], c='red', s=60, marker='x',
                            zorder=5, label='curr_T')

            ax_r = fig.add_subplot(1, 2, 2)
            ax_r.imshow(mask1, cmap="gray")
            ax_r.imshow(right_clip, cmap='jet', alpha=0.5)

            # Warm spline in 2-D projection
            ax_r.plot(right_proj_pts[:, 1], right_proj_pts[:, 0], c='red', lw=1.2, alpha=0.5, label='warm spline (2D)')
            # Mark t=0 and t=1 ends
            ax_r.scatter(right_proj_pts[0,  1], right_proj_pts[0,  0], c='red', s=80, marker='^', zorder=6, label='t=0')
            ax_r.scatter(right_proj_pts[-1, 1], right_proj_pts[-1, 0], c='red', s=80, marker='v', zorder=6, label='t=1')

            ax_r.set_title(f'right clipped mask, r={clip_radius}'); ax_r.legend(fontsize=7)

            ax_r.scatter(right_proj_curr_T[0], right_proj_curr_T[1], c='red', s=60, marker='x', zorder=5, label='curr_T')
            
            plt.tight_layout(); plt.show()

            fig = plt.figure(figsize=(12, 6))
            ax_l = fig.add_subplot(1, 2, 1)
            ax_l.imshow(new_mask1, cmap="gray")
            ax_r = fig.add_subplot(1, 2, 2)
            ax_r.imshow(new_mask2, cmap="gray")
            plt.tight_layout(); plt.show()
        
        return new_mask1, new_mask2
    # ------------------------------------------------------------------
    def save_spline(self, thread, thread_specs):
        if input("save spline from this trial? (y) ") == 'y':
            with open(self.thread_file, "wb") as f:
                print("saving", self.trial_name, "thread\n")
                pickle.dump(thread, f)
            with open(self.thread_specs_file, "wb") as f:
                print("saving", self.trial_name, "thread specs\n")
                pickle.dump(thread_specs, f)

    # ------------------------------------------------------------------
    def main(self, frame=0, needle_mask=None):
        t = Timer()
        t.start()
        img1, img2, mask1, mask2 = \
            self.prep_masks(frame, Warmstart.curr_T, Warmstart.trans_thread, dist_thresh=20, speedy=args.speedy)
        t.stop("prep_mask")
        t.start()
        thread, thread_specs = self.fit_eval(img1=img1, img2=img2, mask1=mask1, frame=frame, needle_mask=needle_mask)

        self.save_spline(thread, thread_specs)

# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_path', default=None)
    parser.add_argument('--trial_name', default=None)
    parser.add_argument('--frame',            type=int, default=0)
    parser.add_argument('--thread_pkl',      type=str, default=None)
    parser.add_argument('--thread_specs_pkl',type=str, default=None)
    parser.add_argument('--speedy',           action="store_true")
    parser.add_argument('--ros_enable',       action='store_true')
    parser.add_argument('--calib',            default=None)
    parser.add_argument('--psm_calibrate',
        default=os.path.dirname(__file__) + "/../../../RaftStereo/assets/psm_calibration.npz")
    parser.add_argument('--hand_order',       action='store_true')
    args = parser.parse_args()

    fit_eval  = FitEvalClass(args)
    Select    = Select(args)
    Order     = Order(args)
    Optim     = Optim(args)
    Warmstart = WarmStart(args)

    prev_thread, prev_keypts = Warmstart._add_prev_thread(args.thread_pkl)
    # prev_thread, prev_keypts = None, None
    if prev_thread is not None:
        Warmstart.init_ros(args)
        use_goal_file = input("use goal file over pose file? y ") == 'y'
        curr_T, prev_T, trans = Warmstart.update_prev_to_curr(args.thread_pkl, use_goal_file=use_goal_file)
        Warmstart.refresh_warm_start(prev_thread=prev_thread, 
                                    prev_keypts=prev_keypts, 
                                    curr_T=curr_T, 
                                    prev_T=prev_T, 
                                    )
        
    try:
        fit_eval.main(frame=args.frame)
    except Exception as e:
        print(f"Caught error: {e}")
        traceback.print_exc()