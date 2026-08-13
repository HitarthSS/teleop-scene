# from Cython import const
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
# import scipy.optimize
# import scipy.integrate
import scipy.interpolate as interp
from scipy.sparse import csc_matrix
from scipy.stats import linregress
import osqp
import copy

from thread_reconstruction.utils import *
from thread_reconstruction.reparam import refit_spline, reparam
from thread_reconstruction import ekf_params
import pickle
import pdb

import time
import os

# Drop into pdb at every error / degenerate return in this module, so a
# silently-skipped frame stops instead of being swallowed by the caller's
# fallback path.  Export THREAD_RECON_BREAK=0 to disable every breakpoint
# in the package at once (unattended runs, live ROS sessions).
DEBUG_BREAK_ON_ERROR = os.environ.get("THREAD_RECON_BREAK", "1") != "0"


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self, enabled=True):
        self._start_time = None
        self.enabled = enabled

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self, name=""):
        """Stop the timer, and report the elapsed time (silent if disabled)"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        if self.enabled:
            print(f"{name} function took: {elapsed_time:0.4f} seconds")

CONSTR_WIDTH_2D = 6

# Per-keypoint constraint-box softening by warm-match quality (the
# keypt_quality arg, Order._match_quality in keypt_ordering).  Every box
# half-width — the 2-D pixel box AND the z bounds — is multiplied by
#   1 + QUALITY_BOX_WIDEN_GAIN * (1 - q),
# so a cleanly matched keypoint (q≈1) keeps the tuned tight box while a
# poorly matched one (ambiguous arm, crossing neighbourhood, big EKF
# residual, q→0) gets up to a (1+GAIN)x wider box: it guides the fit instead
# of dictating it with the same authority as the good points.  0 disables.
QUALITY_BOX_WIDEN_GAIN = 2.0

# Temporal (Tikhonov) prior weight used by Optim.optim().  The prior pulls the
# spline's control points toward the previous frame's (motion-compensated)
# thread.  It is *dimensionless*: the effective coefficient is
#   lambda_temporal * mean_diag(bending_energy),
# so lambda_temporal ≈ 1 makes the prior comparable to the average bending term,
# and 0 disables it entirely.  Keep it modest — too large freezes the thread and
# stops it tracking real motion.
TEMPORAL_LAMBDA = 0.2  # temporal prior disabled for now — set back to ~0.2 to re-enable

# e-folding distance (3-D spline units) for the distance-weighted rotation in the
# motion-compensation warp, so the temporal prior is compensated identically to
# the EKF predict step.  Formerly one of THREE hand-synced copies (with
# SplineEKF(deform_radius=...) and WarmStart.deform_radius — they silently
# disagreed at 50/30/30 until 2026-07-26); now all three read the single
# constant in ekf_params.py.  The name is kept as optim's public alias.
TEMPORAL_DEFORM_RADIUS = ekf_params.DEFORM_RADIUS

# OSQP convergence tolerance / iteration cap.  The default eps (1e-3) is far
# tighter than the ±CONSTR_WIDTH_2D px constraint boxes justify, and on real
# frames the solver burns its full 4000-iteration budget per QP round
# ("maximum iterations reached" → 300-500 ms per optim call).  2e-2 converges
# ~10x faster with identical constraint satisfaction (verified: max
# spline→keypoint px distance unchanged).  Tighten if the spline looks
# under-smoothed; loosen further for more speed.
OSQP_EPS      = 4e-2
OSQP_MAX_ITER = 2000

# OSQP exit statuses whose solution may be USED.  'maximum iterations reached'
# returns whatever iterate the solver stopped on: finite, so it slips past the
# isfinite check, but with no accuracy guarantee whatsoever.  One such solve
# published a spline with bend radius 11901 and control points ~500 km from the
# grasp, which then poisoned prev_thread and the EKF permanently.  Only the
# solve that produces the RETURNED thread is judged — earlier rounds of the
# reparameterise→QP loop may miss and still be useful, since their output is
# only the next round's warm start.
OSQP_OK_STATUS = ('solved', 'solved inaccurate')

# What the published per-point `reliability` in thread_specs actually means:
#
#   'geometry'    — LEGACY.  Derived only from the z-bound width, i.e. how far
#                   a keypoint's depth sits from the local linear trend of its
#                   neighbours.  That is curve SMOOTHNESS, not measurement
#                   quality: a keypoint whose stereo match was ambiguous but
#                   which happens to land on the trend scores high.  Worse, the
#                   EKF's z-denoising overwrites keypoint z with the posterior
#                   spline z before optim runs, so depths are smoother by
#                   construction and this metric is inflated exactly where the
#                   FILTER was confident rather than where the SENSOR was.
#   'measurement' — the stereo matcher's own confidence (sigmoid of the margin
#                   between best and second-best SSD, averaged over the cluster's
#                   pixels).  Computed in keypt_selection straight from the
#                   images, so it is immune to the denoising feedback loop.
#   'combined'    — product of the two.  Flags both failure modes (ambiguous
#                   match AND depth outlier); the strictest, and the one to
#                   prefer once 'measurement' has been sanity-checked.
#
# Falls back to 'geometry' on any frame that supplies no keypt_conf.
RELIABILITY_MODE = 'measurement'

# DATA-GAP degradation of the published reliability.  Reliability is defined
# per ordered KEYPOINT and linearly interpolated over the 200 published
# samples, so a spline stretch with NO keypoints (occlusion, dropped
# clusters, detangle clip) would inherit its bracketing keypoints'
# confidence unchanged — high, despite carrying no data of its own.  Instead
# each published sample is scaled by exp(-max(0, d − deadband) / DECAY),
# where d is its arc-length distance (normalized s) to the nearest
# supporting keypoint and deadband is the median keypoint spacing — so
# normally-supported stretches are untouched and only genuine gaps decay.
# 0.05 → reliability falls to 1/e one-twentieth of the thread past a normal
# gap.  Smaller = punish gaps harder; None/≤0 = off.
GAP_RELIAB_DECAY_S = 0.05

# Outer reparameterise→QP loop.  Each round re-fits the arc-length
# parameterisation (reparam, ~16 ms) and re-solves the QP (~5-25 ms), so every
# round saved is real time.  The loop already early-exits once the sampled
# curve moves less than OPTIM_CONV_TOL (3-D units) between rounds; most frames
# converge in 2-3 rounds, so OPTIM_MAX_ITER=5 rarely bound quality.  Lower the
# cap or raise the tolerance to trade a little accuracy for speed.
OPTIM_MAX_ITER = 5
OPTIM_CONV_TOL = 0.06

# Fitted-spline control-point count = max(CTRL_MIN, num_ordered_keypoints //
# CTRL_DIVISOR).  The spline used to carry one control point per keypoint (an
# interpolating fit that scaled the QP with keypoint count and chased noise);
# this makes it a smoothing least-squares fit whose cost is bounded by the
# thread's length, not its point density.  Lower CTRL_DIVISOR (toward 1) or
# raise CTRL_MIN for more detail/wiggle; raise CTRL_DIVISOR for smoother/faster.
# num_ctrl = max(CTRL_MIN, num_order // CTRL_DIVISOR).  A cubic B-spline cannot
# bend tighter than its control-point spacing, so MORE control points = tighter
# achievable bends.  Lowered the divisor 3->2 to roughly 1.5x the control points
# for a given keypoint count, letting the fit follow sharper curves.  Trade-off:
# more DOF also means more freedom to wiggle between keypoints — if the thread
# starts looking wavy, raise this back toward 3 or narrow CONSTR_WIDTH_2D.
# Raised 2 -> 3 because CTRL_DIVISOR only feeds optim_init, i.e. the COLD
# seed frame (the warm path inherits warm_thread.c).  At 2 that frame fit 70
# keypoints with max(16, 70//2) = 35 control points — ~5.4 units of spacing on
# a ~190-unit thread — while the EKF's own length adapt converges to 16-20
# (run.log: 30→26→23→21→20→16).  With 2x the DOF the pipeline actually wants,
# the seed fit chased per-keypoint DEPTH noise: its 2-D was excellent
# (reprojection 0.97 on mask) but its 3-D arc length came out 283.4 against
# the ~190 every later frame measures, i.e. ~49% of the length was z zigzag.
# The EKF then seeds from that curve, and the ends inherit the excursion.
# 3 gives max(16, 70//3) = 23 on the same frame, inside the adapt's range.
CTRL_DIVISOR = 3
CTRL_MIN     = 16

class Optim():
    # TODO add the init and remove the methods that should be covered in warm-start
    def __init__(self, args):
        # --time: print Timer breakdowns even in speedy mode
        self.timing = getattr(args, 'time', False)

    def optim(self, mask1_t, keypoints, order, cam2img, P1, P2, needle_pos_file=None, speedy=False,
              x_prior_thread=None, trans=None, tool_pos_3d=None,
              lambda_temporal=TEMPORAL_LAMBDA, deform_radius=TEMPORAL_DEFORM_RADIUS,
              keypt_conf=None, keypt_quality=None):
        """
        keypt_conf     : (len(keypoints),) per-keypoint stereo matching
                         confidence in [0,1], aligned with `keypoints` and
                         indexed by `order` like keypoints[order].  Feeds the
                         published per-point reliability — see
                         RELIABILITY_MODE.  None → 'geometry' fallback.
        keypt_quality  : (len(keypoints),) per-keypoint warm-MATCH quality in
                         [0,1], same alignment/indexing as keypt_conf (from
                         Order._match_quality).  Widens each keypoint's
                         constraint box by 1+QUALITY_BOX_WIDEN_GAIN·(1−q) so
                         poorly matched keypoints can't dictate the fit.
                         None → all boxes keep their tuned width.
        x_prior_thread : previous frame's spline callable (BSpline/CubicSpline,
                         t ∈ [t0, t1] → (·, 3)), in the SAME 3-D frame this optim
                         solves in (i.e. a prior optim/warm_start output).  Used
                         to build the temporal prior x_prior on the control
                         points.  None → no temporal regularization.
                         NOTE: the canonical caller passes WarmStart.trans_thread,
                         which is ALREADY motion-compensated to the current frame
                         (refresh_warm_start applies the same warp as
                         SplineEKF.predict()).  In that case leave trans=None.
        trans          : (4, 4) relative tool transform between frames
                         (curr_T @ inv(prev_T)), matching SplineEKF.predict().
                         Only pass this when x_prior_thread is a RAW (un-warped)
                         previous thread that still needs motion compensation —
                         otherwise the motion is double-counted.  None → prior
                         used as-is.
        tool_pos_3d    : (3,) current tool-tip position (curr_T[:3, 3]); scales
                         the distance-weighted rotation in the warp.
        lambda_temporal: dimensionless temporal-prior weight (see TEMPORAL_LAMBDA).
        deform_radius  : e-folding distance for the rotation weighting in the warp.
        """
        t = Timer(enabled=self.timing or not speedy)
        t_step = Timer(enabled=self.timing or not speedy)
        t.start()
        # print("using optim non-warm start")
        # Get necessary values
        # init_pts, keypoint_idxs = augment_keypoints(img1, segpix1, img_3D, keypoints, grow_paths, order)
        # mask1 = mask1_t + mask1_n
        # mask2 = mask2_t + mask2_n
        # pdb.set_trace()
        if order is None or len(order) < 4:
            print(f"\033[38;2;255;165;0moptim: only "
                  f"{0 if order is None else len(order)} ordered keypoints, "
                  f"skipping frame\033[0m")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None
        init_pts, keypoint_idxs = keypoints[order], np.arange(len(order))

        '''
        ###
        init_pts: is just the keypoints in order
        keypoint_idxs: array from 0 to # of points in order
        knots: linearly spaced 20 points along the spline plus 3 extra points at the end points. ie [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
        init_u: initialize cumsum distances between each init_pts (ordered keypoints), transformed into camera coordinates
        constr_lower_d, constr_upper_d: calculated uncertainty based on the point's diviation from its nearby keypoints. The more similar, the less uncertainty. 
        ###
        '''

        knots, init_u, constr_lower_d, constr_upper_d = self.optim_init(init_pts, keypoints, keypoint_idxs, order, cam2img)
        if knots is None or len(knots) == 0:
            print("\033[38;2;255;165;0mknots is None\033[0m")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None
        if not (np.all(np.isfinite(knots)) and np.all(np.isfinite(init_u))):
            # Degenerate ordering (e.g. coincident keypoints → zero-length
            # parameterization). Skip this frame instead of feeding NaNs to
            # the QP solver / BSpline.design_matrix.
            print("\033[38;2;255;165;0moptim: non-finite knots/init_u, skipping frame\033[0m")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None
        t.stop("[optim] optim init")

        # ── Constraint-box softening by match quality ─────────────────────────
        # (num_order,) half-width multiplier applied to BOTH the z bounds
        # (here, before the needle override so that pin stays exact) and the
        # 2-D pixel boxes (at their build below).  See QUALITY_BOX_WIDEN_GAIN.
        box_widen = np.ones(len(order))
        if keypt_quality is not None and QUALITY_BOX_WIDEN_GAIN > 0:
            kq = np.asarray(keypt_quality, dtype=float)
            if kq.size == len(keypoints):
                kq = np.clip(kq[np.asarray(order)], 0.0, 1.0)
                box_widen = 1.0 + QUALITY_BOX_WIDEN_GAIN * (1.0 - kq)
                d_ctr  = 0.5 * (constr_upper_d + constr_lower_d)
                d_half = 0.5 * (constr_upper_d - constr_lower_d)
                constr_lower_d = d_ctr - d_half * box_widen
                constr_upper_d = d_ctr + d_half * box_widen
                print(f"optim: quality box-widen x[{box_widen.min():.2f}, "
                      f"{box_widen.max():.2f}]  "
                      f"({int((box_widen > 1.5).sum())}/{len(box_widen)} "
                      f"keypoints widened >1.5x)")
            else:
                print(f"\033[38;2;255;165;0moptim: keypt_quality has "
                      f"{kq.size} entries but keypoints has "
                      f"{len(keypoints)} — ignoring quality\033[0m")

        # bring in fixed point
        t.start()
        if needle_pos_file is not None:
            point = self.get_needle_point(needle_pos_file)
            aug_pts = np.append(point, 1)
            proj_pts = (P1 @ aug_pts.T).T
            proj_pts /= proj_pts[2].copy() + 1e-7
            proj_pts = [proj_pts[1], proj_pts[0], point[2]]

            keypoints[order[-1]] = proj_pts
            constr_lower_d[order[-1]] = proj_pts[2] - 1e-1
            constr_upper_d[order[-1]] = proj_pts[2] + 1e-1

        keypt_u = init_u[keypoint_idxs] # keypoint_idxs is just 0...len(order)
        k = 3
        num_ctrl = len(knots)-k-1
        num_constr = len(keypt_u)*3
        thread = None

        constr_centers = keypoints[order]
        width_2d = CONSTR_WIDTH_2D * box_widen         # (num_order,) per-point
        constr_lower_px = constr_centers[:, 1] - width_2d
        constr_upper_px = constr_centers[:, 1] + width_2d
        constr_lower_py = constr_centers[:, 0] - width_2d
        constr_upper_py = constr_centers[:, 0] + width_2d

        # Put bounds in a good shape
        # x constr
        constr_lower_px_rshp = np.repeat(
            constr_lower_px, num_ctrl*3
        ).reshape(num_constr//3, num_ctrl*3)
        constr_upper_px_rshp = np.repeat(
            constr_upper_px, num_ctrl*3
        ).reshape(num_constr//3, num_ctrl*3)

        # y constr
        constr_lower_py_rshp = np.repeat(
            constr_lower_py, num_ctrl*3
        ).reshape(num_constr//3, num_ctrl*3)
        constr_upper_py_rshp = np.repeat(
            constr_upper_py, num_ctrl*3
        ).reshape(num_constr//3, num_ctrl*3)
        # knots, num_ctrl, k = thread.t, len(thread.c), thread.k

        
        def QP_step(init_guess, knots, keypt_s):
            # Set up optimization...
            solver = osqp.OSQP()

            # Create objective function
            deriv_coeff = (
                self.get_deriv_matrix(knots[2:-2], num_ctrl-2, k-2) @
                self.get_deriv_matrix(knots[1:-1], num_ctrl-1, k-1) @
                self.get_deriv_matrix(knots, num_ctrl, k)
            )
            weight_coeff = np.diag(
                np.repeat(knots[4:-3] - knots[3:-4], 3)
            )
            loss_coeff = (
                deriv_coeff.T @
                weight_coeff @
                deriv_coeff
            )

            # --- Temporal (shape) prior ---------------------------------------
            # Pull the control-polygon EDGE VECTORS toward the previous frame's
            # (see _temporal_prior_terms — translation-invariant, so real thread
            # motion is free).  reg is scaled by the bending energy so
            # lambda_temporal is dimensionless (see TEMPORAL_LAMBDA).
            q_lin = np.zeros(num_ctrl * 3)
            if lambda_temporal > 0 and x_prior_thread is not None:
                x_prior = self._temporal_prior_ctrl(
                    x_prior_thread, knots, num_ctrl, k,
                    trans=trans, tool_pos_3d=tool_pos_3d,
                    deform_radius=deform_radius)
                if x_prior is not None:
                    p_scale = np.trace(loss_coeff) / (num_ctrl * 3)
                    reg = lambda_temporal * max(p_scale, 1e-9)
                    P_add, q_lin = self._temporal_prior_terms(
                        x_prior, num_ctrl, reg)
                    loss_coeff = loss_coeff + P_add

            # --- OPTIMIZATION 1: C-compiled B-Spline Design Matrix ---
            # Replaces the `for i in range(num_ctrl): basis = interp.BSpline(...)` loop
            valid_min = knots[k]
            valid_max = knots[-k-1]
            
            # 2. Clip evaluation points to completely eliminate floating-point noise out-of-bounds
            # Subtracting 1e-10 prevents SciPy right-boundary bugs without altering math precision
            safe_keypt_s = np.clip(keypt_s, valid_min, valid_max - 1e-10)
            
            B = interp.BSpline.design_matrix(safe_keypt_s, knots, k).toarray()            
            # --- OPTIMIZATION 2: Kronecker Product ---
            # The old code created a massive 3Nx3N block diagonal matrix `cam2img_rep` 
            # and multiplied it by `spl_bases`. Mathematically, this is identical to a Kronecker product.
            spl_eval_matrix = np.kron(B, cam2img)
            
            # --- OPTIMIZATION 3: Array Slicing ---
            # The old code used Identity matrices (I_constr) to slice rows via matrix multiplication. 
            # Standard NumPy slicing accomplishes the exact same thing instantly.
            eval_select_x = spl_eval_matrix[0::3]
            eval_select_y = spl_eval_matrix[1::3]
            eval_select_z = spl_eval_matrix[2::3]

            x_lower = constr_lower_px_rshp * eval_select_z - eval_select_x
            x_upper = eval_select_x - constr_upper_px_rshp * eval_select_z
            y_lower = constr_lower_py_rshp * eval_select_z - eval_select_y
            y_upper = eval_select_y - constr_upper_py_rshp * eval_select_z
            z_lower = eval_select_z
            z_upper = eval_select_z

            constr_A = np.concatenate((x_lower, x_upper, y_lower, y_upper, z_lower), axis=0)
            constr_l = np.ones(num_constr//3*5) * (-np.inf)
            constr_l[-num_constr//3:] = constr_lower_d
            constr_u = np.zeros_like(constr_l)
            constr_u[-num_constr//3:] = constr_upper_d

            solver.setup(P=csc_matrix(loss_coeff), q=q_lin, A=csc_matrix(constr_A), l=constr_l, u=constr_u, verbose=False,
                         eps_abs=OSQP_EPS, eps_rel=OSQP_EPS, max_iter=OSQP_MAX_ITER)
            if init_guess is not None:
                solver.warm_start(x=init_guess)

            result = solver.solve()
            if result.info.status != 'solved':
                print(f"OSQP status: '{result.info.status}' "
                      f"(iter={result.info.iter})")
            # OSQP can return None or a None-filled object array on a
            # primal/dual-infeasible exit; coerce to a float array (None → NaN)
            # or None so the caller's np.isfinite() check is always valid and
            # never raises "ufunc 'isfinite' not supported" on an object array.
            x = result.x
            if x is None:
                if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
                return None, result.info.status
            try:
                return np.asarray(x, dtype=float), result.info.status
            except (TypeError, ValueError):
                if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
                return None, result.info.status
        t.stop("[optim] setup")
        t.start()
        iter = OPTIM_MAX_ITER
        # Early-exit once the sampled curve moves less than CONV_TOL (3-D units)
        # between successive QP rounds (see OPTIM_MAX_ITER / OPTIM_CONV_TOL).
        CONV_TOL = OPTIM_CONV_TOL
        prev_samples = None
        # Safe default: if the loop below never runs, `None` is not in
        # OSQP_OK_STATUS, so the frame is skipped rather than published.
        qp_status    = None
        for i in range(iter):
            if i == 0:
                knots, keypt_s = knots, keypt_u
                init_guess = None
            else:
                t_step.start()
                try:
                    new_thread, knots, keypt_s = reparam(thread, keypt_u)
                except ValueError as e:
                    print(f"\033[38;2;255;165;0moptim: {e} — skipping frame\033[0m")
                    if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
                    return None, None

                param_change = np.abs(np.array(keypt_s) - np.array(keypt_u))
                # print("param change mean, std", np.mean(param_change), np.std(param_change), "\n")

                init_guess = new_thread.c.flatten()
                t_step.stop("[optim] reparam")

            # print(f"solver iteration: {i+1} of {iter}")
            t_step.start()
            qp_out, qp_status = QP_step(init_guess, knots, keypt_s)
            # OSQP returns an all-NaN x when the QP is infeasible/failed; a NaN
            # thread would crash every downstream consumer (quality check, mask
            # reprojection, next frame's warm start).  Skip the frame instead.
            if qp_out is None or not np.all(np.isfinite(qp_out)):
                print("\033[38;2;255;165;0moptim: QP solver returned "
                      "non-finite solution (infeasible?), skipping frame\033[0m")
                self._dump_qp_failure(mask1_t, keypoints, order, keypt_s,
                                      knots, i, tag="optim")
                if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
                return None, None
            new_ctrl = qp_out.reshape(num_ctrl, 3)
            new_thread = interp.BSpline(knots, new_ctrl, k)

            # old_samples = spline(np.linspace(0, keypt_u[-1], 150))
            new_samples = new_thread(np.linspace(0, knots[-1], 150))

            # num_eval_pts = 200
            # left, left_max = reprojection_error(new_thread, mask1, P1, num_eval_pts)
            # right, right_max = reprojection_error(new_thread, mask2, P2, num_eval_pts)
            # print("Reprojection error: mean left %f, max left %f, mean right %f, max right %f" \
            #     % (left, left_max, right, right_max))

            # plt.imshow(img1, cmap="gray")
            if not speedy:
                ax = plt.subplot(projection="3d")
                # ax.plot(old_samples[:, 0], old_samples[:, 1], old_samples[:, 2])
                ax.plot(new_samples[:, 0], new_samples[:, 1], new_samples[:, 2])
                ax.scatter(new_ctrl[..., 0], new_ctrl[..., 1], new_ctrl[..., 2])
                ax.plot(new_thread(keypt_s)[:, 0], new_thread(keypt_s)[:, 1], constr_lower_d, c="turquoise")
                ax.plot(new_thread(keypt_s)[:, 0], new_thread(keypt_s)[:, 1], constr_upper_d, c="turquoise")

                set_axes_equal(ax)
                plt.show() # comment if running profiling
            t_step.stop("[optim] qp step")
            thread, keypt_u = new_thread, keypt_s

            # geometric convergence check on the sampled curve
            if prev_samples is not None and CONV_TOL > 0:
                delta = float(np.max(np.linalg.norm(new_samples - prev_samples, axis=1)))
                if delta < CONV_TOL:
                    # print(f"[optim] converged at iter {i+1}/{iter} (max curve delta={delta:.3f})")
                    break
            prev_samples = new_samples
        # The loop exited on `thread`, which came from the LAST solve above —
        # so that is the one whose convergence decides whether this frame may
        # be published (see OSQP_OK_STATUS).
        if qp_status not in OSQP_OK_STATUS:
            print(f"\033[38;2;255;165;0moptim: final QP did not converge "
                  f"('{qp_status}'), skipping frame — the returned iterate has "
                  f"no accuracy guarantee\033[0m")
            self._dump_qp_failure(mask1_t, keypoints, order, keypt_s,
                                  knots, i, tag="optim-nonconverged")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None
        t.stop("[optim] iterations")
        t.start()
        # Assign reliability values w/ a gaussian
        bounds = constr_upper_d-constr_lower_d
        reliability_bounds = self._reliability_values(
            bounds, keypt_conf=keypt_conf, order=order, speedy=speedy)
        keypt_s[0], keypt_s[-1] = 0.0, 1.0
        # print(bounds)
        # print(reliability_bounds)
        reliability = interp.interp1d(keypt_s, reliability_bounds)

        # trim thread
        num_new_points = 200
        # if not speedy:
        #     # print("reprojection_trim")
        #     new_thread, [start, end] = reproject_trim(thread, mask1_t, P1)
        # else:
        new_thread = thread
        [start, end] = [0, 1]
        trim_thread = new_thread(np.linspace(0, 1, 200)) # sample 200 points from new thread

        # trim reliability, upper and lower constraints as well
        new_idx = np.linspace(start, end, num_new_points)

        constr_lower_d_f = interp.interp1d(keypt_s, constr_lower_d)
        constr_upper_d_f = interp.interp1d(keypt_s, constr_upper_d)
        constr_lower_d_trim = constr_lower_d_f(new_idx)
        constr_upper_d_trim = constr_upper_d_f(new_idx)
        reliability_trim = self._degrade_gap_reliability(
            reliability(new_idx), new_idx, keypt_s, speedy=speedy)

        if not speedy:
            # check trimmed thread
            ax = plt.subplot(projection="3d")
            ax.plot(trim_thread[:, 0], trim_thread[:, 1], trim_thread[:, 2])    
            ax.plot(thread(new_idx)[:, 0], thread(new_idx)[:, 1], constr_lower_d_trim, c="turquoise")
            ax.plot(thread(new_idx)[:, 0], thread(new_idx)[:, 1], constr_upper_d_trim, c="turquoise")
            ax.scatter(new_ctrl[..., 0], new_ctrl[..., 1], new_ctrl[..., 2])
            set_axes_equal(ax)
            plt.show() # comment for profiling

            plt.close('all')


        thread_specs = {"reliability": reliability_trim,
                        "lower_constr": np.transpose(np.array((trim_thread[:, 0], trim_thread[:, 1], constr_lower_d_trim))),
                        "upper_constr": np.transpose(np.array((trim_thread[:, 0], trim_thread[:, 1], constr_upper_d_trim))),
                        "keypt_s": keypt_s
                        }
        thread = {'thread': new_thread}
        t.stop("[optim] clean up")
        return thread, thread_specs

    def optim_warm_start(self, mask1_t, keypoints, order, cam2img, P1, warm_thread=None, warm_keypts=None, speedy=False, ros_enable=False, needle_pos_file=None,
                         lambda_temporal=TEMPORAL_LAMBDA, trans=None, tool_pos_3d=None, deform_radius=TEMPORAL_DEFORM_RADIUS,
                         keypt_conf=None):
        """
        keypt_conf: (len(keypoints),) per-keypoint stereo confidence, aligned
        with `keypoints` and indexed by `order` — see optim() and
        RELIABILITY_MODE.

        Temporal regularization (same as Optim.optim): a Tikhonov prior pulls the
        control points toward warm_thread — the previous frame's spline, which is
        already motion-compensated to this frame by refresh_warm_start (same warp
        as SplineEKF.predict()).  So warm_thread doubles as x_prior and no further
        compensation is needed (leave trans=None).

        lambda_temporal: dimensionless prior weight (see TEMPORAL_LAMBDA); 0 = off.
        trans/tool_pos_3d/deform_radius: only for a RAW (un-warped) prior — see
                         Optim.optim.  Normally unused here since warm_thread is
                         already compensated.
        """
        # print("using optim warm start")
        if warm_keypts is None:
            print("No warm start thread provided for optim")
            thread, thread_specs = None, None
            return thread, thread_specs
        try:
            warm_thread, warm_keypts = refit_spline(warm_thread, warm_keypts)
        except ValueError as e:
            print(f"\033[38;2;255;165;0moptim_warm_start: {e} — skipping frame\033[0m")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None
        k = 3
        num_ctrl = warm_thread.c.shape[0]
        num_constr = len(order)*3

        t = Timer(enabled=self.timing or not speedy)
        t.start()
        init_pts, keypoint_idxs = keypoints[order], np.arange(len(order))
        knots, init_u, constr_lower_d, constr_upper_d = self.warm_optim_init(init_pts=init_pts, 
                                                                        keypoints=keypoints, 
                                                                        keypoint_idxs=keypoint_idxs, 
                                                                        order=order,cam2img=cam2img, 
                                                                        warm_keypts=warm_keypts, 
                                                                        warm_thread=warm_thread
                                                                        )

        if knots is None or len(knots) == 0:
            print(f"\033[38;2;255;165;0mknots is None, returning None thread\033[0m")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None

        if needle_pos_file is not None:
            point = self.get_needle_point(needle_pos_file)
            aug_pts = np.append(point, 1)
            proj_pts = (P1 @ aug_pts.T).T
            proj_pts /= proj_pts[2].copy() + 1e-7
            proj_pts = [proj_pts[1], proj_pts[0], point[2]]

            keypoints[order[-1]] = proj_pts
            constr_lower_d[order[-1]] = proj_pts[2] - 1e-1
            constr_upper_d[order[-1]] = proj_pts[2] + 1e-1


        keypt_u = init_u[keypoint_idxs]
        thread = None

        constr_centers = keypoints[order]
        constr_lower_px = constr_centers[:, 1] - CONSTR_WIDTH_2D
        constr_upper_px = constr_centers[:, 1] + CONSTR_WIDTH_2D
        constr_lower_py = constr_centers[:, 0] - CONSTR_WIDTH_2D
        constr_upper_py = constr_centers[:, 0] + CONSTR_WIDTH_2D

        # Put bounds in a good shape
        # x constr
        constr_lower_px_rshp = np.repeat(
            constr_lower_px, num_ctrl*3
        ).reshape(num_constr//3, num_ctrl*3)
        constr_upper_px_rshp = np.repeat(
            constr_upper_px, num_ctrl*3
        ).reshape(num_constr//3, num_ctrl*3)

        # y constr
        constr_lower_py_rshp = np.repeat(
            constr_lower_py, num_ctrl*3
        ).reshape(num_constr//3, num_ctrl*3)
        constr_upper_py_rshp = np.repeat(
            constr_upper_py, num_ctrl*3
        ).reshape(num_constr//3, num_ctrl*3)
        # knots, num_ctrl, k = thread.t, len(thread.c), thread.k
        
        def QP_step(init_guess, knots, keypt_s):
            # Set up optimization...
            solver = osqp.OSQP()
            k = 3
            if init_guess is not None:
                num_ctrl = int(len(init_guess)/3)
            else:
                num_ctrl = len(knots)-1-k
            # Create objective function
            deriv_coeff = (
                self.get_deriv_matrix(knots[2:-2], num_ctrl-2, k-2) @
                self.get_deriv_matrix(knots[1:-1], num_ctrl-1, k-1) @
                self.get_deriv_matrix(knots, num_ctrl, k)
            )
            weight_coeff = np.diag(
                np.repeat(knots[4:-3] - knots[3:-4], 3)
            )
            loss_coeff = (
                deriv_coeff.T @
                weight_coeff @
                deriv_coeff
            )

            # --- Temporal (shape) prior ---------------------------------------
            # Pull the control-polygon EDGE VECTORS toward warm_thread's
            # (previous frame, already motion-compensated).  Translation-
            # invariant — see _temporal_prior_terms; reg is scaled by the
            # bending energy so lambda_temporal is dimensionless.
            q_lin = np.zeros(num_ctrl * 3)
            if lambda_temporal > 0 and warm_thread is not None:
                x_prior = self._temporal_prior_ctrl(
                    warm_thread, knots, num_ctrl, k,
                    trans=trans, tool_pos_3d=tool_pos_3d,
                    deform_radius=deform_radius)
                if x_prior is not None:
                    p_scale = np.trace(loss_coeff) / (num_ctrl * 3)
                    reg = lambda_temporal * max(p_scale, 1e-9)
                    P_add, q_lin = self._temporal_prior_terms(
                        x_prior, num_ctrl, reg)
                    loss_coeff = loss_coeff + P_add

            # Create constraints
            spl_bases = np.zeros((num_constr, num_ctrl*3))
            I = np.eye(num_ctrl)
            for i in range(num_ctrl):
                basis = interp.BSpline(knots, I[i], k)
                basis_eval = basis(keypt_s)
                spl_bases[::3, 3*i] = basis_eval
                spl_bases[1::3, 3*i+1] = basis_eval
                spl_bases[2::3, 3*i+2] = basis_eval
                
            cam2img_rep = np.zeros((num_constr, num_constr))
            for i in range(0, num_constr, 3):
                cam2img_rep[i:i+3, i:i+3] = cam2img
            spl_eval_matrix = cam2img_rep @ spl_bases
            
            I_constr = np.eye(num_constr)
            eval_select_x = I_constr[::3] @ spl_eval_matrix
            eval_select_y = I_constr[1::3] @ spl_eval_matrix
            eval_select_z = I_constr[2::3] @ spl_eval_matrix
            
            x_lower = constr_lower_px_rshp * eval_select_z - eval_select_x
            x_upper = eval_select_x - constr_upper_px_rshp * eval_select_z
            y_lower = constr_lower_py_rshp * eval_select_z - eval_select_y
            y_upper = eval_select_y - constr_upper_py_rshp * eval_select_z
            z_lower = eval_select_z
            z_upper = eval_select_z

            constr_A = np.concatenate((x_lower, x_upper, y_lower, y_upper, z_lower), axis=0)
            constr_l = np.ones(num_constr//3*5) * (-np.inf)
            constr_l[-num_constr//3:] = constr_lower_d
            constr_u = np.zeros_like(constr_l)
            constr_u[-num_constr//3:] = constr_upper_d

            constr_l_wide = np.ones(num_constr//3*5) * (-np.inf)
            constr_u_wide = np.zeros_like(constr_l)
            # constr_u_wide = np.ones(num_constr//3*5) * (np.inf)

            constr_l_wide[-num_constr//3:] = constr_lower_d # - 5e0
            constr_u_wide[-num_constr//3:] = constr_upper_d # + 5e0

            if speedy:
                verbose = False
            else:
                verbose = True
            solver.setup(P=csc_matrix(loss_coeff), q=q_lin, A=csc_matrix(constr_A), l=constr_l_wide, u=constr_u_wide, verbose=verbose,
                         eps_abs=OSQP_EPS, eps_rel=OSQP_EPS, max_iter=OSQP_MAX_ITER)
            if init_guess is not None:
                solver.warm_start(x=init_guess)
            result = solver.solve()
            if result.info.status != 'solved':
                print(f"OSQP status: '{result.info.status}' "
                      f"(iter={result.info.iter})")
            # print("warm start:", init_guess is not None)
            # OSQP can return None or a None-filled object array on a
            # primal/dual-infeasible exit; coerce to a float array (None → NaN)
            # or None so the caller's np.isfinite() check is always valid and
            # never raises "ufunc 'isfinite' not supported" on an object array.
            x = result.x
            if x is None:
                if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
                return None, result.info.status
            try:
                return np.asarray(x, dtype=float), result.info.status
            except (TypeError, ValueError):
                if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
                return None, result.info.status

        t.stop("[optim_warm] init + setup")
        t.start()
        iter = OPTIM_MAX_ITER
        prev_samples = None
        # Safe default: if the loop below never runs, `None` is not in
        # OSQP_OK_STATUS, so the frame is skipped rather than published.
        qp_status    = None
        for i in range(iter):
            if i == 0:
                knots, keypt_s = knots, keypt_u

                # Reparameterise warm thread onto [0,1] to match the new knot span,
                # then use its control points as the initial guess.
                try:
                    new_thread, _, _ = reparam(warm_thread, warm_keypts)
                except ValueError as e:
                    print(f"\033[38;2;255;165;0moptim_warm_start: {e} — skipping frame\033[0m")
                    if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
                    return None, None
                init_guess = new_thread.c.flatten()
                # keypt_s comes from warm_optim_init (arc-length u on current keypoints)
            else:
                try:
                    new_thread, knots, keypt_s = reparam(thread, keypt_u)
                except ValueError as e:
                    print(f"\033[38;2;255;165;0moptim_warm_start: {e} — skipping frame\033[0m")
                    if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
                    return None, None
                
                param_change = np.abs(np.array(keypt_s) - np.array(keypt_u))
                # print("param change mean, std", np.mean(param_change), np.std(param_change))

                init_guess = new_thread.c.flatten()
                # init_guess_y = warm_start_y

            # print(f"solver iteration: {i+1} of {iter}")
            qp_out, qp_status = QP_step(init_guess, knots, keypt_s)
            # See optim(): infeasible OSQP → all-NaN x → skip frame.
            if qp_out is None or not np.all(np.isfinite(qp_out)):
                print("\033[38;2;255;165;0moptim_warm_start: QP solver returned "
                      "non-finite solution (infeasible?), skipping frame\033[0m")
                self._dump_qp_failure(mask1_t, keypoints, order, keypt_s,
                                      knots, i, tag="optim_warm_start")
                if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
                return None, None
            new_ctrl = qp_out.reshape(num_ctrl, 3)
            new_thread = interp.BSpline(knots, new_ctrl, k)

            # old_samples = thread(np.linspace(0, keypt_u[-1], 150))
            new_samples = new_thread(np.linspace(0, knots[-1], 150))

            num_eval_pts = 200
            if speedy or ros_enable:
                pass
            else:
                # left, left_max = reprojection_error(new_thread, mask1, P1, num_eval_pts,)
                # right, right_max = reprojection_error(new_thread, mask2, P2, num_eval_pts)
                    
                ax = plt.subplot(projection="3d")
                # ax.plot(old_samples[:, 0], old_samples[:, 1], old_samples[:, 2])
                ax.plot(new_samples[:, 0], new_samples[:, 1], new_samples[:, 2], c="blue")
                ax.scatter(new_ctrl[..., 0], new_ctrl[..., 1], new_ctrl[..., 2], c="blue")
                ax.plot(new_thread(keypt_s)[:, 0], new_thread(keypt_s)[:, 1], constr_lower_d, c="turquoise")
                ax.plot(new_thread(keypt_s)[:, 0], new_thread(keypt_s)[:, 1], constr_upper_d, c="turquoise")

                # plot the warm start thread
                warm_sample = warm_thread(np.linspace(0, knots[-1], 150))
                warm_ctrl = warm_thread.c
                ax.plot(warm_sample[:, 0], warm_sample[:, 1], warm_sample[:, 2], c='red')
                ax.scatter(warm_ctrl[..., 0], warm_ctrl[..., 1], warm_ctrl[..., 2], c='red')

                # plot first thread without warm start (compare with first thread with warm start)
                # ax.plot(first_samples[:, 0], first_samples[:, 1], first_samples[:, 2], c='green')
                # ax.scatter(first_ctrl[..., 0], first_ctrl[..., 1], first_ctrl[..., 2], c='green')

                set_axes_equal(ax)
                plt.show() # comment for profiling
                # plt.show(block=False) # comment for profiling
                # plt.pause(0.3)
                # plt.close(1)
            thread, keypt_u = new_thread, keypt_s

            # geometric convergence check on the sampled curve (mirrors optim())
            if prev_samples is not None and OPTIM_CONV_TOL > 0:
                delta = float(np.max(np.linalg.norm(new_samples - prev_samples, axis=1)))
                if delta < OPTIM_CONV_TOL:
                    break
            prev_samples = new_samples
        # Same rule as optim(): only the solve that produced the returned
        # thread has to have converged (see OSQP_OK_STATUS).
        if qp_status not in OSQP_OK_STATUS:
            print(f"\033[38;2;255;165;0moptim_warm_start: final QP did not "
                  f"converge ('{qp_status}'), skipping frame — the returned "
                  f"iterate has no accuracy guarantee\033[0m")
            self._dump_qp_failure(mask1_t, keypoints, order, keypt_s,
                                  knots, i, tag="optim_warm_start-nonconverged")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None
        t.stop("[optim_warm] iterations")
        t.start()

        # Assign reliability values w/ a gaussian
        bounds = constr_upper_d-constr_lower_d
        reliability_bounds = self._reliability_values(
            bounds, keypt_conf=keypt_conf, order=order, speedy=speedy)
        keypt_s[0], keypt_s[-1] = 0.0, 1.0
        # print(bounds)
        # print(reliability_bounds)
        reliability = interp.interp1d(keypt_s, reliability_bounds)

        # trim thread
        num_new_points = 200
        # if speedy or ros_enable:
        new_thread = thread # sample 200 points from new thread
        trim_thread = new_thread(np.linspace(0, 1, 200)) # sample 200 points from new thread for plot
        start = keypt_s[0]
        end = keypt_s[-1]
        # else:
            # print("reprojection_trim")
            # new_thread, [start, end] = reprojection_trim(thread, mask1_t, P1, num_eval_pts=num_new_points)
            # new_thread, [start, end] = reproject_trim(thread, mask1_t, P1, num_show_pts=num_new_points)
            # trim_thread = new_thread(np.linspace(0, 1, 200)) # sample 200 points from new thread for plot


        # trim reliability, upper and lower constraints as well
        new_idx = np.linspace(start, end, num_new_points)

        constr_lower_d_f = interp.interp1d(keypt_s, constr_lower_d)
        constr_upper_d_f = interp.interp1d(keypt_s, constr_upper_d)
        constr_lower_d_trim = constr_lower_d_f(new_idx)
        constr_upper_d_trim = constr_upper_d_f(new_idx)
        reliability_trim = self._degrade_gap_reliability(
            reliability(new_idx), new_idx, keypt_s, speedy=speedy)

        # check trimmed thread
        if speedy or ros_enable:
            pass
        else:
            ax = plt.subplot(projection="3d")
            ax.plot(trim_thread[:, 0], trim_thread[:, 1], trim_thread[:, 2])    
            ax.plot(thread(new_idx)[:, 0], thread(new_idx)[:, 1], constr_lower_d_trim, c="turquoise")
            ax.plot(thread(new_idx)[:, 0], thread(new_idx)[:, 1], constr_upper_d_trim, c="turquoise")
            ax.scatter(new_ctrl[..., 0], new_ctrl[..., 1], new_ctrl[..., 2])
            set_axes_equal(ax)
            plt.show() # comment for profiling

            plt.close('all')


        thread_specs = {"reliability": reliability_trim,
                        "lower_constr": np.transpose(np.array((trim_thread[:, 0], trim_thread[:, 1], constr_lower_d_trim))),
                        "upper_constr": np.transpose(np.array((trim_thread[:, 0], trim_thread[:, 1], constr_upper_d_trim))),
                        "keypt_s": keypt_s
                        }
        thread = {'thread': new_thread}
        t.stop("[optim_warm] clean up")
        return thread, thread_specs

    def optim_init(self, init_pts, keypoints, keypoint_idxs, order, cam2img):
        num_order = len(order)
        
        # --- OPTIMIZATION 1: Pre-allocate and extract data outside the loop ---
        bound_rads = np.zeros(num_order)
        fit_rad = keypoints.shape[0] // 10
        if fit_rad == 0:
            print(f"\033[38;2;255;165;only {keypoints.shape[0]} keypoints, not enough\033[0m")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None, None, None
        bound_thresh = 1e-5
        lowest_nonzero = np.inf
        
        # Extract Z columns once to avoid repeated 3D slicing
        init_z = init_pts[:, 2]
        keypoints_z = keypoints[order, 2]
        
        # Endpoint tracking (solves your TODO)
        endpts_0 = init_z[0]
        endpts_1 = init_z[-1]
        
        for key_ord in range(num_order):
            # Fit line to range of points
            start = max(key_ord - fit_rad, 0)
            end = min(key_ord + fit_rad, num_order - 1)
            
            if start == 0:
                end = min(end + fit_rad // 2, num_order - 1)
            elif end == num_order - 1:
                start = max(start - fit_rad // 2, 0)
                
            start_idx = keypoint_idxs[start]
            end_idx = keypoint_idxs[end] + 1
            x = np.arange(start_idx, end_idx)
            data = init_z[x]
            
            # --- OPTIMIZATION 2: Pure NumPy 1D Linear Regression ---
            # Replaces slow scipy.stats.linregress
            x_mean = np.mean(x)
            y_mean = np.mean(data)
            dx = x - x_mean
            dy = data - y_mean
            var_x = np.dot(dx, dx)
            
            slope = np.dot(dx, dy) / var_x if var_x != 0 else 0.0
            intercept = y_mean - slope * x_mean

            # --- OPTIMIZATION 3: Capture Endpoints Instantly ---
            # Eliminates the need for the entire second loop
            if key_ord == 0:
                endpts_0 = intercept
            elif key_ord == num_order - 1:
                endpts_1 = slope * x[-1] + intercept

            # Construct bound radius from current point
            curr_line_pt = slope * keypoint_idxs[key_ord] + intercept
            line_pts = slope * x + intercept
            
            line_std = np.sqrt(np.mean((data - line_pts)**2))
            
            bound_rad = np.abs(keypoints_z[key_ord] - curr_line_pt) * 1.5
            bound_rad = max(bound_rad, line_std)
            bound_rads[key_ord] = bound_rad
            
            if bound_thresh < bound_rad < lowest_nonzero:
                lowest_nonzero = bound_rad

        # --- OPTIMIZATION 4: Vectorized Bounds Assignment ---
        bound_rads[bound_rads <= bound_thresh] = lowest_nonzero
        
        # Generate only the Z-bounds, as that's all the function returns
        lower_z = keypoints_z - bound_rads
        upper_z = keypoints_z + bound_rads

        # Apply endpoint trends
        init_pts[0, 2] = endpts_0
        init_pts[-1, 2] = endpts_1

        # Bring points into camera coords
        init_pts = change_coords(init_pts, cam2img)

        k = 3
        # Control-point count for the fitted spline.  Was one per ordered
        # keypoint (num_order) — an interpolating, over-parameterised spline
        # that made the QP scale with keypoint count (3·num_order variables)
        # and chased per-keypoint noise.  Cap it to ~num_order/3 so the spline
        # is a genuine least-squares fit: the QP shrinks (3-4x faster at high
        # keypoint counts) and the curve is smoother.  Floor at CTRL_MIN so
        # short threads keep enough freedom; k+1 is the B-spline minimum.
        num_ctrl = max(CTRL_MIN, num_order // CTRL_DIVISOR)

        # --- OPTIMIZATION 5: Fast C-compiled Distances ---
        diffs = init_pts[1:] - init_pts[:-1]
        dists = np.sqrt(np.einsum('ij,ij->i', diffs, diffs))
        dists /= np.sum(dists)
        
        u = np.zeros(init_pts.shape[0])
        u[1:] = np.cumsum(dists)
        u[-1] = 1

        knots = np.concatenate((
            np.zeros(k), 
            np.linspace(0, u[-1], num_ctrl),
            np.full(k, u[-1])
        ))

        return knots, u, lower_z, upper_z


    def warm_optim_init(self, init_pts, keypoints, keypoint_idxs, order, cam2img, warm_keypts=None, warm_thread=None):
        # Construct bounds
        lower = np.zeros((len(order), 3))
        upper = np.zeros((len(order), 3))
        fit_rad = keypoints.shape[0] // 10
        if fit_rad == 0:
            print(f"\033[38;2;255;165;only {keypoints.shape[0]} keypoints, not enough\033[0m")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None, None, None

        bound_rads = []
        bound_thresh = 1e-5
        lowest_nonzero = np.inf
        key_line_pts = np.zeros(len(order))
        for key_ord, key_id in enumerate(order):
            # Fit line to range of points
            start = max(key_ord-fit_rad, 0)
            end = min(key_ord+fit_rad, len(order)-1)
            if start == 0:
                end = min(end+fit_rad//2, len(order)-1)
            elif end == len(order)-1:
                start = max(start-fit_rad//2, 0)
            x = np.arange(keypoint_idxs[start], keypoint_idxs[end]+1)
            data = init_pts[x, 2]
            slope, intercept, *_ = linregress(x, data)

            # Construct bound radius from current point
            line_pts = slope * x + intercept
            curr_line_pt = slope * keypoint_idxs[key_ord] + intercept
            key_line_pts[key_ord] = curr_line_pt
            line_std = np.mean((data - line_pts)**2) ** (1/2)
            bound_rad = np.abs(keypoints[key_id, 2] - curr_line_pt) * 1.5
            bound_rad = max(bound_rad, line_std)
            bound_rads.append(bound_rad)
            if bound_rad > bound_thresh:
                lowest_nonzero = min(lowest_nonzero, bound_rad)
        
        # Set bounds, accounting for minimum radius threshold
        for key_ord, (key_id, bound_rad) in enumerate(zip(order, bound_rads)):
            if bound_rad <= bound_thresh:
                bound_rad = lowest_nonzero
            lower[key_ord] = keypoints[key_id] - np.array([[0, 0, bound_rad]])
            upper[key_ord] = keypoints[key_id] + np.array([[0, 0, bound_rad]])

        k = 3
        num_ctrl = warm_thread.c.shape[0] - 2

        # Compute u from actual ordered keypoint arc-lengths in camera coords,
        # then rescale to [0, 1]. Using warm_keypts directly is only valid when
        # len(warm_keypts) == len(order) and they're in the same order, which is
        # not guaranteed after CPD matching / unmatched keypoint extension.
        init_pts_cam = change_coords(init_pts, cam2img)
        dists = np.linalg.norm(init_pts_cam[1:] - init_pts_cam[:-1], axis=1)
        dists /= np.sum(dists) + 1e-10
        u = np.zeros(init_pts_cam.shape[0])
        u[1:] = np.cumsum(dists)
        u[-1] = 1.0

        # Knots span [0, 1] with clamped ends
        knots = np.concatenate((
            np.repeat(0.0, k),
            np.linspace(0.0, 1.0, num_ctrl),
            np.repeat(1.0, k)
        ))

        # knots is linearly spaced, u is not, it's based on keypoint positions
        return knots, u, lower[:, 2], upper[:, 2]

    def get_needle_point(self, pos_file):
        r = 8.2761
        with open(pos_file, 'rb') as f:
            data = pickle.load(f)

        needle_pos = np.array([data.get('x'), data.get('y'), data.get('z'), data.get('qw'), data.get('qx'), data.get('qy'), data.get('qz')]) * 1000
        theta = np.pi*3/2
        conn_pt = np.array([r*np.cos(theta), r*np.sin(theta), 0])
        R = R.from_quat(needle_pos[3:]).as_matrix()
        conn_pt = np.matmul(R, conn_pt)
        conn_pt = conn_pt + needle_pos[:3]

        return conn_pt

    def _reliability_values(self, bounds, keypt_conf=None, order=None,
                            speedy=False):
        """Per-ordered-keypoint reliability in [0, 1] — see RELIABILITY_MODE.

        bounds     : (num_order,) z-bound WIDTH per ordered keypoint
                     (constr_upper_d - constr_lower_d).
        keypt_conf : (len(keypoints),) stereo confidence aligned with the
                     KEYPOINT array, indexed here by `order` exactly as
                     keypoints[order] is.  None → 'geometry' fallback.
        """
        cutoff, sigma = 3, 8
        clipped = np.clip(bounds, a_min=cutoff, a_max=None)
        geom = (gaussian(clipped, cutoff, sigma) /
                (gaussian(cutoff, cutoff, sigma) + 1e-3))

        if RELIABILITY_MODE == 'geometry':
            return geom
        if keypt_conf is None or order is None:
            print(f"\033[38;2;255;165;0moptim: RELIABILITY_MODE="
                  f"'{RELIABILITY_MODE}' but no stereo confidence was supplied "
                  f"— falling back to 'geometry'\033[0m")
            return geom

        conf = np.asarray(keypt_conf, dtype=float)
        ordr = np.asarray(order)
        if ordr.size and (ordr.max() >= conf.size or ordr.min() < 0):
            print(f"\033[38;2;255;165;0moptim: keypt_conf has {conf.size} "
                  f"entries but order indexes up to {int(ordr.max())} — "
                  f"falling back to 'geometry'\033[0m")
            return geom
        conf = np.clip(conf[ordr], 0.0, 1.0)
        if conf.shape != np.shape(geom):
            print(f"\033[38;2;255;165;0moptim: reliability length mismatch "
                  f"(conf {conf.shape} vs bounds {np.shape(geom)}) — falling "
                  f"back to 'geometry'\033[0m")
            return geom

        if RELIABILITY_MODE == 'measurement':
            out = conf
        elif RELIABILITY_MODE == 'combined':
            out = geom * conf
        else:
            print(f"\033[38;2;255;165;0moptim: unknown RELIABILITY_MODE "
                  f"'{RELIABILITY_MODE}' — falling back to 'geometry'\033[0m")
            return geom

        if not speedy:
            print(f"optim reliability[{RELIABILITY_MODE}]: "
                  f"min={out.min():.2f} median={np.median(out):.2f} "
                  f"max={out.max():.2f}  "
                  f"(geometry-only median would be {np.median(geom):.2f}, "
                  f"stereo-conf median {np.median(conf):.2f})")
        return out

    def _degrade_gap_reliability(self, rel, sample_s, keypt_s, speedy=False):
        """Scale per-sample reliability down in DATA GAPS (GAP_RELIAB_DECAY_S).

        rel/sample_s : per-published-sample reliability and its s positions.
        keypt_s      : s of the ordered keypoints that actually supported the
                       fit.  Samples far from every keypoint carry
                       interpolated, not measured, confidence — degrade them
                       by exp(-max(0, d − deadband)/decay), deadband = median
                       keypoint spacing so normally-supported stretches keep
                       their value exactly."""
        if not GAP_RELIAB_DECAY_S or GAP_RELIAB_DECAY_S <= 0:
            return rel
        ks = np.sort(np.asarray(keypt_s, dtype=float).ravel())
        if ks.size < 2:
            return rel
        d = np.min(np.abs(np.asarray(sample_s, dtype=float)[:, None]
                          - ks[None, :]), axis=1)
        deadband = float(np.median(np.diff(ks)))
        factor = np.exp(-np.maximum(0.0, d - deadband) / GAP_RELIAB_DECAY_S)
        n_deg = int((factor < 0.9).sum())
        if n_deg and not speedy:
            print(f"optim reliability: gap-degraded {n_deg}/{len(factor)} "
                  f"published samples (worst ×{factor.min():.2f}, largest "
                  f"gap-to-keypoint {d.max():.3f}s, deadband {deadband:.3f}s)")
        return rel * factor

    def _motion_compensate_points(self, pts, trans, tool_pos_3d, deform_radius):
        """Warp 3-D points by rigid tool motion, identically to
        SplineEKF.predict() / warm_start: BOTH translation and rotation are
        distance-weighted (w = exp(-‖p-pivot‖/R), pivot = prev-frame grasp), so
        only the near-tool portion follows the tool and farther parts lag.

        pts : (N, 3).  Returns (N, 3).  If trans is None, pts is returned as-is.
        """
        if trans is None:
            return pts
        trans = np.asarray(trans, dtype=float)
        R = trans[:3, :3]
        tvec = trans[:3, 3]
        if tool_pos_3d is None:
            # No pivot for the distance weighting → apply translation only.
            return pts + tvec
        tool = np.asarray(tool_pos_3d, dtype=float).reshape(3)
        delta = pts - tool                       # (N, 3), about the grasp
        rot_dev = (R @ delta.T).T - delta        # (R - I)(p - pivot)
        dist = np.linalg.norm(delta, axis=1)
        w = np.exp(-dist / max(deform_radius, 1e-6))
        return pts + w[:, None] * (tvec + rot_dev)

    @staticmethod
    def _dump_qp_failure(mask, keypoints, order, keypt_s, knots, itr, tag="optim"):
        """Diagnose an infeasible/failed QP: characterise the ordered keypoints
        that produced it and save them for offline inspection.

        Prints step statistics (2-D pixel spacing, z jumps, direction
        reversals, duplicate parameter values) and writes debug_qp_fail.png
        (order polyline over the mask, viridis = position in order) and
        debug_qp_fail.npz (raw arrays)."""
        try:
            pts = np.asarray(keypoints)[np.asarray(order)]
            d2  = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
            dz  = np.abs(np.diff(pts[:, 2]))
            v   = np.diff(pts[:, :2], axis=0)
            dots = (v[:-1] * v[1:]).sum(axis=1)
            revs = int((dots < 0).sum())               # >90° turns = zigzag
            ds  = np.diff(np.asarray(keypt_s, dtype=float))
            print(f"[{tag} QP-fail diag] N={len(pts)} "
                  f"ctrl={len(knots) - 4} iter={itr}\n"
                  f"  2D step px : median={np.median(d2):.1f}  max={d2.max():.1f}\n"
                  f"  z step     : median={np.median(dz):.2f}  max={dz.max():.2f}\n"
                  f"  direction reversals (>90 deg): {revs}/{len(dots)}\n"
                  f"  keypt_s steps: min={ds.min():.4g} "
                  f"(dupes={(np.abs(ds) <= 1e-9).sum()})  max={ds.max():.4g}")
            np.savez("debug_qp_fail.npz",
                     keypoints=np.asarray(keypoints), order=np.asarray(order),
                     keypt_s=np.asarray(keypt_s), knots=np.asarray(knots))
            fig, ax = plt.subplots(figsize=(10, 8))
            if mask is not None:
                ax.imshow(np.asarray(mask), cmap='gray')
            ax.plot(pts[:, 1], pts[:, 0], c='orange', lw=1, alpha=0.7)
            ax.scatter(pts[:, 1], pts[:, 0],
                       c=plt.cm.viridis(np.linspace(0, 1, len(pts))),
                       s=14, zorder=3)
            ax.scatter(pts[0, 1], pts[0, 0], c='lime', s=70, marker='>',
                       zorder=5, label='order start')
            ax.scatter(pts[-1, 1], pts[-1, 0], c='red', s=70, marker='s',
                       zorder=5, label='order end')
            ax.legend(fontsize=8)
            ax.set_title(f"QP failed — ordered keypoints (viridis = order), "
                         f"{revs} reversals, N={len(pts)}")
            plt.savefig("debug_qp_fail.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("  saved debug_qp_fail.png / debug_qp_fail.npz")
        except Exception as e:
            print(f"  (QP-fail diagnostics failed: {e})")

    def _temporal_prior_terms(self, x_prior, num_ctrl, reg):
        """Translation-invariant temporal-prior quadratic terms.

        Penalizes  reg/2 · ‖D·x − D·x_prior‖²  where D takes first differences
        of consecutive control points (per coordinate): the prior acts on the
        control-polygon EDGE VECTORS — the local shape — not on absolute
        positions.  An absolute-position prior fully penalizes a rigid
        translation of the whole thread (which costs zero bending energy), so
        nothing in the objective follows real motion and the thread sticks to
        its previous position, pinned against the constraint boxes.  The shape
        prior leaves translation completely free while still damping
        wobble/waviness (a pure shape artifact).

        Returns (P_add, q_add) for OSQP's 0.5·xᵀPx + qᵀx objective.
        """
        nD = num_ctrl - 1
        D  = np.zeros((nD * 3, num_ctrl * 3))
        i  = np.arange(nD * 3)
        D[i, i]     = -1.0
        D[i, i + 3] =  1.0
        DtD = D.T @ D
        return reg * DtD, -reg * (DtD @ x_prior)

    def _temporal_prior_ctrl(self, prior_thread, knots, num_ctrl, k,
                             trans=None, tool_pos_3d=None,
                             deform_radius=TEMPORAL_DEFORM_RADIUS):
        """Build a per-control-point temporal prior x_prior for the CURRENT knot
        layout from a previous frame's spline.

        The knot count / spacing changes every frame (reparam), so the previous
        control points can't be used directly.  Instead sample the previous
        spline at the Greville abscissae of the current knots — the natural
        parameter associated with each control point — then motion-compensate the
        sampled positions the same way SplineEKF.predict() does.

        Returns a flat (num_ctrl*3,) array, or None if a prior can't be built.
        """
        if prior_thread is None:
            return None
        knots = np.asarray(knots, dtype=float)

        lo, hi = knots[k], knots[-k - 1]         # valid parameter span
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None

        # Map the normalized fraction onto the prior spline's own domain.  Both
        # are arc-length-ish parameterizations of the same physical thread, so a
        # normalized-fraction correspondence aligns them.
        try:
            p_lo, p_hi = float(prior_thread.t[0]), float(prior_thread.t[-1])
        except AttributeError:
            p_lo, p_hi = 0.0, 1.0                # generic callable on [0, 1]

        # Least-squares refit of the prior CURVE onto the current knot layout.
        # Do NOT use on-curve samples at the Greville abscissae directly as the
        # control-point targets: control points of a cubic B-spline lie OUTSIDE
        # the curve wherever it bends, so pulling them onto on-curve positions
        # systematically flattens bends and shrinks loops every frame, even for
        # a static thread.  The lstsq refit returns the control points whose
        # spline reproduces the prior curve on these knots — an unbiased prior.
        n_samp = max(4 * num_ctrl, 64)
        us     = np.linspace(lo, hi, n_samp)
        frac   = (us - lo) / (hi - lo)
        prior_pts = np.asarray(prior_thread(p_lo + frac * (p_hi - p_lo)))
        if prior_pts.shape != (n_samp, 3) or not np.all(np.isfinite(prior_pts)):
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None
        prior_pts = self._motion_compensate_points(
            prior_pts, trans, tool_pos_3d, deform_radius)

        B = interp.BSpline.design_matrix(
            np.clip(us, lo, hi - 1e-10), knots, k).toarray()
        ctrl, _, rank, _ = np.linalg.lstsq(B, prior_pts, rcond=None)
        if rank < num_ctrl or not np.all(np.isfinite(ctrl)):
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None
        return ctrl.flatten()

    def get_deriv_matrix(self, knots, num_ctrl, k):
        mat = np.zeros((3*(num_ctrl-1), 3*num_ctrl))
        
        # Vectorized coefficient calculation
        diffs = knots[k+1 : num_ctrl+k] - knots[1 : num_ctrl]
        # Avoid division by zero if any knots overlap identically 
        coeffs = np.divide(k, diffs, out=np.zeros_like(diffs), where=diffs!=0)
        
        idx = np.arange(num_ctrl-1)
        
        # Populate Diagonal elements instantly
        mat[3*idx, 3*idx] = -coeffs
        mat[3*idx+1, 3*idx+1] = -coeffs
        mat[3*idx+2, 3*idx+2] = -coeffs
        
        # Populate Off-diagonal elements instantly
        mat[3*idx, 3*(idx+1)] = coeffs
        mat[3*idx+1, 3*(idx+1)+1] = coeffs
        mat[3*idx+2, 3*(idx+1)+2] = coeffs
        
        return mat
    def reverse_bspline(self, s):
        """Return a B-spline whose parameterization runs in the opposite direction
        over the same physical curve. new_S(u) == old_S(a + b - u)."""
        a, b = s.t[0], s.t[-1]
        new_knots = a + b - s.t[::-1]
        new_ctrl = s.c[::-1]
        return interp.BSpline(new_knots, new_ctrl, s.k)

    def flip_spline(self, thread_dict, thread_specs_dict):
        new_thread = thread_dict['thread']
        a, b = new_thread.t[0], new_thread.t[-1]
        new_thread = self.reverse_bspline(new_thread)
        thread_dict = {'thread': new_thread}
        if 'keypt_s' in thread_specs_dict:
            keypt_s = np.asarray(thread_specs_dict['keypt_s'])
            thread_specs_dict = dict(thread_specs_dict)
            thread_specs_dict['keypt_s'] = list((a + b - keypt_s)[::-1])
            thread_specs_dict['reliability'] = thread_specs_dict['reliability'][::-1]
            thread_specs_dict['lower_constr'] = thread_specs_dict['lower_constr'][::-1]
            thread_specs_dict['upper_constr'] = thread_specs_dict['upper_constr'][::-1]
        print("thread flipped")
        return thread_dict, thread_specs_dict
    
    _FLIP_PROMPT_PNG = "flip_thread_prompt.png"

    def _ask_flip_interactive(self, img1, proj_pts, title="flip thread? y"):
        """Save the projected thread to a PNG and block for a y/n answer.

        This is the human-in-the-loop initial-direction step.  It used to open a
        TkAgg window, but the ROS node now spins a MultiThreadedExecutor, so this
        runs on a WORKER thread — a Tk GUI off the main thread raises
        "main thread is not in main loop" and fails.  stdin, however, is not
        thread-restricted, so we render the figure with the thread-safe Agg
        backend to a file the user opens, and read the y/n from the terminal.
        The colour along the thread (hot colormap, dark->bright) shows the t=0->
        t=1 direction so the user can decide whether to flip.
        """
        try:
            fig = plt.figure()          # Agg (module-level backend) — thread-safe
            plt.imshow(img1)
            sc = plt.scatter(proj_pts[:, 0], proj_pts[:, 1],
                             c=np.arange(len(proj_pts)), cmap="hot")
            plt.colorbar(sc, label="thread parameter t (dark=t0  bright=t1)")
            plt.title(title)
            fig.savefig(self._FLIP_PROMPT_PNG, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[flip prompt] saved projected thread to "
                  f"{self._FLIP_PROMPT_PNG} — open it, then answer below.")
        except Exception as e:
            # Never let a plotting failure swallow the direction decision.
            print(f"[flip prompt] could not render preview ({e}); "
                  "answer from the reconstruction if you can.")
        try:
            user = input(title + " [y/N]: ")
        except EOFError:
            # No interactive stdin (e.g. launched detached) → don't flip.
            print("[flip prompt] no interactive stdin; keeping current direction.")
            return False
        return user.strip().lower() == 'y'

    # Below this |Kendall-tau| the warm/new correspondences don't agree
    # strongly enough on a direction; keep the current one rather than risk a
    # wrong flip.
    _DIR_TAU_MARGIN = 0.2
    # ── match_warm_order robustness (occlusion-proof direction) ───────────────
    # Per-correspondence STABILITY weight = exp(-residual / _DIR_STAB_SCALE),
    # where residual is the new-vs-(motion-compensated)-warm 3-D distance at the
    # match.  A correspondence on a segment that did NOT move between frames has
    # a small residual → weight ≈ 1 and dominates the direction vote; a segment
    # that shifted (tool manipulation / de-occlusion) has a large residual →
    # near-zero weight.  So the flip is decided by the stationary thread, and an
    # occluded/moving span can no longer flip it.  Units: same as keypoints.
    _DIR_STAB_SCALE = 5.0
    # Minimum TOTAL stability weight required to trust the vote at all.  Below
    # this the stationary overlap with the warm thread is too small — keep the
    # current direction rather than gamble on a flip.
    _MIN_STABLE_SUPPORT = 3.0
    # NOTE: optim no longer flips the fit on direction evidence at all — the
    # tau<0 branch logs and does nothing.  SplineEKF.update_from_thread's
    # direction guard is the single owner of direction.  The old hysteresis
    # (_FLIP_CONFIRM_FRAMES) and hard lock (LOCK_ORIENTATION_AFTER_INIT) both
    # acted here and both fought that guard frame-by-frame; see the tau<0
    # branch below.  Direction is still SET once, by the user, on the first
    # warm-less frame.

    def match_warm_order(self, img1, thread_dict, thread_specs_dict, warm_thread,
                         P, dist_thresh=10.0, n_samples=50, interactive=True):
        """If part of the previous trial's warm_start_spline lies close (in 3D)
        to the new spline, ensure the new spline's parameter direction matches
        the warm spline's. Reverses spline (and its keypt_s in spline_specs)
        when their orders disagree."""
        new_thread = thread_dict['thread']

        if warm_thread is not None:
            u_new = np.linspace(new_thread.t[0], new_thread.t[-1], n_samples)
            u_warm = np.linspace(warm_thread.t[0], warm_thread.t[-1], n_samples)
            pts_new = new_thread(u_new)
            pts_warm = warm_thread(u_warm)

            # ── correspondences via MUTUAL nearest neighbour ──────────────────
            # A plain warm→new nearest match latches onto whichever strand is
            # closest, so near a self-crossing it pairs a warm sample with the
            # WRONG branch and injects an outlier that can flip the direction
            # vote.  Requiring the match to be reciprocal (new's nearest warm is
            # also i) rejects those crossing ambiguities.
            D      = np.linalg.norm(pts_warm[:, None, :] - pts_new[None, :, :], axis=2)
            j_of_i = D.argmin(axis=1)   # nearest new sample for each warm sample
            i_of_j = D.argmin(axis=0)   # nearest warm sample for each new sample
            matched_warm_u, matched_new_u, matched_res = [], [], []
            for i, j in enumerate(j_of_i):
                if D[i, j] < dist_thresh and i_of_j[j] == i:
                    matched_warm_u.append(u_warm[i])
                    matched_new_u.append(u_new[j])
                    matched_res.append(D[i, j])   # 3-D residual → stability weight
            if len(matched_warm_u) < 3:
                print("not enough correspondences to determine direction")
                if not interactive:
                    # Headless (ROS) run: a blocking input() would hang the
                    # executor.  The thread was already assembled in warm-t
                    # order, so keep the current direction rather than prompt.
                    # This frame is too occluded to vote → clear any flip streak
                    # so stale evidence can't leak into a later flip.
                    self._flip_streak = 0
                    print("non-interactive: keeping current thread direction.")
                    return thread_dict, thread_specs_dict
                keypoints = thread_dict.get('thread')(np.linspace(0, 1, 50))
                aug_pts = np.concatenate((keypoints, np.ones((keypoints.shape[0], 1))), axis=1)
                proj_pts = (P @ aug_pts.T).T
                proj_pts /= proj_pts[:, 2:].copy() + 1e-7

                # not enough correspondences to determine direction → ask user
                if self._ask_flip_interactive(img1, proj_pts):
                    thread_dict, thread_specs_dict = self.flip_spline(thread_dict, thread_specs_dict)
                return thread_dict, thread_specs_dict

            # ── direction via STABILITY-WEIGHTED rank concordance ─────────────
            # A least-squares slope is dominated by the spread of u and by any
            # surviving outlier, so on a loop it can report the wrong sign even
            # when most correspondences agree.  Instead vote over every pair:
            # concordant (both u increase together) vs discordant.  Each pair is
            # WEIGHTED by the stability of its two correspondences (exp(-residual
            # /scale)): segments that did not move between frames dominate, so an
            # occluded/manipulated span cannot flip the thread.  tau ∈ [-1,1].
            w  = np.asarray(matched_warm_u)
            n  = np.asarray(matched_new_u)
            wt = np.exp(-np.asarray(matched_res) / max(self._DIR_STAB_SCALE, 1e-6))
            support = float(wt.sum())          # effective # of stationary matches

            agree = np.sign(w[:, None] - w[None, :]) * np.sign(n[:, None] - n[None, :])
            Wij   = wt[:, None] * wt[None, :]
            iu    = np.triu_indices(len(w), k=1)
            den   = float(Wij[iu].sum())
            tau   = float((Wij[iu] * agree[iu]).sum() / den) if den > 1e-9 else 0.0

            if support < self._MIN_STABLE_SUPPORT:
                # Too little stationary overlap to trust the vote → hold
                # direction and clear the streak (this frame is unreliable).
                print(f"direction: only {support:.1f} stable support "
                      f"(< {self._MIN_STABLE_SUPPORT}); keeping current direction.")
                self._flip_streak = 0
            elif tau < -self._DIR_TAU_MARGIN:
                # This fit came out reversed relative to the reference, so flip
                # it back — immediately, no hysteresis.  The reference is the
                # EKF spline (see the caller), i.e. the temporally-stable
                # curve keypt_ordering already t-sorted against, so it is the
                # trustworthy party and the fresh fit is the suspect one.  That
                # is exactly the occlusion case: the warm match collapses, the
                # cold ordering fallback has arbitrary direction, and without
                # this the reversed fit gets PUBLISHED.
                #
                # This previously referenced the raw warped previous thread
                # while SplineEKF.update_from_thread's guard referenced the
                # filter — two different curves, which is why the two undid
                # each other every frame (run.log: 15 flips here, 13 same-frame
                # "thread runs OPPOSITE the filter" reversals).  Sharing one
                # reference is what stops that, not removing the flip: removing
                # it left the PUBLISHED direction unprotected, since the
                # filter's guard only re-labels the measurement it fuses.
                thread_dict, thread_specs_dict = self.flip_spline(
                    thread_dict, thread_specs_dict)
                print(f"direction: fit opposed the reference (weighted "
                      f"tau={tau:.2f}, support={support:.1f}); flipped back.")
                self._flip_streak = 0
            elif tau > self._DIR_TAU_MARGIN:
                print(f"thread already matches warm start direction "
                      f"(weighted tau={tau:.2f}, support={support:.1f})")
                self._flip_streak = 0
            else:
                # too close to call → don't gamble on a flip, keep direction
                print(f"direction ambiguous (weighted tau={tau:.2f}, "
                      f"support={support:.1f}); keeping current direction.")
                self._flip_streak = 0

        else:
            # First reconstruction: there is no warm reference to infer
            # direction from, so the initial direction MUST be set by the user
            # even in an otherwise non-interactive run.  This fires only once
            # (subsequent frames have a warm_thread), so it does not stall the
            # per-frame ROS loop.
            print("no warm start thread, flip thread? y")
            keypoints = thread_dict.get('thread')(np.linspace(0, 1, 50))
            aug_pts = np.concatenate((keypoints, np.ones((keypoints.shape[0], 1))), axis=1)
            proj_pts = (P @ aug_pts.T).T
            proj_pts /= proj_pts[:, 2:].copy() + 1e-7

            if self._ask_flip_interactive(img1, proj_pts):
                thread_dict, thread_specs_dict = self.flip_spline(thread_dict, thread_specs_dict)
            return thread_dict, thread_specs_dict
        
        return thread_dict, thread_specs_dict

'''
fixing more than one needle point to be worked on
def optim(img1, mask1_t, mask2_t, mask1_n, mask2_n, img_3D, keypoints, grow_paths, order, cam2img, P1, P2, needle_pos_file=None):
    # Get necessary values
    # init_pts, keypoint_idxs = augment_keypoints(img1, segpix1, img_3D, keypoints, grow_paths, order)
    mask1 = mask1_t + mask1_n
    mask2 = mask2_t + mask2_n

    if needle_pos_file is not None:
        pts_amount = 2 # mult point or use needle_pts above

        # add two extra sudo points to fix the position and orientation of the thread connecting to needle
        # conn_pts = np.array([-r, -r/2, 0]) # single point
        needle_pts = get_needle_point(needle_pos_file)
        needle_order = (len(order) + np.array([0, 1])).tolist()
        
        order = order + needle_order # multi point
        keypoints = np.append(keypoints, needle_pts, axis=0)


    init_pts, keypoint_idxs = keypoints[order], np.arange(len(order))
    knots, init_u, constr_lower_d, constr_upper_d = optim_init(init_pts, keypoints, keypoint_idxs, order, cam2img)

    # bring in fixed point

    if needle_pos_file is not None:
        # pts_amount = 1 # single point
        aug_pts = np.concatenate((needle_pts, np.ones((needle_pts.shape[0], 1))), axis=1) # multi point
        # aug_pts = np.append(pt, 1) # single point
        proj_pts = (P1 @ aug_pts.T).T
        proj_pts /= proj_pts[:, 2:].copy() + 1e-7 # multi point
        # proj_pts /= proj_pts[2].copy() + 1e-7 # single point
        # proj_pts = np.array((proj_pts[:, 1], proj_pts[:, 0], needle_pts[:, 2])).T
        proj_pts[:, 2] = np.array((needle_pts[:, 2])).T

        # point = np.array([180.28571429, 294, 189.06850680974424]) # trial 24, end of needle
        # point = np.array([246., 344., 189]) # trial 24, end of thread/start of needle
        keypoints[order[-pts_amount:]] = proj_pts
        constr_lower_d[order[-pts_amount:]] = proj_pts[:, 2] - 1e0 # multi point
        constr_upper_d[order[-pts_amount:]] = proj_pts[:, 2] + 1e0 # multi point
        # constr_lower_d[order[-pts_amount]] = proj_pts[2] - 1e0 # single point
        # constr_upper_d[order[-pts_amount]] = proj_pts[2] + 1e0 # single point
    # tr_trace()
    keypt_u = init_u[keypoint_idxs]
    k = 3
    num_ctrl = len(knots)-k-1
    num_constr = len(keypt_u)*3
    spline = None

def get_needle_point(pos_file):
    r = 8.23
    import pickle
    import open3d as o3d
    with open(pos_file, 'rb') as f:
        data = pickle.load(f)

    needle_pos = np.array([data.get('x'), data.get('y'), data.get('z'), data.get('qw'), data.get('qx'), data.get('qy'), data.get('qz')]) * 1000
    conn_pts = np.array([[-r, -(r/2 - 1), 0], [-r, -r/2, 0]]) # connection point, plus next closest point on needle
    R = o3d.geometry.get_rotation_matrix_from_quaternion(needle_pos[3:])
    # single pt
    # conn_pt = conn_pt @ R.T
    # conn_pt = conn_pt + needle_pos[:3]

    # multi pts
    conn_pts = conn_pts @ R.T
    conn_pts = np.stack([needle_pos[:3], needle_pos[:3]]) - conn_pts

    return conn_pts
'''