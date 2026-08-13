"""
spline_ekf.py
─────────────
Kalman filter over a fixed-knot cubic spline whose state is the M 3-D
control-point positions.

  Process model : rigid-body tool motion (self.trans) + spatially-varying
                  elastic residual noise (large near curr_T, small far away).
  Measurement   : N matched keypoints from warm_ordering, each with a known
                  spline parameter t_i.  Observation equation is LINEAR in the
                  control points (cardinal-basis interpolation), so this is a
                  plain linear KF — no Jacobians needed.

Dependency: scipy (CubicSpline), numpy.
"""

import numpy as np
from scipy.interpolate import CubicSpline

# All tuning values live in ekf_params.py (single source of truth shared with
# keypt_ordering / warm_start / optim / the ros node).  The constructor
# defaults below ARE those values, so Order() builds the filter with no args.
from thread_reconstruction import ekf_params as EP


class SplineEKF:
    """
    State  x  : (3M,)   flattened control points [p_0; p_1; … p_{M-1}]
                         each p_i ∈ ℝ³ in 3-D camera frame.
    Cov    P  : (3M, 3M)

    Knot locations t_ctrl = linspace(0, 1, M) are fixed for the lifetime of
    the filter.  The cardinal basis functions are precomputed once at init and
    never change.
    """

    def __init__(self,
                 n_ctrl: int   = EP.N_CTRL,
                 sigma_meas: float    = EP.SIGMA_MEAS,
                 sigma_meas_z: float  = EP.SIGMA_MEAS_Z,
                 sigma_proc_base: float = EP.SIGMA_PROC_BASE,
                 sigma_proc_tip: float  = EP.SIGMA_PROC_TIP,
                 deform_radius: float   = EP.DEFORM_RADIUS,
                 chi2_thresh: float     = EP.CHI2_UPDATE_3D,
                 motion_trans_ref: float = EP.MOTION_TRANS_REF,
                 motion_rot_ref: float   = EP.MOTION_ROT_REF,
                 q_motion_floor: float   = EP.Q_MOTION_FLOOR,
                 motion_decay: float     = EP.MOTION_DECAY,
                 sigma_smooth: float     = EP.SIGMA_SMOOTH,
                 sigma_stretch: float    = EP.SIGMA_STRETCH,
                 len_track_alpha: float  = EP.LEN_TRACK_ALPHA,
                 sigma_end_straight: float = EP.SIGMA_END_STRAIGHT,
                 end_span: int           = EP.END_SPAN,
                 gate_recover_frac: float = EP.GATE_RECOVER_FRAC,
                 gate_recover_frames: int = EP.GATE_RECOVER_FRAMES,
                 motion_recover_max: float = EP.MOTION_RECOVER_MAX,
                 recover_P0_scale: float  = EP.RECOVER_P0_SCALE,
                 motion_prior_floor: float = EP.MOTION_PRIOR_FLOOR,
                 gate_motion_gain: float  = EP.GATE_MOTION_GAIN):
        """
        n_ctrl          number of control points  (state dim = 3 * n_ctrl)
        sigma_meas      LATERAL (x, y) measurement noise std, in the same units
                        as the keypoint positions returned by warm_ordering.
                        Small — the 2-D projection is accurate.
        sigma_meas_z    DEPTH (z) measurement noise std.  Stereo/triangulated
                        depth is much noisier than the lateral position, so this
                        is typically several× sigma_meas.  Defaults to
                        4·sigma_meas if not given.  Keeping depth loose stops
                        noisy z from (a) over-rejecting keypoints at the χ² gate
                        and (b) dragging the spline into 3-D waviness.
        sigma_proc_base process noise std far from the tool tip (rigid motion
                        is a near-perfect model there)
        sigma_proc_tip  process noise std right at the tool tip (elastic
                        deformation is largest here)
        deform_radius   e-folding distance for elevated process noise;
                        tune to the physical reach of tool-induced deformation
        chi2_thresh     Mahalanobis gating threshold in χ²(3) units;
                        7.81 ≈ 95 % confidence, 11.34 ≈ 99 %
        motion_trans_ref tool translation (same units as tool_pos) that counts
                        as "full" motion for process-noise scaling.  A per-frame
                        translation at/above this uses the full Q.
        motion_rot_ref  tool rotation angle in radians that counts as "full"
                        motion (defaults ≈ 2.9°).
        q_motion_floor  fraction of Q retained when the tool is perfectly
                        stationary.  This is the "memory" knob: with no PSM
                        movement the process noise collapses to this floor, the
                        Kalman gain shrinks, and the estimate leans on its prior
                        instead of chasing per-frame measurement noise — so the
                        reconstruction stops fluctuating.  A small non-zero floor
                        keeps the filter from locking up / becoming overconfident.
        sigma_stretch   std of the LENGTH pseudo-measurement that keeps the
                        spline from growing/shrinking past its seeded arc
                        length.  A thread is inextensible, but the filter has no
                        notion of arc length, so measurement noise (especially
                        at the under-observed ends) and the non-rigid predict
                        warp let the control polygon spread out over frames.
                        Each frame this penalises every control-polygon edge's
                        length toward its value at initialize() (‖e_j‖ ≈ ref_j,
                        linearised along the current edge direction), so the
                        curve can bend/loop freely — only its LENGTH is held.
                        Units: allowed per-edge length deviation (spline units).
                        Smaller → length held tighter; larger → looser; ≤0 or
                        None disables.  The reference length is NOT frozen at
                        the seed — it tracks the max observed length (see
                        len_track_alpha), so an occluded seed still converges to
                        the true length.
        len_track_alpha EMA rate for the tracked reference length.  Each update
                        the reference edge lengths are scaled so the total
                        follows the running MAX of an EMA-smoothed observed
                        length: the EMA (this rate) averages out per-frame
                        noise, the max ratchets up on genuine length increases
                        and never shrinks on occlusion.  Smaller → slower, more
                        noise-robust growth; larger → faster to react.  Only
                        used when sigma_stretch is active.
        sigma_smooth    std of the SMOOTHNESS pseudo-measurement  D3·x ≈ 0
                        folded into every update — the same bending-energy idea
                        optim uses when fitting the reconstructed thread, here as
                        a soft prior instead of a QP objective.  D3 takes third
                        differences of consecutive control points (per
                        coordinate), so alternating-curvature WOBBLE is damped
                        while constant-curvature loops (whose third differences
                        are naturally small) pass almost untouched.  Units:
                        allowed third-difference magnitude (spline units).
                        Smaller → stiffer/smoother spline; larger → follows
                        measurements more literally; ≤0 or None disables.
        sigma_end_straight std of a boundary CURVATURE pseudo-measurement
                        (2nd difference ≈ 0) applied only to the end_span
                        control points at each end — a natural-spline end
                        condition.  The end control points are usually
                        under-observed (the thread ends / occludes there), so
                        the smoothness prior extrapolates their curvature and
                        the length prior pushes spare length outward, and over
                        many frames the tips CURL UP and FOLD BACK.  Penalising
                        end curvature straightens the last segments: it costs
                        almost nothing for a mild real end-curve but strongly
                        resists a curl/fold.  Units: allowed end 2nd-difference
                        (spline units); smaller → straighter ends; ≤0/None off.
        end_span        number of control points at EACH end covered by the
                        end-straightness penalty (≥3; larger stiffens more of
                        the ends).
        gate_recover_frac  divergence monitor: if fewer than this FRACTION of
                        candidate keypoints pass the image-frame χ² gate, the
                        filter's state disagrees with the data.  A few bad
                        frames during manipulation can corrupt the state; once
                        the tool stops the process noise collapses, P shrinks,
                        and the (confidently wrong) filter then REJECTS the good
                        keypoints — locked out, never reconverging.  When the
                        gate pass-rate stays below this for gate_recover_frames
                        consecutive frames AND the tool is nearly still (motion
                        < motion_recover_max), P is reset to recover_P0_scale·I
                        so the next update re-acquires the data.  Lower → only
                        recover on severe disagreement.  ≤0 disables recovery.
        gate_recover_frames consecutive low-gate frames required before recovery
                        fires (avoids reacting to a single bad frame).
        motion_recover_max recovery only fires when the motion scalar (0…1, see
                        predict) is below this — i.e. manipulation has stopped.
                        During active manipulation Q is high and the gate is
                        already loose, so no lock-out occurs.
        motion_prior_floor MOTION-ADAPTIVE priors.  The shape priors
                        (smoothness, length, end-straight) are strongest when
                        the tool is still (denoise / stabilise) and RELAXED when
                        the thread is being moved (so measurements drive the
                        deforming shape instead of the prior fighting it).  Each
                        prior's weight is scaled by
                          prior_scale = floor + (1-floor)·(1-motion),
                        so still (motion=0) → full strength, full motion=1 →
                        `motion_prior_floor` of it.  1.0 disables adaptation
                        (constant priors); smaller floor → priors relax more
                        during motion (better tracking, less denoising while
                        moving).
        gate_motion_gain MOTION-ADAPTIVE gate.  The χ² gate threshold is
                        multiplied by (1 + gain·motion) so a fast-moved thread
                        (large innovation) is not rejected during manipulation;
                        the gate stays tight at rest.  0 disables.
        recover_P0_scale target largest-diagonal variance P is INFLATED up to on
                        recovery (never a reset — see maybe_recover_from_
                        divergence).  P is scaled by a single factor so its
                        biggest diagonal reaches this value, which loosens the
                        gate enough to re-acquire the good data while KEEPING P's
                        correlation structure (so the state's direction survives
                        and the thread can't flip).  Must be LARGE (default 400):
                        a locked-out state can be far from the data, so P must be
                        big enough that the recovery update leans hard on the
                        current keypoints.  Too small and the gate still rejects
                        the good data and the filter stays stuck.
        """
        self.M               = n_ctrl
        self.sigma_meas      = sigma_meas
        self.sigma_meas_z    = sigma_meas_z if sigma_meas_z is not None else 4.0 * sigma_meas
        self.sigma_proc_base = sigma_proc_base
        self.sigma_proc_tip  = sigma_proc_tip
        self.deform_radius   = deform_radius
        self.chi2_thresh     = chi2_thresh
        self.motion_trans_ref = motion_trans_ref
        self.motion_rot_ref   = motion_rot_ref
        self.q_motion_floor   = q_motion_floor
        self.motion_decay     = motion_decay

        # Per-keypoint 3-D measurement covariance: tight laterally, loose in
        # depth.  Reused by both the update and the χ² gate.
        self._R_block = np.diag([self.sigma_meas ** 2,
                                 self.sigma_meas ** 2,
                                 self.sigma_meas_z ** 2])

        self.t_ctrl = np.linspace(0.0, 1.0, n_ctrl)

        self.x: np.ndarray | None = None   # (3M,)
        self.P: np.ndarray | None = None   # (3M, 3M)

        # ── Smoothness + boundary-curvature penalties (see _build_penalties) ──
        # Rebuilt whenever M changes (adaptive control-point count), so they
        # live in a helper instead of inline here.
        self.sigma_smooth       = sigma_smooth
        self.sigma_end_straight = sigma_end_straight
        self.end_span           = end_span
        self._build_penalties()

        # Length prior state (see sigma_stretch / len_track_alpha).
        #   _edge_profile : seed edge lengths normalised to sum 1 (the shape).
        #   _len_ref      : tracked reference TOTAL length = running max of…
        #   _len_ema      : …an EMA-smoothed observed polygon length.
        #   _edge_ref     : per-edge targets = _edge_profile * _len_ref.
        self.sigma_stretch   = sigma_stretch
        self.len_track_alpha = len_track_alpha
        self._edge_profile = None
        self._len_ref      = None
        self._len_ema      = None
        self._edge_ref     = None

        # Divergence / lock-out recovery (see gate_recover_frac).
        self.gate_recover_frac   = gate_recover_frac
        self.gate_recover_frames = gate_recover_frames
        self.motion_recover_max  = motion_recover_max
        self.recover_P0_scale    = recover_P0_scale
        self.motion_prior_floor  = motion_prior_floor
        self.gate_motion_gain    = gate_motion_gain
        self._last_motion     = 0.0    # set by predict() (held/decayed motion)
        self._motion_raw      = 0.0    # instantaneous motion, before the hold
        self._last_gate_frac  = 1.0    # set by the match gate (mahalanobis_gate
                                       # with update_monitor=True)
        self._low_gate_streak = 0
        # Trigger state of the last maybe_recover_from_divergence() call, for
        # debug visualisation (see that method).
        self._last_recovery_info = {"fired": False, "gate_frac": 1.0,
                                    "frac_thresh": gate_recover_frac,
                                    "streak": 0,
                                    "frames_thresh": gate_recover_frames,
                                    "motion": 0.0,
                                    "motion_max": motion_recover_max,
                                    "factor": 1.0}
        # Whether the last update() reversed the spline's t=0→t=1 sense (the
        # under-observed ends crossing over), for the debug view.  See update().
        self._last_orient_info = {"flipped": False, "d_same": 0.0,
                                  "d_flip": 0.0, "corrected": False}
        # Reference near-grasp tangent expressed in the TOOL frame (rigidly
        # attached to the gripper), used by lock_orientation_to_grasp() to hold
        # the t-direction fixed.  None until anchored on the first frame a grasp
        # position is available.
        self._grip_tangent_tool = None
        # Id of the tool the reference is anchored to; a change (bimanual grasp
        # handoff) forces a re-anchor.  See lock_orientation_to_grasp().
        self._grip_tool_id = None
        # Control-point index the reference tangent was anchored at, and how far
        # (in knots) the nearest-to-grasp index may drift before the comparison
        # is considered to be at a DIFFERENT thread point (→ re-anchor, never
        # fire).  Guards against the spurious reversal that happened when the
        # lock anchored mid-approach and the fresh grasp landed elsewhere.
        self._grip_idx = None
        self._GRIP_IDX_TOL = EP.GRIP_IDX_TOL

        # ── Length-adaptive control-point count (see maybe_adapt_ctrl_count) ──
        # A shrunken visible thread with the full knot count crammed onto it has
        # the spatial DOF to fold its surplus into high-frequency WAVINESS
        # (scribble → gate collapse → cold re-acquire → flip risk).  The knot
        # SPACING recorded at initialize() is held roughly constant instead:
        # when the observed thread length shrinks/grows past hysteresis, the
        # state+covariance are RESAMPLED (not reseeded) to the matching count.
        self._M_init       = n_ctrl   # count ceiling (seed count)
        self._M_MIN        = EP.ADAPT_M_MIN   # never fewer knots than this
        self._ctrl_spacing = None     # set at initialize(): L_seed/(M-1)
        self._L_obs_ema    = None     # EMA of observed thread length
        self._ADAPT_EMA_ALPHA = EP.ADAPT_EMA_ALPHA  # observed-length smoothing
        self._ADAPT_HYST      = EP.ADAPT_HYST  # knots of change to resample
        # The change must PERSIST: the RAW per-frame length must want a
        # different count for this many consecutive frames before the state
        # is resampled.  The EMA alone could cross the hysteresis off a
        # single bad frame (a 50% length drop moves it 15% in one step) and
        # its lag kept it crossed during recovery — so one occluded frame
        # triggered the lossy resample (the frame-6 state damage).  The raw
        # streak resets the moment a single frame reads normal again.
        self._ADAPT_STREAK_FRAMES = EP.ADAPT_STREAK_FRAMES
        self._adapt_streak        = 0

        # Current frame number, set by the caller each frame (= the ros node's
        # self._vis_z_count, the number in the debug_z_noise_<frame>_*.png names)
        # so every EKF log line can be cross-referenced to a debug PNG.  None →
        # logs print without a frame tag.
        self.frame = None

        # Cardinal basis: c_j(t) = 1 at t_ctrl[j], 0 at all other knots.
        # These depend only on the knot locations (fixed), so we build them
        # once and reuse every frame.
        self._cardinal: list[CubicSpline] = self._build_cardinal()

    def _log(self, msg: str) -> None:
        """print() prefixed with the current frame number for cross-referencing
        against the debug_z_noise_<frame>_*.png views."""
        print(f"[frame {self.frame}] {msg}" if self.frame is not None else msg)

    # ══════════════════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════════════════

    def initialize(self, warm_thread, P0_scale: float = EP.P0_SCALE) -> None:
        """
        Seed the filter by sampling an existing warm_thread callable.

        warm_thread : callable  t ∈ [0,1]^D → (D, 3)  (the same contract
                      as the warm_thread argument to warm_ordering).
        P0_scale    : initial variance per control-point coordinate.
                      9.0 ≈ ±3 units of uncertainty; raise if the first
                      warm_thread is known to be noisy.
        """
        ctrl_pts = warm_thread(self.t_ctrl)           # (M, 3)
        self.x   = ctrl_pts.flatten().copy()
        self.P   = np.eye(3 * self.M) * P0_scale
        # Length prior: seed the edge-length PROFILE (shape) and the tracked
        # reference length.  The reference is not frozen — it will follow the
        # max observed length during update() (see _track_ref_length).
        seed_edges = np.linalg.norm(np.diff(ctrl_pts, axis=0), axis=1)
        total_len  = float(seed_edges.sum())
        if self.sigma_stretch is not None and self.sigma_stretch > 0:
            self._edge_profile = seed_edges / (total_len + 1e-9)   # sums to 1
            self._len_ref      = total_len
            self._len_ema      = total_len
            self._len_seed     = total_len   # fixed baseline for the ratchet log
            self._edge_ref     = self._edge_profile * self._len_ref
        # Knot spacing the adaptive control-point count preserves (see
        # maybe_adapt_ctrl_count): seed length per seed knot interval.
        self._ctrl_spacing = total_len / max(self.M - 1, 1)
        self._L_obs_ema    = total_len
        self._log(f"SplineEKF: initialised  M={self.M}  "
              f"P0={P0_scale:.1f}·I  "
              f"ctrl_pts range [{ctrl_pts.min():.1f}, {ctrl_pts.max():.1f}]  "
              f"polygon_len={total_len:.1f}")

    def predict(self, trans: np.ndarray, tool_pos_3d: np.ndarray) -> None:
        """
        Predict step driven by rigid-body tool kinematics.

        trans       : (4, 4) relative transform between frames,
                      i.e. self.trans from the calling class
                      ( = curr_T @ inv(prev_T) in camera frame ).
        tool_pos_3d : (3,) grasp position at the PREVIOUS frame in 3-D camera
                      coords, i.e. prev_T[:3, 3] — the pivot of the warp (must
                      match warm_start's warp: pivoting on the pre-move grasp
                      makes w=1 the exact rigid transform curr_T @ inv(prev_T)).
                      Also scales the spatially-varying process noise.
        """
        if self.x is None:
            raise RuntimeError("SplineEKF: call initialize() before predict().")

        R = trans[:3, :3]   # (3, 3) rotation part
        t = trans[:3,  3]   # (3,)   translation part
        M = self.M

        # ── Apply tool motion: distance-weighted translation + rotation ────
        # BOTH components decay with distance from the grasp,
        # w = exp(-d/deform_radius): only the near-tool portion of the thread
        # follows the tool, farther parts lag behind (slack/anchored thread is
        # not dragged whole).  Same warp as warm_start.refresh_warm_start so
        # the EKF prediction and the transformed warm thread agree.  At w=1
        # (at the grasp, pivot = prev grasp) this is the exact rigid transform;
        # at w=0 the thread does not move.
        ctrl_pts = self.x.reshape(M, 3)                        # (M, 3)
        delta    = ctrl_pts - tool_pos_3d                      # about the grasp
        rot_dev  = (R @ delta.T).T - delta                    # (R - I)(p - pivot)
        dist     = np.linalg.norm(delta, axis=1)
        w        = np.exp(-dist / max(self.deform_radius, 1e-6))   # (M,)
        ctrl_pts_pred = ctrl_pts + w[:, None] * (t + rot_dev)
        self.x        = ctrl_pts_pred.flatten()

        # ── Block-diagonal F:  F[3i:3i+3, 3i:3i+3] = I + w_i (R - I)  ─────
        # Per-point Jacobian consistent with the weighted warp above.
        I3 = np.eye(3)
        F = np.zeros((3*M, 3*M))
        for i in range(M):
            F[3*i:3*i+3, 3*i:3*i+3] = I3 + w[i] * (R - I3)

        # ── Motion-scaled process noise (temporal memory) ──────────────────
        # The thread only deforms when the tool moves it.  Scale the whole Q by
        # how much the PSM actually moved this frame: near-zero motion → Q
        # collapses to q_motion_floor·Q, the Kalman gain shrinks, and the state
        # holds its previous estimate instead of chasing measurement noise.
        # This is what keeps the reconstruction from fluctuating while the PSM
        # is stationary.
        t_norm = float(np.linalg.norm(t))
        theta  = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
        motion_raw = min(1.0, t_norm / max(self.motion_trans_ref, 1e-9)
                              + theta  / max(self.motion_rot_ref,   1e-9))
        # ── Asymmetric motion hold ────────────────────────────────────────
        # motion rises INSTANTLY (a fast move must loosen Q/gate/prior on the
        # same frame it happens) but FALLS gradually: motion_held decays by
        # motion_decay each still frame.  This removes the cliff when the tool
        # stops — q_scale, prior_scale and the gate all ease back to rest over
        # ~5 frames, so the thread settles instead of snapping the instant
        # motion hits 0 (which was making it jump around after motion).
        motion = max(motion_raw, self.motion_decay * self._last_motion)
        self._motion_raw  = motion_raw
        q_scale = self.q_motion_floor + (1.0 - self.q_motion_floor) * motion
        self._last_motion = motion       # for the divergence-recovery monitor

        # ── Spatially-varying process noise ────────────────────────────────
        Q = self._build_Q(ctrl_pts_pred, tool_pos_3d) * q_scale

        # ── Covariance propagation ─────────────────────────────────────────
        self.P = F @ self.P @ F.T + Q

        # Report Q at the control point nearest / farthest from the tool tip
        # (by distance, not by index) so the near-tool=high, far=low profile
        # is visible in the log.
        d_ctrl   = np.linalg.norm(ctrl_pts_pred - tool_pos_3d, axis=1)
        i_near   = int(np.argmin(d_ctrl))
        i_far    = int(np.argmax(d_ctrl))
        self._log(f"SplineEKF predict: "
              f"max_diag(P)={np.diag(self.P).max():.2f}  "
              f"motion={motion:.2f}(raw={motion_raw:.2f}) q_scale={q_scale:.2f}  "
              f"Q_near_tool={Q[3*i_near, 3*i_near]:.2f}  "
              f"Q_far_tool={Q[3*i_far, 3*i_far]:.2f}")

    def update(self,
               matched_t: np.ndarray,
               matched_z_3d: np.ndarray,
               conf: np.ndarray = None) -> dict:
        """
        Measurement update from warm_ordering observations.

        matched_t    : (N,)   spline parameter values for matched keypoints,
                               i.e. matched_t_dedup from warm_ordering.
        matched_z_3d : (N, 3) corresponding 3-D camera-frame positions,
                               i.e. unproject(keypoints[matched_kpt_ids_dedup]).
        conf         : optional (N,) per-observation stereo confidence in
                               [0, 1] (keypt_selection full_conf).  Inflates
                               each observation's noise as R_i = R /
                               max(conf_i, EP.EKF_CONF_R_FLOOR), so keypoints
                               with ambiguous stereo (epipolar-parallel
                               sections) barely tug the filter and it coasts
                               on the motion model there.  None → shared R.

        Returns a diagnostics dict (innovation, S, K) for debug plotting.
        """
        N = len(matched_t)
        if N < 2:
            self._log("SplineEKF update: too few observations, skipping.")
            return {}
        M = self.M

        H      = self._build_H(matched_t)              # (3N, 3M)

        z      = matched_z_3d.flatten()                # (3N,)
        innov  = z - H @ self.x                        # (3N,)

        # ── Information-form update ────────────────────────────────────────
        # The classic form solves S = HPH'+R, a (3N, 3N) system — 336x336 for
        # ~112 observations, even though the state is only 3M=48 dimensional.
        # Algebraically identical (R is block-diagonal → R^-1 is trivial):
        #   P_post = (P^-1 + H' R^-1 H)^-1          (3M, 3M — 48x48 solve)
        #   x_post = x + P_post H' R^-1 innov
        # ~300x fewer flops in the solve, so a busy process can't stretch it.
        Rinv_block = np.linalg.inv(self._R_block)      # (3, 3)
        # H' R^-1 applied blockwise: scale each observation's 3 rows by Rinv.
        HtRinv = (H.reshape(N, 3, 3 * M).transpose(0, 2, 1) @ Rinv_block)  # (N, 3M, 3)
        if conf is not None:
            conf = np.asarray(conf, dtype=float)
            if conf.shape == (N,):
                # Confidence weighting: R_i = R / w_i is, in information form,
                # just Rinv_i = w_i · Rinv — one scalar per observation block.
                # Applied before both info and rhs below, so gain AND
                # covariance see the inflated noise consistently.
                w = np.clip(conf, EP.EKF_CONF_R_FLOOR, 1.0)
                HtRinv = HtRinv * w[:, None, None]
                self._log(f"SplineEKF update: conf-weighted R  "
                          f"w μ={w.mean():.2f} min={w.min():.2f} "
                          f"(<0.5: {int((w < 0.5).sum())}/{N})")
            else:
                self._log(f"SplineEKF update: conf shape {conf.shape} != "
                          f"({N},) — ignoring confidence weighting.")
        HtRinv = HtRinv.transpose(1, 0, 2).reshape(3 * M, 3 * N)
        info   = np.linalg.inv(self.P) + HtRinv @ H    # (3M, 3M)
        rhs    = HtRinv @ innov                        # (3M,)

        # ── Smoothness pseudo-measurement (optim's bending term as a prior) ──
        # Augment the MAP problem with  w_s·‖D3 x‖²  (D3 x ≈ 0, σ = sigma_smooth):
        #   x⁺ = argmin (x-x̄)ᵀP⁻¹(x-x̄) + (z-Hx)ᵀR⁻¹(z-Hx) + w_s‖D3 x‖²
        # In information form this adds S_pen to the information matrix and,
        # writing x = x̄ + δ, the term  -S_pen·x̄  to the right-hand side — i.e.
        # the update is pulled toward a low-bending shape exactly as optim's QP
        # objective pulls the reconstructed thread.  Wobble (alternating
        # curvature → large third differences) is damped; smooth loops
        # (small third differences) are barely affected.
        # Motion-adaptive prior weight: full when still, relaxed toward
        # motion_prior_floor at full motion, so the shape priors stop fighting
        # the thread while it is actively deforming (see motion_prior_floor).
        prior_scale = (self.motion_prior_floor
                       + (1.0 - self.motion_prior_floor) * (1.0 - self._last_motion))

        if self._S_pen is not None:
            info = info + prior_scale * self._S_pen
            rhs  = rhs - prior_scale * (self._S_pen @ self.x)

        # ── Boundary curvature prior (anti curl/fold at the ends) ────────────
        if self._E_pen is not None:
            info = info + prior_scale * self._E_pen
            rhs  = rhs - prior_scale * (self._E_pen @ self.x)

        # ── Length pseudo-measurement (arc-length prior; see sigma_stretch) ──
        # For each control-polygon edge e_j = p_{j+1}-p_j, penalise its length
        # toward the seeded reference ref_j.  ‖e_j‖ = û_jᵀ e_j with û_j the
        # current edge direction, so linearised about the predicted state this
        # is a LINEAR measurement  A x ≈ ref  with A_j = [-û_j, +û_j] on blocks
        # (j, j+1).  Only length is constrained (direction û_j is free), so the
        # thread bends/loops freely but can't stretch past its physical length.
        if (self.sigma_stretch is not None and self.sigma_stretch > 0
                and self._edge_ref is not None):
            xb    = self.x.reshape(M, 3)
            edges = np.diff(xb, axis=0)                 # (M-1, 3)
            lens  = np.linalg.norm(edges, axis=1)
            u     = edges / (lens[:, None] + 1e-9)      # (M-1, 3) unit dirs
            A     = np.zeros((M - 1, 3 * M))
            for j in range(M - 1):
                A[j, 3 * j:3 * j + 3]           = -u[j]
                A[j, 3 * (j + 1):3 * (j + 1) + 3] =  u[j]
            wL      = prior_scale / (self.sigma_stretch ** 2)   # motion-adaptive
            innov_L = self._edge_ref - lens             # ref − current length
            info    = info + wL * (A.T @ A)
            rhs     = rhs + wL * (A.T @ innov_L)

        P_post = np.linalg.inv(info)
        P_post = 0.5 * (P_post + P_post.T)             # guard against drift

        # ── Orientation-flip detector (diagnostic) ─────────────────────────────
        # The shape priors are direction-symmetric, so an update can slide the
        # under-observed END control points across each other and REVERSE the
        # spline's t=0→t=1 sense without any recovery/reseed firing (the 2-D gate
        # is position-only → orientation-blind).  Compare the endpoints before vs
        # after this update: if the (end0,end1) pair is spatially closer FLIPPED
        # than SAME, the update reversed the thread this frame.  Recorded in
        # self._last_orient_info for the debug view; does not alter the update.
        ends_pre  = self.x.reshape(M, 3)[[0, -1]]
        x_post    = self.x + P_post @ rhs
        ends_post = x_post.reshape(M, 3)[[0, -1]]
        d_same = (np.linalg.norm(ends_post[0] - ends_pre[0])
                  + np.linalg.norm(ends_post[1] - ends_pre[1]))
        d_flip = (np.linalg.norm(ends_post[0] - ends_pre[1])
                  + np.linalg.norm(ends_post[1] - ends_pre[0]))
        self._last_orient_info = {"flipped": bool(d_flip < d_same),
                                  "d_same": float(d_same),
                                  "d_flip": float(d_flip),
                                  "corrected": False}
        if self._last_orient_info["flipped"]:
            self._log(f"SplineEKF update: ORIENTATION FLIP this update "
                  f"(d_same={d_same:.1f} > d_flip={d_flip:.1f}) — ends crossed.")

        self.x = x_post
        self.P = P_post

        # Track the reference length toward the max observed (post-update)
        # length, so the length prior converges to the true thread length even
        # from an occluded seed (see _track_ref_length).
        self._track_ref_length()

        innov_norms = np.linalg.norm(innov.reshape(N, 3), axis=1)
        # len_ref is a running MAX of an EMA of the filter's OWN polygon length
        # (_track_ref_length), so it can only climb.  A kinked polygon is longer
        # than a smooth one, so if wobble is feeding the length prior the ratio
        # below rises monotonically on a static thread — which would mean the
        # prior is defending the drift rather than the thread's real length.
        _lr   = self._len_ref if self._len_ref is not None else float('nan')
        _seed = getattr(self, '_len_seed', None)
        self._log(f"SplineEKF update: N={N}  "
              f"innov μ={innov_norms.mean():.2f}  "
              f"σ={innov_norms.std():.2f}  "
              f"max={innov_norms.max():.2f}  "
              f"| polygon_len={getattr(self, '_len_obs_last', float('nan')):.1f} "
              f"len_ema={self._len_ema if self._len_ema is not None else float('nan'):.1f} "
              f"len_ref={_lr:.1f}"
              + (f" ({_lr / _seed:.3f}x seed)" if _seed else ""))
        return {"innovation": innov.reshape(N, 3)}

    def _track_ref_length(self) -> None:
        """Track the length-prior reference to an EMA of the observed length.

        Called once per update with the posterior state.  The reference follows
        that EMA in BOTH directions.

        The running-MAX ratchet this replaces could only ever climb, so a
        single over-long seed became a permanent target: run.log showed the
        filter seeded at polygon_len=275.6 on a thread optim measured at 149.7,
        and `len_ref=275.6 (1.000x seed)` on all 71 frames afterwards.  The
        length prior then spent the whole session pulling the state back toward
        a length that did not exist, and since the well-observed middle is
        pinned by the image-plane match, the surplus went into z at the
        under-observed ends — the depth runaway.

        Trade-off this reintroduces: the reference now follows the filter's own
        polygon, so it no longer anchors absolute length — it damps CHANGE
        (~1/len_track_alpha frames of lag) rather than fixing scale, and a slow
        genuine stretch is no longer opposed.  It also shrinks under occlusion,
        which the ratchet existed to prevent."""
        if self._edge_profile is None or self._len_ref is None:
            return
        xb    = self.x.reshape(self.M, 3)
        L_obs = float(np.sum(np.linalg.norm(np.diff(xb, axis=0), axis=1)))
        self._len_obs_last = L_obs          # exposed for the update log
        a = self.len_track_alpha
        self._len_ema = a * L_obs + (1.0 - a) * self._len_ema
        self._len_ref = self._len_ema
        self._edge_ref = self._edge_profile * self._len_ref

    def sample_pos_sigma(self, t) -> np.ndarray | None:
        """Per-sample position standard deviation of the posterior spline.

        Propagates the state covariance through the cardinal basis to each
        sampled t — Σ_i = B_i P B_iᵀ, a 3×3 per sample — and returns
        sqrt(diag(Σ_i)) as (N, 3) in (x, y, z).

        This is the SAME propagation the χ² match gate uses (see _gate_mask),
        with one deliberate difference: R is NOT added.  There it forms the
        innovation covariance of a future MEASUREMENT; here we want the
        uncertainty of the CURVE itself, which is what a published depth
        envelope should describe.

        Returns None when the filter has no state yet.
        """
        if self.x is None or self.P is None:
            return None
        t = np.atleast_1d(np.asarray(t, dtype=float))
        M = self.M
        B = np.column_stack([c(t) for c in self._cardinal])        # (N, M)
        P4 = self.P.reshape(M, 3, M, 3)
        Sig = np.einsum('na,abcd,nc->nbd', B, P4, B, optimize=True)  # (N,3,3)
        var = np.einsum('nii->ni', Sig)                             # diagonals
        return np.sqrt(np.maximum(var, 0.0))

    def degrade(self, ambiguity: float,
                gain: float = EP.AMB_P_GAIN,
                max_diag: float = EP.AMB_P_MAXDIAG) -> None:
        """
        Inflate the state covariance in proportion to an external ambiguity
        score in [0, 1], to be called BEFORE the measurement update.

        A higher ambiguity makes the filter less confident (P ×(1 + gain·amb)),
        which raises the Kalman gain so the following update leans on the (raw)
        measurements instead of the prior shape.  A thread that has become
        ambiguous is therefore re-derived from data over the next few frames
        rather than being locked to a stale prediction.  ambiguity ≤ 0 is a
        no-op.  The result is rescaled so the largest diagonal entry never
        exceeds `max_diag`, so a run of ambiguous frames can't blow up the
        variance unbounded.
        """
        if self.P is None or ambiguity <= 0.0:
            return
        factor = 1.0 + gain * float(ambiguity)
        self.P = self.P * factor
        dmax = float(np.diag(self.P).max())
        if dmax > max_diag:
            self.P *= max_diag / dmax                # keeps correlation structure
        self._log(f"SplineEKF degrade: ambiguity={ambiguity:.2f} → "
              f"P ×{factor:.2f}  (max_diag={float(np.diag(self.P).max()):.1f})")

    def get_spline(self) -> CubicSpline:
        """
        Return a CubicSpline callable  f(t) → (len(t), 3)  built from
        the current (post-predict or post-update) control points.

        Drop-in replacement for the warm_thread argument to warm_ordering.
        """
        if self.x is None:
            raise RuntimeError("SplineEKF: call initialize() before get_spline().")
        return CubicSpline(self.t_ctrl, self.x.reshape(self.M, 3))

    def update_from_thread(self, thread, reliability=None) -> dict:
        """Correction step for the KF↔optim loop: measure a robust external
        spline (optim's box-QP fit) instead of the raw keypoints.  The thread
        is sampled at the M control-point parameters t_ctrl and fed to update()
        as M 3-D observations, so the filter fuses optim's per-frame estimate
        over time (temporal denoising + prediction) without ever trusting the
        raw, sometimes-persistently-wrong keypoints.

        reliability : optional per-sample reliability of the thread over
        t ∈ [0, 1] (thread_specs['reliability'], already gap-degraded and
        interpolation-degraded upstream).  Interpolated at t_ctrl and passed
        to update() as conf, so thread stretches that carried no fresh data
        this frame (occlusion, ambiguous stereo) barely correct the filter —
        it coasts on its prediction there instead of fusing optim's
        interpolation as if it were a measurement.

        DIRECTION GUARD: the thread is supposed to run the same direction as
        the filter (match_warm_order aligns it), but a cold-ordering re-acquire
        can slip through reversed — and a reversed thread here is a LABELLING
        mismatch, not a measurement: feeding it drags the whole state across
        itself and flips the reconstruction (innov tens of units, "ends
        crossed").  So the samples' endpoints are paired against the current
        state first; if the flipped pairing is closer, the sample ORDER is
        reversed (same curve, matching labelling) before the update."""
        if self.x is None or thread is None:
            return {}

        def _conf_at_ctrl():
            # reliability is sampled uniformly over the thread's t ∈ [0, 1]
            # (200 published samples); resample it at the CURRENT t_ctrl.
            if reliability is None:
                return None
            r = np.asarray(reliability, dtype=float).ravel()
            if r.size < 2 or not np.all(np.isfinite(r)):
                return None
            return np.interp(self.t_ctrl, np.linspace(0.0, 1.0, r.size), r)

        z = np.asarray(thread(self.t_ctrl), dtype=float)
        conf = _conf_at_ctrl()
        if z.shape != (self.M, 3) or not np.all(np.isfinite(z)):
            self._log("SplineEKF update_from_thread: bad thread samples, skipping.")
            return {}

        # ── Innovation plausibility gate ──────────────────────────────────────
        # The thread physically cannot rigidly displace from the filter by tens
        # of units in one frame at rest — a median innovation beyond the
        # motion-scaled bound is a wrong-segment or teleport MEASUREMENT
        # (detangle fragment, bad-depth optim fit), and fusing it drags the
        # whole state off the thread (observed μ=21.9 and μ=172.9, both at
        # motion=0).  Judged on the direction-ALIGNED samples (a reversed
        # labelling is not a displacement — see the guard below) and checked
        # BEFORE the length-adapt, so a collapsed frame's length never enters
        # the adapt EMA either.  THREAD_INNOV_PERSIST_FRAMES consecutive
        # rejections mean the filter, not the measurement, is the wrong one —
        # then the update is accepted and snaps the filter back.
        # xb      = self.x.reshape(self.M, 3)
        # z_gate  = (z[::-1] if (np.linalg.norm(z[0] - xb[-1])
        #                        + np.linalg.norm(z[-1] - xb[0]))
        #                      < (np.linalg.norm(z[0] - xb[0])
        #                         + np.linalg.norm(z[-1] - xb[-1]))
        #            else z)
        # innov_med = float(np.median(np.linalg.norm(z_gate - xb, axis=1)))
        # innov_max = EP.THREAD_INNOV_MAX_BASE * (
        #     1.0 + self.gate_motion_gain * self._last_motion)
        # if innov_med > innov_max:
        #     streak = getattr(self, '_innov_reject_streak', 0) + 1
        #     self._innov_reject_streak = streak
        #     if streak < EP.THREAD_INNOV_PERSIST_FRAMES:
        #         self._log(f"SplineEKF update_from_thread: REJECTED — median "
        #                   f"innov {innov_med:.1f} > {innov_max:.1f} "
        #                   f"(motion={self._last_motion:.2f}); implausible "
        #                   f"thread measurement, skipping update & length-adapt "
        #                   f"({streak}/{EP.THREAD_INNOV_PERSIST_FRAMES}).")
        #         return {}
        #     self._log(f"SplineEKF update_from_thread: median innov "
        #               f"{innov_med:.1f} > {innov_max:.1f} PERSISTED "
        #               f"{streak} frames — accepting (filter is the stale one).")
        self._innov_reject_streak = 0

        # Length-adaptive knot count: the measured thread's polyline length is
        # the OBSERVED length (shrinks under occlusion / a pulled-away thread,
        # unlike the ratcheting _len_ref).  Resample the state if it moved
        # enough; then re-sample the measurement on the new knots.
        L_obs = float(np.sum(np.linalg.norm(np.diff(z, axis=0), axis=1)))
        if self.maybe_adapt_ctrl_count(L_obs):
            z = np.asarray(thread(self.t_ctrl), dtype=float)
            conf = _conf_at_ctrl()          # t_ctrl changed with the resample
            if z.shape != (self.M, 3) or not np.all(np.isfinite(z)):
                self._log("SplineEKF update_from_thread: bad re-samples, skipping.")
                return {}
        ends   = self.x.reshape(self.M, 3)[[0, -1]]
        d_same = (np.linalg.norm(z[0]  - ends[0])
                  + np.linalg.norm(z[-1] - ends[1]))
        d_flip = (np.linalg.norm(z[0]  - ends[1])
                  + np.linalg.norm(z[-1] - ends[0]))
        if d_flip < d_same:
            z = z[::-1]
            if conf is not None:
                conf = conf[::-1]           # keep conf aligned with the samples
            self._log(f"SplineEKF update_from_thread: thread runs OPPOSITE the "
                      f"filter (d_same={d_same:.1f} > d_flip={d_flip:.1f}); "
                      "reversed the samples to match the filter's labelling.")
        return self.update(self.t_ctrl, z, conf=conf)

    def maybe_adapt_ctrl_count(self, L_obs: float) -> bool:
        """Length-adaptive control-point count (see the user-facing rationale in
        __init__).  EMA-smooth the observed thread length, derive the knot count
        that keeps the seed knot SPACING, and resample the state when it differs
        from the current count by at least _ADAPT_HYST knots.  Bounded to
        [_M_MIN, _M_init].  Returns True if the state was resampled."""
        if (self._ctrl_spacing is None or self.x is None
                or not np.isfinite(L_obs) or L_obs <= 0):
            return False
        a = self._ADAPT_EMA_ALPHA
        self._L_obs_ema = (L_obs if self._L_obs_ema is None
                           else a * L_obs + (1.0 - a) * self._L_obs_ema)
        # Persistence streak on the RAW frame (not the EMA): the EMA lags, so
        # after a transient it keeps wanting the wrong count for several
        # recovery frames — the raw value snaps back immediately and resets
        # the streak, so only a SUSTAINED length change can adapt.
        M_raw = int(np.clip(int(round(L_obs / self._ctrl_spacing)) + 1,
                            self._M_MIN, self._M_init))
        if abs(M_raw - self.M) >= self._ADAPT_HYST:
            self._adapt_streak += 1
        else:
            self._adapt_streak = 0
        M_want = int(round(self._L_obs_ema / self._ctrl_spacing)) + 1
        M_want = int(np.clip(M_want, self._M_MIN, self._M_init))
        if abs(M_want - self.M) < self._ADAPT_HYST:
            return False
        if self._adapt_streak < self._ADAPT_STREAK_FRAMES:
            self._log(f"SplineEKF: ctrl-count change {self.M}→{M_want} "
                      f"PENDING ({self._adapt_streak}/"
                      f"{self._ADAPT_STREAK_FRAMES} consecutive frames)")
            return False
        self._adapt_streak = 0
        self._log(f"SplineEKF: ADAPTING control-point count {self.M}→{M_want} "
                  f"(observed length {self._L_obs_ema:.1f}, seed spacing "
                  f"{self._ctrl_spacing:.1f}) — resampling state, no reseed.")
        self._resample_state(M_want)
        return True

    def _resample_state(self, M_new: int) -> None:
        """Resample the filter to M_new control points along the SAME curve.

        State: new control points are the current spline evaluated at the new
        knots (exact — cardinal-basis matrix B, (M_new, M_old)).  Covariance is
        pushed through the same linear map, P_new = (B⊗I3) P (B⊗I3)ᵀ, plus a
        small diagonal floor (an up-sampling B is rank-deficient — M_new >
        M_old columns span only the old space — and update() inverts P).  All
        M-dependent structures (cardinal basis, smoothness/end penalties, the
        length-prior edge profile, the grasp-lock anchor index) are rebuilt or
        remapped.  Temporal state (len_ref tracking, motion, orientation
        anchors) carries over — this is a REPARAMETERIZATION, not a reseed."""
        M_old = self.M
        t_new = np.linspace(0.0, 1.0, M_new)
        B = np.column_stack([c(t_new) for c in self._cardinal])   # (M_new, M_old)
        self.x = (B @ self.x.reshape(M_old, 3)).flatten()
        K = np.kron(B, np.eye(3))                                 # (3M_new, 3M_old)
        P = K @ self.P @ K.T
        self.P = 0.5 * (P + P.T) + np.eye(3 * M_new) * 1e-2
        self.M      = M_new
        self.t_ctrl = t_new
        self._cardinal = self._build_cardinal()
        self._build_penalties()
        # Length prior: keep the tracked total length AND the reference
        # length DISTRIBUTION.  The per-edge profile is re-binned from the
        # OLD reference itself (cumulative profile interpolated at the new
        # uniform knots) — NEVER re-derived from the current polygon:
        # adaptation fires exactly on frames where the observed length just
        # changed (occlusion, a transient collapse), so the polygon at this
        # moment is the least trustworthy shape there is, and re-deriving
        # from it enshrined a one-frame collapse as the permanent length
        # reference (the frame-6 state damage).
        if self._edge_profile is not None and len(self._edge_profile):
            c_old = np.concatenate([[0.0], np.cumsum(self._edge_profile)])
            t_old = np.linspace(0.0, 1.0, len(self._edge_profile) + 1)
            prof  = np.diff(np.interp(np.linspace(0.0, 1.0, M_new),
                                      t_old, c_old))
            self._edge_profile = prof / (prof.sum() + 1e-9)
            self._edge_ref     = self._edge_profile * self._len_ref
        # Grasp lock: same physical anchor point, new index scale.
        if self._grip_idx is not None:
            self._grip_idx = int(round(self._grip_idx * (M_new - 1)
                                       / max(M_old - 1, 1)))

    def mahalanobis_gate(self,
                         kpts_3d: np.ndarray,
                         nn_idxs: np.ndarray,
                         t_dense: np.ndarray,
                         chi2_thresh: float = None,
                         update_monitor: bool = False) -> np.ndarray:
        """
        Compute per-keypoint Mahalanobis distances and return a gate mask.
        This is warm_ordering's match gate (3-D camera frame): the innovation
        covariance H P Hᵀ + R is anisotropic — tight laterally, loose in z per
        _R_block — so noisy stereo depth is tolerated in proportion to
        sigma_meas_z rather than judged like a lateral error.

        kpts_3d  : (N, 3) unprojected 3-D positions of current keypoints
        nn_idxs  : (N,)   index into t_dense for each keypoint's NN match
                           (nn_idxs from kd_warm.query)
        t_dense  : (D,)   the dense parameter grid used during NN search
        chi2_thresh    : optional χ²(3) threshold override; defaults to the
                          filter-level self.chi2_thresh.  Both get the same
                          motion-adaptive loosening.
        update_monitor : feeds the divergence monitor read by
                          maybe_recover_from_divergence().  warm_ordering now
                          reports its FINAL acceptance via note_match_frac()
                          instead (the χ² gate alone reads ~1.0 when P is
                          inflated), so leave this False unless the caller has
                          no post-gate filtering of its own.

        Returns
        -------
        gate_mask : (N,) bool — True = keypoint is within the χ² gate.
        maha_dists: (N,) float — squared Mahalanobis distances (for debug).
        """
        N = len(kpts_3d)
        maha2  = np.full(N, np.inf)

        # Vectorised over all N keypoints: the per-keypoint loop spent most of
        # its time in _eval_basis (M scalar scipy calls PER keypoint — thousands
        # per frame across the gate calls).  Here each cardinal spline is
        # evaluated over all N t's at once (M vectorised calls), and the
        # Kronecker structure H_i = kron(b_i, I3) collapses the per-keypoint
        # (3,3M)@(3M,3M)@(3M,3) products into one einsum over P as (M,3,M,3):
        #     Sig_i[b,d] = Σ_ac B[i,a] B[i,c] P4[a,b,c,d]  (+ R)
        # Innovation covariance MUST include measurement noise R, otherwise
        # the gate treats keypoints as noise-free and over-rejects (this is
        # what made depth-noisy keypoints fail the χ² test).
        if N:
            kpts  = np.asarray(kpts_3d, dtype=float)
            t_sel = np.asarray(t_dense, dtype=float)[np.asarray(nn_idxs, int)]
            M     = self.M
            B     = np.column_stack([c(t_sel) for c in self._cardinal])  # (N, M)
            mu    = B @ self.x.reshape(M, 3)                             # (N, 3)
            P4    = self.P.reshape(M, 3, M, 3)
            Sig   = (np.einsum('na,abcd,nc->nbd', B, P4, B, optimize=True)
                     + self._R_block)                                    # (N,3,3)
            diff  = kpts - mu
            try:
                sol   = np.linalg.solve(Sig, diff[:, :, None])[:, :, 0]
                maha2 = np.einsum('ni,ni->n', diff, sol)
                # NaN inputs (depthless placeholder rows) → inf → rejected,
                # matching the old per-row semantics.
                maha2 = np.where(np.isfinite(maha2), maha2, np.inf)
            except np.linalg.LinAlgError:
                # Singular Sig should be impossible (R is PD); keep the old
                # per-row path as the safety net.
                for i in range(N):
                    Sig_i = Sig[i]
                    d_i   = diff[i]
                    try:
                        maha2[i] = float(d_i @ np.linalg.solve(Sig_i, d_i))
                    except np.linalg.LinAlgError:
                        maha2[i] = np.inf

        # Motion-adaptive: loosen the gate while the thread is being moved so a
        # fast-moved keypoint (large innovation) is not rejected.
        base = self.chi2_thresh if chi2_thresh is None else float(chi2_thresh)
        thr  = base * (1.0 + self.gate_motion_gain * self._last_motion)
        gate_mask = maha2 < thr
        if update_monitor:
            # Divergence monitor (formerly fed by the 2-D gate): a sustained
            # low pass-rate means the state disagrees with the keypoints —
            # see maybe_recover_from_divergence().
            self._last_gate_frac = float(gate_mask.mean()) if N else 1.0
            if self._last_gate_frac < self.gate_recover_frac:
                self._low_gate_streak += 1
            else:
                self._low_gate_streak = 0
        finite = maha2[np.isfinite(maha2)]
        mean_str = f"{finite.mean():.1f}" if finite.size else "n/a"
        self._log(f"SplineEKF gate: {gate_mask.sum()}/{N} keypoints pass "
              f"(χ²<{thr:.2f}, motion={self._last_motion:.2f})  "
              f"maha² μ={mean_str}")
        return gate_mask, maha2

    def mahalanobis_gate_2d(self,
                            kpts_px: np.ndarray,
                            nn_idxs: np.ndarray,
                            t_dense: np.ndarray,
                            P1: np.ndarray,
                            chi2_thresh_2d: float = 5.99) -> np.ndarray:
        """
        Image-frame χ² gate: like mahalanobis_gate, but the innovation lives in
        PIXELS.  The predicted spline point and its covariance (state + 3-D
        measurement noise) are pushed through the pinhole projection Jacobian,
        so a keypoint with noisy stereo depth is judged only on where it sits
        in the image — depth noise cannot fail the gate.

        kpts_px  : (N, 2) keypoint (row, col) pixel positions
        nn_idxs  : (N,)   index into t_dense of each keypoint's matched t
        t_dense  : (D,)   dense parameter grid used during matching
        P1       : (3, 4) rectified left-camera projection matrix
        chi2_thresh_2d : gate threshold — χ²(2) units (5.99 = 95%), NOT the
                          3-DOF self.chi2_thresh.

        Returns (gate_mask (N,) bool, maha2 (N,) float).
        """
        fx, fy = P1[0, 0], P1[1, 1]
        cx, cy = P1[0, 2], P1[1, 2]
        N = len(kpts_px)
        maha2 = np.full(N, np.inf)

        for i in range(N):
            t_i   = float(t_dense[nn_idxs[i]])
            H_i   = self._eval_H_row(t_i)                    # (3, 3M)
            Sig3  = H_i @ self.P @ H_i.T + self._R_block     # (3, 3)
            mu    = H_i @ self.x                             # (X, Y, Z)
            X, Y, Z = mu
            if Z <= 1e-6:
                continue                                     # behind camera → reject
            # projected prediction, (row, col) to match kpts_px
            pred = np.array([fy * Y / Z + cy, fx * X / Z + cx])
            # Jacobian of (row, col) wrt (X, Y, Z)
            J = np.array([[0.0,     fy / Z, -fy * Y / Z**2],
                          [fx / Z,  0.0,    -fx * X / Z**2]])
            Sig2 = J @ Sig3 @ J.T                            # (2, 2)
            diff = np.asarray(kpts_px[i], dtype=float) - pred
            try:
                maha2[i] = float(diff @ np.linalg.solve(Sig2, diff))
            except np.linalg.LinAlgError:
                maha2[i] = np.inf

        # Motion-adaptive: loosen the image-frame gate during motion too.
        thr2 = chi2_thresh_2d * (1.0 + self.gate_motion_gain * self._last_motion)
        gate_mask = maha2 < thr2
        finite = maha2[np.isfinite(maha2)]
        mean_str = f"{finite.mean():.1f}" if finite.size else "n/a"
        # Feed the divergence monitor: a low pass-rate means the filter's state
        # disagrees with the current keypoints (see maybe_recover_from_divergence).
        self._last_gate_frac = float(gate_mask.mean()) if N else 1.0
        if self._last_gate_frac < self.gate_recover_frac:
            self._low_gate_streak += 1
        else:
            self._low_gate_streak = 0
        self._log(f"SplineEKF 2D gate: {gate_mask.sum()}/{N} keypoints pass "
              f"(χ²(2)<{thr2:.2f}, motion={self._last_motion:.2f})  "
              f"maha² μ={mean_str}")
        return gate_mask, maha2

    def note_match_frac(self, frac: float) -> None:
        """Feed the divergence monitor the FINAL per-frame match acceptance —
        χ² gate AND the absolute lateral/depth caps and validity strikes.

        The χ² gate alone is the wrong signal exactly when it matters: after a
        run of unaccepted frames P inflates (predict-only growth), the gate
        passes every keypoint (the state is UNCERTAIN, not right) and the
        monitor read gate_frac=1.0 while the spline sat 160px off the data.
        The cap rejections are the real state-vs-data disagreement.  Call once
        per frame after the final matched mask is known; replaces this frame's
        gate-frac bookkeeping."""
        self._last_gate_frac = float(frac)
        if self._last_gate_frac < self.gate_recover_frac:
            self._low_gate_streak += 1
        else:
            self._low_gate_streak = 0

    def maybe_recover_from_divergence(self) -> bool:
        """Unlock a diverged/locked-out filter.  If the image-frame gate has
        rejected most keypoints for gate_recover_frames consecutive frames AND
        the tool is nearly still (motion < motion_recover_max), the state is
        confidently wrong and the shrunken P is rejecting the good data — INFLATE
        P (leaving x) so the next update re-acquires it.

        Inflation, NOT an isotropic reset: P is scaled by a single factor up to
        recover_P0_scale on its largest diagonal (never shrinking it), exactly
        like degrade().  A uniform scalar multiply keeps P's CORRELATION
        structure intact — only the eigenvalues grow, the eigenvectors don't
        move — so the state's direction anchoring survives.  Resetting to
        recover_P0_scale·I instead wiped those correlations (P⁻¹→0, isotropic),
        letting the next update refit the keypoints in reverse and FLIP the
        thread.  Returns True if recovery fired.  Call once per frame after
        predict, before the next gate/update.

        Every call records the monitor state it evaluated in
        self._last_recovery_info (the gate pass-rate, low-gate streak and motion
        it saw, the thresholds each is tested against, whether it fired and by
        what factor) so a debug view can show WHEN recovery fires and the VALUES
        that triggered it.  Captured here (predict-time) because this frame's
        gate has not run yet, so _last_gate_frac / _low_gate_streak still hold
        the accumulated state the decision is actually made on; the upcoming
        update overwrites them."""
        info = {"fired": False,
                "gate_frac":   float(self._last_gate_frac),
                "frac_thresh": self.gate_recover_frac,
                "streak":      int(self._low_gate_streak),
                "frames_thresh": self.gate_recover_frames,
                "motion":      float(self._last_motion),
                "motion_max":  self.motion_recover_max,
                "factor":      1.0}
        self._last_recovery_info = info
        if (self.gate_recover_frac is None or self.gate_recover_frac <= 0
                or self.x is None):
            return False
        if (self._low_gate_streak >= self.gate_recover_frames
                and self._last_motion < self.motion_recover_max):
            dmax   = float(np.diag(self.P).max())
            factor = max(1.0, self.recover_P0_scale / max(dmax, 1e-9))
            self.P = self.P * factor
            info["fired"], info["factor"] = True, float(factor)
            self._log(f"SplineEKF: DIVERGENCE RECOVERY — gate pass-rate "
                  f"{self._last_gate_frac:.2f} for {self._low_gate_streak} "
                  f"frames at low motion ({self._last_motion:.2f}); P inflated "
                  f"×{factor:.2f} (max_diag→{float(np.diag(self.P).max()):.1f}) "
                  f"to unlock the filter.")
            self._low_gate_streak = 0
            return True
        return False

    def lock_orientation_to_grasp(self, grasp_pos_3d, tool_R=None, tool_id=None,
                                  grasp_max_dist: float = EP.GRASP_LOCK_MAX_DIST) -> bool:
        """Hold the spline's t-direction fixed using the near-gripper TANGENT,
        anchored to the TOOL (not tracked frame-to-frame), HANDOFF-aware.

        The gripper grasps the thread rigidly, so the near-grasp tangent (the
        spline's direction of increasing t there) is fixed in the GRIPPER frame:
        it moves ONLY as the tool rotates, never with the thread reshaping.  So
        the reference is stored in the tool frame (self._grip_tangent_tool) and
        rotated into the current camera frame each call by the current tool
        orientation `tool_R` (curr_T[:3,:3]).  The current grasp tangent is
        compared to that.

        Why tool-anchored, not frame-to-frame tracked: an earlier version stored
        the reference in the camera frame and re-set it to the current tangent
        every accepted frame.  A GRADUAL, motion-driven reversal (the thread
        reshaping through a straight/ambiguous pose over several frames) then
        kept dot>0 at every single step and the reference simply tracked the flip
        all the way around — never firing.  Tool-anchoring removes that loophole:
        the reference can't follow a thread reshape, so a creeping reversal
        accumulates against a fixed physical anchor until dot<0 and fires.

        BIMANUAL HANDOFF: `tool_id` identifies the active grasp tool (e.g. the
        PSM number).  When it changes — a bimanual transfer, gripper A releases
        and gripper B takes over at a different thread location/orientation — the
        old tool-frame reference is INVALID, so we re-anchor to the new tool,
        carrying the current orientation forward as the new baseline.  Also, while
        the grasp is farther than `grasp_max_dist` from the thread (the tracked
        gripper has released and moved off mid-handoff), the near-grasp tangent is
        meaningless, so the lock stays idle rather than acting on garbage.

        SAME-POINT GUARD (frame-17 spurious-fire fix): the tangent comparison is
        only physically meaningful at the SAME thread point the reference was
        anchored at — a rigidly held grasp point cannot change its control-point
        index.  During an APPROACH (gripper still sliding toward/along the
        thread) the nearest control point wanders, and on a curved thread the
        tangents at two different points can legitimately OPPOSE (dot<0), which
        made the lock reverse a CORRECT state the moment a fresh grasp landed at
        a different point than the mid-approach anchor.  So: if the nearest
        control-point index moved more than a couple of knots since anchoring,
        the reference is for a different point — RE-ANCHOR silently instead of
        comparing.  Only a stable grasp point ever fires the reversal.

        First (re-)anchor stores the tool-frame tangent + its control-point
        index.  Later calls at the SAME point: if the grasp tangent points
        opposite the tool-anchored reference (dot<0, the reversed labelling
        explains it better), reverse the whole state (identical curve, only
        t=0↔t=1 labelling swaps) and re-anchor to the corrected direction.
        Returns True if it re-aligned.  No-op when no grasp is given or before
        initialize().  Call once per frame AFTER the update with the current
        grasp (curr_T[:3,3]), tool orientation (curr_T[:3,:3]) and tool id (the
        active PSM)."""
        if self.x is None or grasp_pos_3d is None:
            return False
        grasp = np.asarray(grasp_pos_3d, dtype=float)
        # Stay idle while the grasp is far from the thread (mid-handoff: the
        # tracked gripper has released and moved off) — the tangent is garbage.
        g_dist = float(np.linalg.norm(self.x.reshape(self.M, 3) - grasp,
                                      axis=1).min())
        if grasp_max_dist is not None and g_dist > grasp_max_dist:
            self._log(f"SplineEKF grasp orient: grasp {g_dist:.1f} from thread "
                      f"(> {grasp_max_dist:.0f}); lock idle (likely mid-handoff).")
            return False
        # Bimanual handoff: the active grasp tool changed → old reference invalid.
        if tool_id is not None and tool_id != self._grip_tool_id:
            self._grip_tool_id      = tool_id
            self._grip_tangent_tool = None            # force re-anchor below
            self._log(f"SplineEKF: grasp tool changed to {tool_id}; re-anchoring "
                      "orientation reference to the new tool.")
        tang, g_idx = self._grasp_tangent(grasp)      # camera frame + ctrl index
        if tang is None:
            return False
        R = np.asarray(tool_R, dtype=float) if tool_R is not None else np.eye(3)
        # Same-point guard: grasp point moved along the thread since anchoring
        # (approach/slide/regrasp) → reference is for a DIFFERENT point, tangent
        # comparison invalid → re-anchor at the new point instead of firing.
        # EXCEPTION: a jump to the MIRRORED index (j → M-1-j) is exactly what a
        # parameterization flip does to a rigidly held point — that case must
        # still be compared (dot<0 there confirms and corrects the flip).
        if self._grip_tangent_tool is not None and self._grip_idx is not None:
            same     = abs(g_idx - self._grip_idx) <= self._GRIP_IDX_TOL
            mirrored = abs(g_idx - (self.M - 1 - self._grip_idx)) <= self._GRIP_IDX_TOL
            if not (same or mirrored):
                self._log(f"SplineEKF grasp orient: grasp point moved along thread "
                          f"(ctrl {self._grip_idx}→{g_idx}); re-anchoring, no compare.")
                self._grip_tangent_tool = None
        if self._grip_tangent_tool is None:
            self._grip_tangent_tool = R.T @ tang      # store in tool frame
            self._grip_idx          = g_idx
            self._log(f"SplineEKF: orientation anchored to grasp tangent "
                      f"{np.round(tang, 2)} at ctrl {g_idx} (tool-frame "
                      f"{np.round(self._grip_tangent_tool, 2)}).")
            return False
        ref = R @ self._grip_tangent_tool             # expected in camera frame
        dot = float(np.dot(tang, ref))
        # Always log the alignment so a gradual reversal is visible: dot creeping
        # 1 → 0 → -1 over several frames is a flip the OLD tracked reference hid.
        self._log(f"SplineEKF grasp orient: dot(tang,ref)={dot:+.2f}  "
                  f"ctrl={g_idx}  tang={np.round(tang, 2)}  ref={np.round(ref, 2)}")
        if dot < 0.0:
            self._reverse_state()
            # Reversal flips the grasp tangent's sign and mirrors the control
            # points, so re-anchor to the corrected direction and the mirrored
            # index (ctrl j → M-1-j).
            self._grip_tangent_tool = R.T @ (-tang)
            self._grip_idx          = self.M - 1 - g_idx
            self._last_orient_info["corrected"] = True
            self._log("SplineEKF: GRASP ORIENTATION LOCK — grasp tangent opposed "
                      "the tool-anchored reference; flipped the spline back.")
            return True
        return False

    def _grasp_tangent(self, grasp_pos_3d, half_window: int = 2):
        """Unit tangent (direction of increasing t) of the spline at the control
        point nearest the grasp, plus that control point's index.  Uses a
        ±half_window control-point baseline so the direction is stable; the
        rigid near-grasp segment makes it reliable.  Returns (None, idx) if the
        state is degenerate (coincident control points)."""
        pts = self.x.reshape(self.M, 3)
        g   = int(np.argmin(np.linalg.norm(
            pts - np.asarray(grasp_pos_3d, dtype=float), axis=1)))
        lo  = max(0, g - half_window)
        hi  = min(self.M - 1, g + half_window)
        if hi == lo:
            return None, g
        v = pts[hi] - pts[lo]
        n = float(np.linalg.norm(v))
        return (v / n if n > 1e-9 else None), g

    def _reverse_state(self) -> None:
        """Reverse the parameterization end-for-end: control points, covariance
        and the length-prior edge arrays all flip.  new(t) = old(1-t) — the
        curve is unchanged, only the t=0↔t=1 sense swaps."""
        M = self.M
        order = np.arange(M)[::-1]
        idx   = np.repeat(order * 3, 3) + np.tile([0, 1, 2], M)  # coord permute
        self.x = self.x[idx].copy()
        self.P = self.P[np.ix_(idx, idx)].copy()
        if self._edge_ref is not None:
            self._edge_ref = self._edge_ref[::-1].copy()
        if self._edge_profile is not None:
            self._edge_profile = self._edge_profile[::-1].copy()

    # ══════════════════════════════════════════════════════════════════════
    #  Private helpers
    # ══════════════════════════════════════════════════════════════════════

    def _build_penalties(self) -> None:
        """Build the M-dependent shape-prior matrices for the CURRENT self.M.
        Called from __init__ and again on every control-point resample.

        _S_pen — smoothness pseudo-measurement D3·x ≈ 0: third differences of
        consecutive control points per coordinate, stencil [-1, 3, -3, 1] over
        4 neighbouring points → ((M-3)·3, 3M); _S_pen = D3ᵀD3/σ² enters the
        information-form update.

        _E_pen — boundary curvature penalty D2_end·x ≈ 0: 2nd-difference
        stencil [1, -2, 1] on the triples inside the first and last end_span
        control points (natural-spline ends, stops under-observed tips curling
        up / folding back)."""
        M = self.M
        # Knot-spacing-invariant prior weights.  A third difference of knots
        # spaced h apart is ≈ h³·x''', so with a FIXED σ the penalty on
        # PHYSICAL waviness scales as h⁵ ∝ 1/(M−1)⁵ — densifying the knots
        # from 16 to 30 silently made the thread ~25× floppier (the "wavy
        # with more control points" bug), and every adaptive knot-count
        # resample changed the stiffness again.  Rescale to the reference
        # count the σ values were tuned at (EP.SMOOTH_REF_M): identical
        # behaviour at M = REF, same physical stiffness at every other M.
        # The end 2nd-difference penalty scales as h⁴ per stencil (its
        # stencil count is FIXED at end_span−2 per end), hence exponent 4.
        s_scale = ((M - 1) / (EP.SMOOTH_REF_M - 1)) ** 5
        e_scale = ((M - 1) / (EP.SMOOTH_REF_M - 1)) ** 4
        if self.sigma_smooth is not None and self.sigma_smooth > 0 and M >= 4:
            D3 = np.zeros(((M - 3) * 3, 3 * M))
            for i in range(M - 3):
                for c in range(3):
                    row = 3 * i + c
                    D3[row, 3 * (i + 0) + c] = -1.0
                    D3[row, 3 * (i + 1) + c] =  3.0
                    D3[row, 3 * (i + 2) + c] = -3.0
                    D3[row, 3 * (i + 3) + c] =  1.0
            self._S_pen = (D3.T @ D3) * (s_scale / self.sigma_smooth ** 2)
        else:
            self._S_pen = None

        if (self.sigma_end_straight is not None and self.sigma_end_straight > 0
                and self.end_span >= 3 and M >= self.end_span):
            # triple START indices near each boundary (triple = s, s+1, s+2)
            n_tri  = self.end_span - 2
            starts = list(range(0, n_tri)) + list(range(M - self.end_span, M - 2))
            starts = sorted(set(starts))
            D2e = np.zeros((len(starts) * 3, 3 * M))
            for r, s in enumerate(starts):
                for c in range(3):
                    row = 3 * r + c
                    D2e[row, 3 * (s + 0) + c] =  1.0
                    D2e[row, 3 * (s + 1) + c] = -2.0
                    D2e[row, 3 * (s + 2) + c] =  1.0
            self._E_pen = (D2e.T @ D2e) * (e_scale / self.sigma_end_straight ** 2)
        else:
            self._E_pen = None

    def _build_cardinal(self) -> list:
        """
        Precompute M cardinal basis functions  c_j(t_ctrl[j]) = 1,
        c_j(t_ctrl[k]) = 0  for k ≠ j.

        Called once at __init__ — t_ctrl is fixed for the filter lifetime.
        """
        cardinal = []
        M = self.M
        for j in range(M):
            e_j = np.zeros(M)
            e_j[j] = 1.0
            cardinal.append(CubicSpline(self.t_ctrl, e_j))
        return cardinal

    def _eval_basis(self, t: float) -> np.ndarray:
        """
        (M,) cardinal basis values at scalar t ∈ [0,1].
        Partition-of-unity property: values sum to ~1.
        """
        return np.array([c(float(t)) for c in self._cardinal])

    def _eval_H_row(self, t: float) -> np.ndarray:
        """
        (3, 3M) measurement row for a single observation at parameter t.
        H_row @ x  =  spline(t)  (3-D position).

        Uses the Kronecker structure:
            H_row[:, 3j:3j+3] = B_j(t) * I_3
        """
        basis = self._eval_basis(t)                    # (M,)
        # np.kron(basis.reshape(1, M), I_3) gives (3, 3M)
        # with block (i, j):  basis[j] * I_3[i, :]  — exactly what we need.
        return np.kron(basis.reshape(1, self.M), np.eye(3))  # (3, 3M)

    def _build_H(self, t_values: np.ndarray) -> np.ndarray:
        """
        (3N, 3M) stacked measurement matrix for N observations.

        Vectorised: each cardinal spline is evaluated over ALL t at once
        (M vectorised calls) instead of N×M scalar calls, and the Kronecker
        block structure H[3i+k, 3j+k] = B[i, j] is assembled with einsum.
        """
        t_values = np.asarray(t_values, dtype=float)
        N, M = len(t_values), self.M
        B = np.column_stack([c(t_values) for c in self._cardinal])  # (N, M)
        return np.einsum('ij,kl->ikjl', B, np.eye(3)).reshape(3 * N, 3 * M)

    def _build_Q(self,
                 ctrl_pts: np.ndarray,
                 tool_pos: np.ndarray) -> np.ndarray:
        """
        Spatially-varying isotropic process noise.

        σ_i = σ_tip + (σ_base − σ_tip) · (1 - exp(−‖p_i − tool‖ / R))

        Thread points close to the tool tip receive low noise (they move 
        rigidly with the tool); points far from the tip receive high noise 
        (due to elastic deformation and lagging).
        """
        M = self.M
        Q = np.zeros((3*M, 3*M))
        for i in range(M):
            dist   = np.linalg.norm(ctrl_pts[i] - tool_pos)
            
            # alpha is 1.0 at the tool tip and decays to 0.0 far away.
            alpha  = np.exp(-dist / max(self.deform_radius, 1e-6))

            # Near the grasped tool the thread deforms most, so process noise is
            # HIGH there and LOW at the far/anchored end (rigid motion is a good
            # model there). This matches the __init__ contract:
            #   dist = 0   -> alpha = 1 -> sigma = sigma_proc_tip  (High noise)
            #   dist = inf -> alpha = 0 -> sigma = sigma_proc_base (Low noise)
            sigma  = self.sigma_proc_base + (self.sigma_proc_tip - self.sigma_proc_base) * alpha
            
            Q[3*i:3*i+3, 3*i:3*i+3] = np.eye(3) * sigma ** 2
            
        return Q
