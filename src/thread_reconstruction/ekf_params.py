"""
ekf_params.py
─────────────
Every tuning knob for the SplineEKF thread tracker, in one place.

These constants were previously scattered across five files (spline_ekf.py
constructor defaults, the Order() constructor args in keypt_ordering.py,
locals inside warm_ordering, and mirrors in ros_thread_reconstruct.py /
warm_start.py / optim.py).  They are consolidated here so a tuning session
touches ONE file; the consumers import this module, so an edit here is live
on the next run everywhere at once.

Merged duplicates (same function, previously separate copies):
  • DEFORM_RADIUS — was three "keep in sync" copies: SplineEKF(deform_radius),
    WarmStart.deform_radius, optim.TEMPORAL_DEFORM_RADIUS.  They silently
    disagreed at 50/30/30 until 2026-07-26.
  • CHI2_MATCH_3D — was warm_ordering's CHI2_GATE_3D and Order._CROSSING_CHI2
    (both χ²(3) 95% = 7.81, documented as "the same threshold").
  • AMB_P_GAIN / AMB_P_MAXDIAG — were both SplineEKF.degrade()'s defaults and
    Order._AMB_P_GAIN/_AMB_P_MAXDIAG passed explicitly at the call site.
  • EKF_OUTPUT_MODE — was the USE_EKF_THREAD + KF_OPTIM_LOOP boolean pair in
    ros_thread_reconstruct (KF_OPTIM_LOOP overrode USE_EKF_THREAD anyway).

Where the long-form rationale lives: the mechanism comments stay next to the
code that implements them (spline_ekf.py docstrings, keypt_ordering.py
comment blocks).  This file keeps each knob's value plus a short "which way
to turn it" note.
"""

# ══════════════════════════════════════════════════════════════════════════
#  Output / coupling mode  (ros_thread_reconstruct.py)
# ══════════════════════════════════════════════════════════════════════════
# What gets published, and how the KF couples to optim:
#   'kf_optim_loop' — Option A predict→fit→correct loop: KF predicts (no
#                     keypoint update), optim fits the raw keypoints with its
#                     robust boxes using the KF prediction as temporal prior,
#                     KF is corrected from OPTIM'S thread.  optim's thread is
#                     published.  (Replaces KF_OPTIM_LOOP = True.)
#   'ekf_thread'    — the EKF posterior spline replaces optim's fit as the
#                     published/validated/warm-started output; the filter is
#                     updated from the raw matched keypoints.
#                     (Replaces USE_EKF_THREAD = True, KF_OPTIM_LOOP = False.)
#   'optim'         — optim's thread is published; EKF updated from raw
#                     matched keypoints (both experiment flags off).
EKF_OUTPUT_MODE = 'ekf_thread' # 'ekf_thread' kf_optim_loop

# Weight of the KF-spline shape prior in optim's QP under 'kf_optim_loop'
# (dimensionless, like optim.TEMPORAL_LAMBDA).  Larger → optim hews closer to
# the KF's denoised estimate; 0 → optim ignores the KF.
KF_PRIOR_LAMBDA = 0.9

# DATA-TRUST scaling of that prior: beyond PRIOR_TRUST_NN_REF px of median
# predicted-spline→keypoint disagreement the prior is scaled by REF/nn_median
# (healthy tracking ~2-3 px), never below PRIOR_TRUST_MIN.  Raise NN_REF to
# tolerate more disagreement before distrusting the prior; lower TRUST_MIN to
# let the data take over more completely.
PRIOR_TRUST_NN_REF = 4.0    # px: disagreement considered "healthy"
PRIOR_TRUST_MIN    = 0.05   # never scale the prior below 5%

# ══════════════════════════════════════════════════════════════════════════
#  Filter core  (SplineEKF constructor)
# ══════════════════════════════════════════════════════════════════════════
N_CTRL     = 16     # control points (state dim = 3·N_CTRL); also the ceiling
                    # for the length-adaptive count below
P0_SCALE   = 49.0    # initial variance per coordinate at initialize();
                    # 49 ≈ ±7 units — raise if the first warm thread is noisy
SIGMA_MEAS   = 1.0  # LATERAL (x,y) measurement noise std, spline units
SIGMA_MEAS_Z = 3.0  # DEPTH (z) noise std — stereo depth is noisier; keeping
                    # it loose stops z noise over-rejecting at the gate

# Per-observation confidence weighting of R (SplineEKF.update `conf` arg):
# each observation's noise is inflated as R_i = R / max(conf_i, FLOOR), with
# conf_i the keypoint's stereo confidence from keypt_selection (full_conf).
# Ambiguous-stereo keypoints (epipolar-parallel/horizontal sections, conf→0)
# then barely tug the filter, which coasts on its motion model there.  The
# floor caps the inflation (0.05 → ≤20× variance) so a zero-confidence point
# still contributes a little and R stays finite.  Raise the floor to let
# low-confidence points pull harder; 1.0 disables the weighting entirely.
EKF_CONF_R_FLOOR = 0.2 # raise to set optim''s fitted depth pull harder

SIGMA_PROC_BASE = 6.0   # process noise std far from the tool (rigid motion
                        # is a good model there)
SIGMA_PROC_TIP  = 12.0  # process noise std at the tool tip (elastic
                        # deformation largest here)

# e-folding distance (spline units ≈ mm) of tool-induced deformation: the
# warp weight is w = exp(-d/DEFORM_RADIUS).  THE single shared copy — used by
# SplineEKF.predict, WarmStart.refresh_warm_start and optim's temporal-prior
# motion compensation, which must warp identically or the prediction and the
# prior pull in different directions.  Was 30: with the whole thread inside
# ~40mm of the grasp that gave w=0.27 at the far end, so a fast drag
# under-applied the warp (~0.2x) AND stretched the spline — bend radius
# collapsed, scramble saturated the ambiguity floor, every match demoted.
DEFORM_RADIUS = 50.0

# ══════════════════════════════════════════════════════════════════════════
#  χ² gates
# ══════════════════════════════════════════════════════════════════════════
# χ²(3): 6.25 → 90%, 7.81 → 95%, 11.34 → 99%.
CHI2_UPDATE_3D = 6.25  # filter-level gate (SplineEKF.chi2_thresh): screens
                       # the matched observations fed to ekf.update()
                       # (_make_ekf_obs)
CHI2_MATCH_3D  = 11.34  # warm_ordering's NN match gate AND the crossing
                       # re-insertion gate (_crossing_t_by_axis) — one
                       # threshold, merged
GATE_MOTION_GAIN = 2.0 # every χ² threshold ×(1 + gain·motion) so fast-moved
                       # keypoints pass during manipulation; 0 = off

# Absolute lateral caps on the NN match (px, converted to spline units at the
# working depth).  The χ² gate alone accepts far points when P is inflated,
# so these cap distance regardless; motion-scaled between base and hard.
NN_MAX_DIST_BASE_PX = 20.0    # lateral cap when the tool is still
NN_MAX_DIST_HARD_PX = 80.0   # lateral sanity ceiling, never exceeded

# ══════════════════════════════════════════════════════════════════════════
#  Motion adaptivity
# ══════════════════════════════════════════════════════════════════════════
MOTION_TRANS_REF = 1.5   # tool translation counting as "full" motion
MOTION_ROT_REF   = 0.10  # tool rotation (rad) counting as "full" motion
Q_MOTION_FLOOR   = 0.1  # fraction of Q kept when the tool is still — the
                         # temporal-memory knob; smaller = calmer at rest
MOTION_DECAY     = 0.6   # asymmetric hold: motion rises instantly, decays by
                         # this factor per still frame (~5-frame ease-out);
                         # 0 = off (cliff)
MOTION_PRIOR_FLOOR = 0.2 # shape priors relax to this fraction at full motion
                         # (track deformation while moving); 1.0 = constant

# ══════════════════════════════════════════════════════════════════════════
#  Shape priors  (pseudo-measurements folded into every update)
# ══════════════════════════════════════════════════════════════════════════
SIGMA_SMOOTH       = 5.0   # D3·x ≈ 0 bending prior; smaller = stiffer,
                           # larger = wigglier, None/≤0 = off
SMOOTH_REF_M       = 16    # knot count SIGMA_SMOOTH / SIGMA_END_STRAIGHT were
                           # tuned at.  A third difference of knots spaced h
                           # apart is ≈ h³·x''', so a FIXED σ makes physical
                           # stiffness collapse as h⁵ when knots densify — at
                           # N_CTRL=30 the thread was ~25× floppier than at 16
                           # (the "wavy with more control points" bug).  The
                           # penalties are rescaled by ((M−1)/(REF−1))^5 (D3)
                           # and ^4 (end D2) in _build_penalties, so stiffness
                           # is knot-count-invariant: identical to before at
                           # M=16, ~27× stiffer at M=30.  Retune SIGMA_SMOOTH
                           # only at this reference count.
SIGMA_STRETCH      = 0.99  # arc-length prior toward the tracked reference
                           # length; smaller = stiffer, None/≤0 = off
LEN_TRACK_ALPHA    = 0.15  # EMA rate of the tracked reference length (running
                           # max ratchet — grows to true length, never shrinks
                           # on occlusion); smaller = slower, more robust
SIGMA_END_STRAIGHT = 8.0   # natural-spline end condition — stops the
                           # under-observed tips curling/folding; smaller =
                           # straighter ends, None/≤0 = off
END_SPAN           = 3     # control points at each end it straightens (≥3)

# ══════════════════════════════════════════════════════════════════════════
#  Divergence / lock-out recovery
# ══════════════════════════════════════════════════════════════════════════
GATE_RECOVER_FRAC   = 0.15  # final match acceptance below this = "diverged";
                            # ≤0 disables recovery
GATE_RECOVER_FRAMES = 10    # consecutive diverged frames before firing
MOTION_RECOVER_MAX  = 0.03  # only fire when motion is below this (tool still)
RECOVER_P0_SCALE    = 50.0  # P inflated (never reset — correlations survive,
                            # thread can't flip) until its max diagonal
                            # reaches this.  NOTE: spline_ekf's docstring
                            # argues for ~400 ("too small and the gate still
                            # rejects the good data"); 50 is the value that
                            # has actually been running.

# ══════════════════════════════════════════════════════════════════════════
#  KF↔optim measurement plausibility  (SplineEKF.update_from_thread)
# ══════════════════════════════════════════════════════════════════════════
# The thread measured from optim cannot RIGIDLY displace from the filter state
# by more than this (median innovation norm, spline units ≈ mm) in one frame
# at rest; scaled ×(1 + GATE_MOTION_GAIN·motion) like the χ² gates.  Beyond it
# the measurement is a wrong-segment / teleport artefact (observed: a detangle
# fragment at innov μ=21.9, a bad-depth jump at μ=172.9, both at motion=0.00),
# so the update AND the length-adapt EMA are skipped for that frame.  Healthy
# innov μ runs 1-4.  Checked AFTER the direction guard (a reversed labelling
# is not a displacement).
THREAD_INNOV_MAX_BASE = 10.0
# Deadlock escape: if the gate rejects this many CONSECUTIVE frames, the
# "implausible" reading is evidently the new reality (the filter is the wrong
# one) — accept the measurement and let the update snap the filter back.
THREAD_INNOV_PERSIST_FRAMES = 5

# ══════════════════════════════════════════════════════════════════════════
#  Ambiguity / graceful degradation  (Order._ambiguity and consumers)
# ══════════════════════════════════════════════════════════════════════════
AMB_GATE_HI = 0.70   # gate pass-fraction ≥ this → no ambiguity
AMB_GATE_LO = 0.25   # ≤ this → fully ambiguous
AMB_MAHA_LO = 0.30   # median(maha²)/χ² ≤ this → no ambiguity
AMB_MAHA_HI = 0.90   # ≥ this → fully ambiguous
AMB_WIDEN_GAIN = 1.5 # unmatched-segment join thresholds ×(1 + gain·amb)
AMB_P_GAIN     = 8.0   # covariance inflated ×(1 + gain·amb) pre-update
                       # (single source for SplineEKF.degrade — merged)
AMB_P_MAXDIAG  = 400.0 # cap on that inflation's max diagonal
AMB_FLOOR_MAX  = 0.75  # ceiling on the warm-scramble FLOOR alone — a
                       # heuristic may discount the evidence, never silence it
AMB_KEEP_FLOOR = 0.20  # keep_bound never shrinks below this fraction of the
                       # χ² gate, so full ambiguity keeps the most confident
                       # matches instead of demoting all by construction

# Coverage-aware acceptance normalisation (detangle-clipped splines must not
# read as diverged):
COV_BAND_MULT    = 2.0
COV_SUPPORT_BINS = 20

# ══════════════════════════════════════════════════════════════════════════
#  Graph-consistent NN matching
# ══════════════════════════════════════════════════════════════════════════
MATCH_N_CAND      = 6    # candidate spline arms per keypoint (local minima of
                         # the distance profile)
MATCH_CONT_W      = 2.0  # along-spline continuity weight vs raw distance in
                         # the Viterbi cost; raise to reject wrong-arm jumps
MATCH_Z_DOWNWEIGHT = 20.0 # z divided by this in the MATCHING metric only
                         # (ordering pinned to trustworthy lateral position;
                         # depth still breaks crossing ties)

# ══════════════════════════════════════════════════════════════════════════
#  Match-quality score  (warm_ordering → optim constraint-box softening)
# ══════════════════════════════════════════════════════════════════════════
# Continuous per-keypoint quality q ∈ [MATCH_Q_MIN, 1] built in
# Order._match_quality from signals the match already computes: lateral/z
# residual vs their caps, EKF maha², arm-ambiguity margin (nearest ALTERNATIVE
# spline arm vs the chosen one), and crossing proximity.  optim widens each
# keypoint's constraint box by (1 + QUALITY_BOX_WIDEN_GAIN·(1−q)) so a poorly
# matched keypoint guides the fit instead of dictating it.
MATCH_Q_CROSS_CORE_PX  = 40.0  # px: keypoints within this radius of a crossing
                               # CENTRE are struck from the NN match entirely
                               # (t there is genuinely ambiguous) and recovered
                               # via _crossing_t_by_axis.  Was previously an
                               # implicit 50 (intersection_radius) around ANY
                               # arm pixel — arm keypoints now match normally.
MATCH_Q_CROSS_SIGMA_PX = 60.0  # px: falloff of the continuous crossing-
                               # proximity penalty on q (arms near the core
                               # score lower, smoothly)
MATCH_Q_CROSS_PEN      = 0.9   # max fraction of q removed AT the centre
MATCH_Q_AMB_TSEP       = 0.05  # |Δt| beyond which a candidate counts as a
                               # DIFFERENT arm for the ambiguity margin
MATCH_Q_DEPTHLESS      = 0.5   # q multiplier for keypoints with no stereo z
MATCH_Q_MIN            = 0.02  # quality floor (never zero a keypoint outright)

# ══════════════════════════════════════════════════════════════════════════
#  Grasp orientation lock  (anti-flip)
# ══════════════════════════════════════════════════════════════════════════
GRASP_LOCK_MAX_DIST = 40.0  # lock idles when the grasp is farther than this
                            # from the thread (mid-handoff); NB detangle's
                            # bind ceiling (DETANGLE_ANCHOR_LOST_MULT ×
                            # DETANGLE_GRASP_RADIUS in ros_thread_reconstruct)
                            # is independently 2×20 = the same 40
GRIP_IDX_TOL        = 2     # knots the nearest-to-grasp index may drift
                            # before re-anchor-instead-of-compare

# ══════════════════════════════════════════════════════════════════════════
#  Length-adaptive control-point count
# ══════════════════════════════════════════════════════════════════════════
ADAPT_M_MIN         = 8    # never fewer knots than this (ceiling is N_CTRL)
ADAPT_EMA_ALPHA     = 0.3  # observed-length smoothing
ADAPT_HYST          = 3    # knots of change required to resample
ADAPT_STREAK_FRAMES = 5    # consecutive RAW frames wanting the new count
                           # before the (lossy) resample may fire

# ══════════════════════════════════════════════════════════════════════════
#  Warm-spline quality  (scribble rejection; read by ros node's gates too)
# ══════════════════════════════════════════════════════════════════════════
WARM_MIN_BEND_RADIUS = 0.5   # min local bend radius before "folded scribble"
                             # (real output ≈ R 1.07–1.81, so 0.5 clears it)
WARM_SCRAMBLE_MAX    = 0.15  # max fraction of ordering-reversing steps

# ══════════════════════════════════════════════════════════════════════════
#  Feature switches  (Order)
# ══════════════════════════════════════════════════════════════════════════
EKF_DENOISE_Z       = True  # write the posterior's z back onto the χ²-gated
                            # matched keypoints (lateral stays raw)
KF_PROJECTION_ORDER = True  # order = argsort(t on the KF spline); skips the
                            # segment-split + greedy assembly (stable order)
