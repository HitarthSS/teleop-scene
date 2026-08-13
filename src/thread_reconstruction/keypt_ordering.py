import copy
from collections import deque
import pdb

# Select the non-interactive Agg backend before importing pyplot: this module
# runs inside a ROS executor thread and the interactive Tk backend leaks/crashes
# ("main thread is not in main loop") when figures are destroyed off-thread.
# Debug plots savefig(), so Agg is sufficient and plt.show() is a no-op.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
import cv2
import pickle
from scipy.interpolate import interp1d

# Top-level imports (previously scattered as lazy imports inside methods)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor
from thread_reconstruction.spline_ekf import SplineEKF
from thread_reconstruction import ekf_params
from thread_reconstruction.optim import Timer
import numpy as np
import os

# Drop into pdb at every error / degenerate return in this module, so a
# silently-skipped frame stops instead of being swallowed by the caller's
# fallback path.  Export THREAD_RECON_BREAK=0 to disable every breakpoint
# in the package at once (unattended runs, live ROS sessions).
DEBUG_BREAK_ON_ERROR = os.environ.get("THREAD_RECON_BREAK", "1") != "0"


class Order():
    def __init__(self, args):
        # All filter tuning lives in ekf_params.py (single source of truth,
        # shared with warm_start / optim / the ros node) — the SplineEKF
        # constructor defaults ARE those values, so no args here.
        self.ekf = SplineEKF()
        self._ekf_initialized = False
        # Captured each frame by warm_ordering: the DEBUG-1 (NN t-assignment)
        # inputs, so the ROS node can render the same view into its debug PNGs.
        self._dbg1 = None
        # Median px distance from the predicted spline to this frame's keypoints
        # (set by warm_ordering).  Drives the data-trust scaling of the optim
        # temporal prior — see PRIOR_TRUST_NN_REF in ros_thread_reconstruct.
        self._last_nn_median = float('nan')
        # Per-keypoint match quality aligned with warm_ordering's RETURNED
        # keypoint array (KF-projection path only; None otherwise).  The ros
        # node forwards it to optim as keypt_quality to soften the constraint
        # boxes of poorly matched keypoints — see _match_quality.
        self._match_q_ordered = None
        # Set each frame by run_warm_ordering_with_ekf: True when the EKF was
        # re-seeded from the raw warm spline this frame (drift/scribble recovery),
        # for the debug visualiser to flag.  Frame-0 initial seed is NOT a reseed.
        self._ekf_reseeded = False
        # Write the EKF posterior's depth back onto the χ²-gated matched
        # keypoints after each update (z only — lateral px stays raw), so optim
        # receives depth-denoised keypoints.  See run_warm_ordering_with_ekf.
        self.EKF_DENOISE_Z = ekf_params.EKF_DENOISE_Z
        # When True, warm_ordering ORDERS keypoints purely by their (graph-
        # consistent, crossing-robust) t-projection onto the KF spline —
        # order = argsort(matched_t) — and SKIPS the segment-split + greedy
        # assembly (Steps A-F).  The KF spline is temporally stable, so the
        # t-order is stable frame-to-frame (no assembly re-joins/flips → no
        # "order changes completely every few frames").  False = old assembly.
        self.KF_PROJECTION_ORDER = ekf_params.KF_PROJECTION_ORDER
        # --time: print per-stage Timer breakdowns even in speedy mode
        self.timing = getattr(args, 'time', False)

    # ══════════════════════════════════════════════════════════════════════
    #  warm-spline sanity check
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _spline_quality(spline, ref=None, n=200, k=4):
        """
        Loop/intersection-invariant health metrics for a [0,1]-parameterised
        warm spline.

        The thread is physically allowed to loop and self-intersect freely, so
        global shape measures (arc/chord tortuosity, adjacent-segment reversals)
        are useless here — a legal loop has a tiny chord and tight apex yet is
        perfectly valid.  What actually distinguishes a valid thread from a
        "scribble" is (a) it respects a minimum bend radius, and (b) projecting
        keypoints onto it yields monotone t.  Both are measured locally and are
        invariant to loops and crossings.

        Returns (min_bend_radius, scramble_frac):
          • min_bend_radius = smallest local radius of curvature along the curve
            (spatial units of the spline), computed as 1/κ from the Menger
            curvature of point triples spaced k samples apart.  A physical
            thread stays above _WARM_MIN_BEND_RADIUS everywhere, INCLUDING
            through loops; only a folded scribble has sub-radius kinks.  A low
            percentile (not the raw min) is used so a single noisy sample does
            not veto an otherwise clean spline.
          • scramble_frac  = fraction of ordering-reversing steps when the
            ordered samples of the trusted reference spline `ref` are
            nearest-point-projected onto this spline.  0 for a warp that
            preserves ordering — even through loops — and high for one whose
            NN-projection would scramble the keypoint ordering downstream.
            np.nan when no reference is supplied (e.g. judging raw itself).
        """
        t = np.linspace(0.0, 1.0, n)
        p = np.asarray(spline(t), dtype=float)

        # A spline with non-finite samples (e.g. from a failed QP solve) is the
        # worst possible quality — report it as such instead of crashing the
        # cKDTree / curvature math downstream.
        if not np.all(np.isfinite(p)):
            return 0.0, 1.0

        # ── minimum local bend radius (Menger curvature over a k-sample base) ──
        A, B, C = p[:-2 * k], p[k:-k], p[2 * k:]
        AB = np.linalg.norm(B - A, axis=1)
        BC = np.linalg.norm(C - B, axis=1)
        CA = np.linalg.norm(A - C, axis=1)
        area  = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
        denom = AB * BC * CA
        curv  = np.where(denom > 1e-9, 4.0 * area / np.where(denom > 1e-9, denom, 1.0), 0.0)
        radius = 1.0 / np.where(curv > 1e-9, curv, 1e-9)
        min_bend_radius = float(np.percentile(radius, 2)) if radius.size else np.inf

        # ── ordering monotonicity relative to the trusted reference ───────────
        if ref is None:
            scramble_frac = float('nan')
        else:
            rp = np.asarray(ref(t), dtype=float)
            if not np.all(np.isfinite(rp)):
                # reference unusable → scramble unmeasured (same as no ref)
                return min_bend_radius, float('nan')
            _, idx = cKDTree(p).query(rp)
            t_proj = t[idx]
            d = np.diff(t_proj)
            pos, neg = float((d > 0).sum()), float((d < 0).sum())
            # steps opposing the dominant direction → orientation-invariant, so
            # a reference that runs the opposite way is not counted as scramble.
            scramble_frac = min(pos, neg) / max(pos + neg, 1.0)
        return min_bend_radius, scramble_frac

    # Minimum local bend radius (spatial units) a warm/prior/optim spline may
    # have before it is treated as a folded scribble.  Read cross-class by
    # ros_thread_reconstruct's prior_ok / quality_ok gates.  Sit WELL BELOW the
    # thread's real minimum bend radius or legitimately tight threads are
    # falsely rejected (real output ≈ R 1.07–1.81, so 0.5 clears it).
    _WARM_MIN_BEND_RADIUS = ekf_params.WARM_MIN_BEND_RADIUS
    # Above this fraction of ordering-reversing steps the warp would scramble
    # the downstream keypoint ordering and is rejected.
    _WARM_SCRAMBLE_MAX    = ekf_params.WARM_SCRAMBLE_MAX

    # Max endpoint separation (px) the COLD greedy assembly may bridge.  The
    # warm path has had its own cap (ASSEMBLY_MAX_JOIN_PX in warm_ordering)
    # for a while; the cold path had none, so its `while segments:` loop
    # consumed every segment no matter how far away — `best_cost` was only
    # ever compared against other candidates, never against a threshold.  That
    # is a direct source of optim QP infeasibility, since one distant join
    # becomes a huge arc-length step in the parameterisation.
    #
    # Applied to the RAW endpoint distance, deliberately NOT to the cost: the
    # cost is distance x (1 + 3(1-alignment)), so a well-aligned far segment
    # can outrank a poorly-aligned near one (500px x 1.0 beats 200px x 4.0)
    # and alignment must not be able to buy down an implausible gap.
    # Segments left unreachable are dropped rather than bridged.
    COLD_ASSEMBLY_MAX_JOIN_PX = 60.0

    # ══════════════════════════════════════════════════════════════════════
    #  graph-consistent NN matching  (Viterbi over adjacency chains)
    # ══════════════════════════════════════════════════════════════════════
    # Independent k=1 NN projection onto the warm spline picks the spatially
    # nearest spline point, which at a loop/self-crossing can be the WRONG arm
    # while the correct arm sits only slightly farther.  These two constants
    # govern the graph-consistent replacement in _graph_consistent_match.
    #
    # Number of candidate spline "arms" considered per keypoint.  Candidates
    # are the K smallest LOCAL MINIMA of the keypoint→spline distance profile
    # (each local min = one distinct pass of the spline near the keypoint), NOT
    # the K globally nearest dense samples (those cluster on one arm and never
    # expose the alternative arm).  Raise if the spline can pass a keypoint on
    # more than ~5 distinct arms.
    _MATCH_N_CAND = ekf_params.MATCH_N_CAND
    # Weight of the along-spline continuity residual vs. the raw point-to-spline
    # distance in the Viterbi cost.  Both terms are in the same spatial units —
    # the residual is |arc-length between two consecutive matches − physical
    # keypoint spacing|, ~0 for a strand that stays on one arm and large for a
    # wrong-arm jump.  1.0 weights the two equally; raise to reject wrong-arm
    # jumps harder, lower to trust the raw spatial NN more.
    _MATCH_CONT_W = ekf_params.MATCH_CONT_W
    # Depth down-weight of the MATCHING metric: the z axis of both the
    # keypoints and the dense warm spline is divided by this before the NN /
    # Viterbi matching, so a unit of z-residual counts 1/this as much as a
    # unit of lateral residual when deciding WHICH spline point (and hence
    # which t) a keypoint matches.  Rationale: stereo z is ~4x noisier than
    # the lateral position (see sigma_meas vs sigma_meas_z), and wherever the
    # spline slopes in z, z-noise drags the nearest-point projection ALONG t —
    # two neighbouring keypoints with opposite z-noise can swap rank, and a
    # locally flipped order makes the optim QP infeasible.  Whitening z keeps
    # the ordering pinned to the trustworthy lateral position while depth (arm
    # separation ~10 units ≫ scaled noise) still breaks crossing ties.
    # 1.0 = trust z fully (pure Euclidean); larger = lateral dominates; the
    # acceptance caps and χ² gates are NOT affected (they judge unscaled
    # residuals with their own lateral/depth budgets).
    _MATCH_Z_DOWNWEIGHT = ekf_params.MATCH_Z_DOWNWEIGHT

    # ── coverage-aware acceptance normalisation ───────────────────────────
    # Under DETANGLE the warm spline deliberately describes only a SEGMENT of
    # the thread, so matched/all-keypoints reads "ambiguous" (and the
    # divergence monitor reads "diverged") on perfectly healthy frames.  The
    # acceptance fraction is instead measured against what the spline CLAIMS:
    # keypoints within _COV_BAND_MULT x the lateral hard cap AND within the
    # z acceptance cap (a depth-separated section can never match, so it must
    # not count against coverage).  Combined (min) with the fraction of the
    # spline's own t-span that has matched support, over _COV_SUPPORT_BINS
    # bins — so a stale spline kept alive by a few scattered survivors still
    # reads diverged even when the band is empty of other keypoints.
    _COV_BAND_MULT    = ekf_params.COV_BAND_MULT
    _COV_SUPPORT_BINS = ekf_params.COV_SUPPORT_BINS

    # ══════════════════════════════════════════════════════════════════════
    #  graceful EKF degradation under ambiguity
    # ══════════════════════════════════════════════════════════════════════
    # A per-frame ambiguity score in [0,1] blends the ordering continuously
    # from EKF-driven (amb=0: trust the warm spline, order by spline-t) toward
    # raw-keypoint-driven (amb→1: let the adjacency graph re-order), with full
    # cold keypt_ordering at the extreme.  It is built from the Mahalanobis
    # gate — a low pass-fraction and/or a high median maha² both mean the EKF
    # prediction no longer explains the keypoints — plus a floor from the warm
    # spline's scramble metric.  amb then (a) demotes the least-confident warm
    # matches to the raw adjacency path, (b) widens the unmatched-segment join
    # thresholds, and (c) inflates the EKF covariance before the update so the
    # filter unlocks and follows the data over the next few frames.
    #
    # Gate pass-fraction mapping: ≥HI → no ambiguity, ≤LO → fully ambiguous.
    _AMB_GATE_HI = ekf_params.AMB_GATE_HI
    _AMB_GATE_LO = ekf_params.AMB_GATE_LO
    # median(maha²)/χ² mapping: ≤LO → no ambiguity, ≥HI → fully ambiguous.
    _AMB_MAHA_LO = ekf_params.AMB_MAHA_LO
    _AMB_MAHA_HI = ekf_params.AMB_MAHA_HI
    # Unmatched-segment join thresholds are scaled by (1 + gain·amb) so raw
    # adjacency segments span more freely when the warm-t is untrustworthy.
    _AMB_WIDEN_GAIN = ekf_params.AMB_WIDEN_GAIN
    # EKF covariance is inflated ×(1 + gain·amb) before the update; capped so a
    # run of ambiguous frames can't blow the state variance up unbounded.
    _AMB_P_GAIN    = ekf_params.AMB_P_GAIN
    _AMB_P_MAXDIAG = ekf_params.AMB_P_MAXDIAG
    # Ceiling on how far the `floor` argument alone may drive ambiguity.
    # a_gate/a_maha are DIRECT measurements of whether this frame's keypoints
    # agree with the EKF; the floor is a HEURISTIC about the previous frame's
    # spline shape (warm-spline scramble).  Uncapped it can saturate ambiguity
    # to 1.0 while both measurements read exactly 0.0 — observed on a stationary
    # frame (motion=0.00, median nn 2.6px, 66/77 through the χ² gate) whose 54
    # good matches were all discarded on the strength of scramble=0.17 alone.
    # A scrambled warm spline should heavily discount the EKF, not silence the
    # evidence.  a_gate/a_maha are NOT capped — real disagreement still reaches
    # 1.0 and hands off to cold ordering as designed.
    _AMB_FLOOR_MAX = ekf_params.AMB_FLOOR_MAX
    # Minimum fraction of the χ² gate that keep_bound may shrink to (see the
    # keep_bound comment in warm_ordering).  At full ambiguity the warm path
    # then survives on its most confident matches (maha² ≤ 0.2·5.99 ≈ 1.2)
    # instead of collapsing to zero survivors; if fewer than 4 clear that bar
    # the cold fallback still fires, but because the DATA failed rather than
    # because the bound hit zero.  Lower → hands off to cold ordering sooner.
    _AMB_KEEP_FLOOR = ekf_params.AMB_KEEP_FLOOR

    @classmethod
    def _ambiguity(cls, pass_frac, maha_ratio, floor=0.0):
        """
        Combine the gate pass-fraction and the median normalised Mahalanobis
        distance into a single ambiguity score in [0,1] (worst signal wins),
        never below `floor` (used to inject the warm-spline scramble metric).

        `floor` is capped at _AMB_FLOOR_MAX: it is a heuristic about the warm
        spline, not a measurement of this frame, so it may discount the direct
        signals but never override them completely.
        """
        a_gate = np.clip((cls._AMB_GATE_HI - pass_frac) /
                         max(cls._AMB_GATE_HI - cls._AMB_GATE_LO, 1e-6), 0.0, 1.0)
        a_maha = np.clip((maha_ratio - cls._AMB_MAHA_LO) /
                         max(cls._AMB_MAHA_HI - cls._AMB_MAHA_LO, 1e-6), 0.0, 1.0)
        floor  = min(float(floor), cls._AMB_FLOOR_MAX)
        return float(max(a_gate, a_maha, floor))

    @staticmethod
    def _candidate_matches(kpts, warm_dense, n_cand):
        """
        For each keypoint return up to `n_cand` candidate matches on the warm
        spline — one per distinct local minimum of the keypoint→spline distance
        profile (i.e. one per arm of the spline that passes near the keypoint),
        sorted nearest-first.

        Dimension-agnostic: `kpts` (N, d) and `warm_dense` (M, d) just need the
        same space.  warm_ordering calls it with CAMERA-FRAME 3-D points
        (unprojected keypoints vs the warm spline itself), so at an image-space
        crossing the two strands are still separated by their depth.

        Returns
        -------
        cand_idx : (N, n_cand) int   dense-spline index of each candidate
                                     (unused slots padded with the nearest one)
        cand_d   : (N, n_cand) float distance to each candidate
        """
        D = cdist(kpts, warm_dense)                    # (N, M)
        N, M = D.shape

        # interior local minima: strictly below the left neighbour and <= the
        # right one, so a flat plateau spawns a single candidate, not two.
        lo = np.zeros_like(D, dtype=bool)
        lo[:, 1:-1] = (D[:, 1:-1] < D[:, :-2]) & (D[:, 1:-1] <= D[:, 2:])
        lo[:, 0]  = D[:, 0]  < D[:, 1]                 # endpoints descend inward
        lo[:, -1] = D[:, -1] < D[:, -2]

        cand_idx = np.zeros((N, n_cand), dtype=int)
        cand_d   = np.full((N, n_cand), np.inf)
        for i in range(N):
            mins = np.where(lo[i])[0]
            if len(mins) == 0:                         # degenerate flat/short
                mins = np.array([int(np.argmin(D[i]))])
            order = mins[np.argsort(D[i, mins])]       # nearest arm first
            k = min(len(order), n_cand)
            cand_idx[i, :k] = order[:k]
            cand_d[i, :k]   = D[i, order[:k]]
            cand_idx[i, k:] = order[0]                 # pad w/ nearest (no-op state)
            cand_d[i, k:]   = D[i, order[0]]
        return cand_idx, cand_d

    @staticmethod
    def _match_quality(d_lat, d_z, lat_cap, z_cap, z_valid, maha2,
                       nn_dists, matched_t, cand_idx, cand_d, t_dense,
                       d_cross):
        """Continuous per-keypoint match quality q ∈ [MATCH_Q_MIN, 1].

        The binary gates upstream decide WHO gets matched; this scores HOW
        WELL, so downstream (optim's constraint boxes) can prefer clean,
        unambiguous matches over survivors that only just cleared the gates.
        Everything is a product of [0, 1] terms from signals the match
        already computed:

          lateral / depth residual — exp(-(d/cap)²) against the SAME caps the
              gate used, so "barely passed" scores low and "dead on" ≈ 1;
          depthless               — flat MATCH_Q_DEPTHLESS (its z is invented);
          EKF agreement           — exp(-maha²/χ²_ref) when a filter gated;
          arm ambiguity margin    — Lowe-style 1 − d_chosen/d_alt where d_alt
              is the nearest CANDIDATE on a different arm (|Δt| >
              MATCH_Q_AMB_TSEP).  No alternative arm → 1 (unambiguous).  A
              Viterbi choice that overrode a nearer arm scores ~0 — the
              assignment leaned on continuity, i.e. it IS geometrically
              ambiguous (overlap);
          crossing proximity      — smooth penalty by distance to the nearest
              crossing CENTRE (the hard core strike is separate).
        """
        q = np.exp(-(d_lat / max(lat_cap, 1e-9)) ** 2)
        q = q * np.where(z_valid,
                         np.exp(-(d_z / max(z_cap, 1e-9)) ** 2),
                         ekf_params.MATCH_Q_DEPTHLESS)
        if maha2 is not None:
            m = np.where(np.isfinite(maha2), maha2,
                         2.0 * ekf_params.CHI2_MATCH_3D)
            q = q * np.exp(-m / ekf_params.CHI2_MATCH_3D)
        t_cand = t_dense[cand_idx]                        # (N, n_cand)
        alt    = np.where(np.abs(t_cand - matched_t[:, None])
                          > ekf_params.MATCH_Q_AMB_TSEP, cand_d, np.inf)
        d_alt  = alt.min(axis=1)
        margin = np.where(np.isfinite(d_alt),
                          np.clip(1.0 - nn_dists / np.maximum(d_alt, 1e-9),
                                  0.0, 1.0),
                          1.0)
        q = q * margin
        q = q * (1.0 - ekf_params.MATCH_Q_CROSS_PEN
                 * np.exp(-0.5 * (d_cross
                                  / ekf_params.MATCH_Q_CROSS_SIGMA_PX) ** 2))
        return np.clip(q, ekf_params.MATCH_Q_MIN, 1.0)

    def _graph_consistent_match(self, kpts, warm_dense, t_dense,
                                adjacents, cont_w, n_cand):
        """
        Match every keypoint to a warm-spline point that is both close to the
        spline AND continuous along the keypoint adjacency graph.

        Dimension-agnostic (all distances live in the space of `kpts` /
        `warm_dense`); warm_ordering calls it with z-WHITENED camera-frame 3-D
        points (z divided by _MATCH_Z_DOWNWEIGHT), so depth still separates
        the arms of an image-space crossing but depth NOISE cannot drag a
        keypoint's t along a z-sloped spline.  Keypoints whose
        stereo z is missing are given a placeholder depth by the caller and
        struck from the warm match afterwards — they only ride along here so a
        strand is not severed at every depth dropout.

        Independent NN matching fails where the spline loops or self-crosses:
        the spatially-nearest arm can be the wrong one.  Here the keypoints are
        linearised into strands along `adjacents` (breaking at junction nodes),
        and a Viterbi pass over each strand chooses, per keypoint, the candidate
        arm whose along-spline spacing best matches the physical spacing of its
        neighbours — so a whole strand commits to one consistent arm.

        Runs over ALL keypoints, crossing ones included — see the strand
        comment below for why they must stay in the graph.  Which keypoints
        become warm matches is the caller's decision (matched_mask), not this
        function's.

        Returns nn_idxs (N,), nn_dists (N,), matched_t (N,), plus the raw
        per-arm candidates cand_idx/cand_d (N, n_cand) so the caller can score
        match AMBIGUITY (chosen arm vs nearest alternative arm).
        """
        N = len(kpts)
        cand_idx, cand_d = self._candidate_matches(
            kpts, warm_dense, n_cand)

        # default for every keypoint = nearest arm (candidate 0) — same as the
        # old k=1 NN; only chain members get overridden below.
        nn_idxs  = cand_idx[:, 0].copy()
        nn_dists = cand_d[:, 0].copy()

        # cumulative arc length along the dense spline → along-spline distance
        seg     = np.linalg.norm(np.diff(warm_dense, axis=0), axis=1)
        s_dense = np.concatenate([[0.0], np.cumsum(seg)])          # (M,)

        n_adj = len(adjacents)

        def nbrs(v):
            if v >= n_adj:
                return []
            return [int(u) for u in adjacents[v] if int(u) < N]

        # ── linearise the adjacency graph into strands ────────────────────────
        # Nodes with >2 in-graph neighbours are junctions; strands break there
        # so each is a simple thread run.  Greedy nearest-neighbour walk from
        # each degree-≤1 endpoint (mirrors the cold-crawl in keypt_ordering).
        #
        # Strands span EVERY keypoint, crossing ones included: `exclude`
        # governs membership of the WARM MATCH, not membership of the graph.
        # Deleting excluded keypoints from the graph as well used to sever a
        # strand at every crossing, so the runs either side did INDEPENDENT
        # Viterbi passes and were free to commit to different arms — exactly
        # where the two arms are equidistant and continuity is the only usable
        # signal, i.e. the one place this matcher exists to help.  Their t is
        # now graph-consistent instead of raw single-NN, which is strictly
        # better for the ordering; the caller still strikes them from
        # matched_mask, so none of them becomes a warm match or an EKF
        # observation.  True junction nodes (degree > 2) still break the walk
        # below — a real junction cannot be traversed unambiguously.
        all_nodes = list(range(N))
        deg       = {v: len(nbrs(v)) for v in all_nodes}
        visited   = np.zeros(N, dtype=bool)
        chains    = []

        def walk(start):
            chain = [start]; visited[start] = True; curr = start
            while True:
                cand = [u for u in nbrs(curr)
                        if not visited[u] and deg[u] <= 2]
                if not cand:
                    break
                nxt = min(cand, key=lambda u:
                          np.linalg.norm(kpts[u] - kpts[curr]))
                chain.append(nxt); visited[nxt] = True; curr = nxt
            return chain

        for v in all_nodes:                 # endpoints first (open strands)
            if not visited[v] and deg[v] <= 1:
                chains.append(walk(v))
        for v in all_nodes:                 # then any closed-loop remainder
            if not visited[v] and deg[v] <= 2:
                chains.append(walk(v))
        # leftover junction nodes (deg ≥ 3) keep their raw single-NN match.

        # ── Viterbi per strand ────────────────────────────────────────────────
        for chain in chains:
            m = len(chain)
            if m < 2:
                continue                                 # nothing to smooth
            cs    = s_dense[cand_idx[chain]]             # (m, n_cand) arc pos
            cd    = cand_d[chain]                        # (m, n_cand) emission
            dphys = np.linalg.norm(
                np.diff(kpts[chain], axis=0), axis=1) # (m-1,) physical spacing

            cost = cd[0].copy()                          # (n_cand,)
            back = np.zeros((m, n_cand), dtype=int)
            for i in range(1, m):
                # trans[k_prev, k_cur] = |arc(prev,cur) − physical spacing|
                arc   = np.abs(cs[i][None, :] - cs[i - 1][:, None])
                trans = cont_w * np.abs(arc - dphys[i - 1])
                tot   = cost[:, None] + trans            # (n_cand, n_cand)
                back[i] = np.argmin(tot, axis=0)
                cost    = cd[i] + tot.min(axis=0)
            k = int(np.argmin(cost))
            for i in range(m - 1, -1, -1):
                v = chain[i]
                nn_idxs[v]  = cand_idx[v, k]
                nn_dists[v] = cand_d[v, k]
                k = back[i, k]

        matched_t = t_dense[nn_idxs]
        return nn_idxs, nn_dists, matched_t, cand_idx, cand_d

    # ══════════════════════════════════════════════════════════════════════
    #  crossing keypoint t-recovery (KF_PROJECTION_ORDER path)
    # ══════════════════════════════════════════════════════════════════════

    # Max perpendicular distance (px) from a crossing arm's RANSAC axis for a
    # keypoint to count as lying ON that arm.  Too large and the OTHER arm's
    # keypoints get swept in near the core, which is the ambiguity this whole
    # routine exists to avoid; too small and a slightly curved arm loses its
    # anchors.
    _ARM_PERP_MAX = 15.0
    # How far along the arm (px, from the arm centroid) to look for t-anchors.
    # Must comfortably exceed warm_ordering's 50 px crossing-exclusion radius,
    # otherwise every keypoint on the arm is itself excluded and there is
    # nothing left to interpolate between.
    _ARM_ANCHOR_RADIUS = 150.0
    # Anchors required on EACH side of the arm centroid.  Demanding both sides
    # makes every recovered t an interpolation across the crossing, never an
    # extrapolation off one end.
    _ARM_MIN_ANCHORS_PER_SIDE = 2
    # An arm whose anchors span less t than this carries no usable gradient.
    _ARM_MIN_T_SPAN = 5e-3
    # Fraction of consecutive anchor pairs that must agree with the arm's
    # overall t-direction.  Below this the anchors straddle two t-branches
    # (the arm was fitted across a loop) and the arm is abandoned.
    _ARM_MONOTONE_FRAC = 0.75
    # χ²(3) gate applied to each re-inserted keypoint AT ITS ASSIGNED t.  Same
    # 95% threshold as the 3-D match gate in warm_ordering, and it inherits the
    # same motion-adaptive loosening inside mahalanobis_gate.  Lower it to be
    # stricter about what the axis interpolation is allowed to put back.  Note
    # this judges the full 3-D innovation (R keeps z loose), so a crossing
    # keypoint with garbage stereo depth now fails here instead of re-entering
    # the order on image position alone.
    _CROSSING_CHI2 = ekf_params.CHI2_MATCH_3D  # merged with warm_ordering's match gate
    # A keypoint sitting on the crossing CORE is near-equidistant from both
    # arms, so perpendicular distance cannot say which branch it belongs to —
    # and the two arms' t-values are far apart, so guessing wrong injects a
    # large ordering error.  When the two best arm claims differ by less than
    # this margin (px), the keypoint is dropped instead of assigned.  This is
    # the KF_PROJECTION_ORDER analogue of Step C.5's CORE_EXCLUSION_RADIUS.
    _ARM_AMBIG_MARGIN = 4.0

    def _crossing_t_by_axis(self, kpts_2d, kd_tree, intersection_segments,
                            crossing_kpt_ids, matched_mask, matched_t,
                            ekf=None, t_dense=None, kpts_3d=None, speedy=False):
        """Recover a spline parameter t for the crossing keypoints that were
        excluded from the warm/EKF match.

        Those keypoints were excluded precisely because their nearest-point
        projection onto the warm spline is ambiguous: at a crossing the wrong
        arm is often the spatially nearer one, so matched_t[i] can name a t on
        the other branch.  Interpolating instead ALONG the crossing's own
        RANSAC axis sidesteps that — the axis says which arm a keypoint is on,
        and the matched keypoints further out along that same arm supply
        trustworthy t-values to interpolate between.

        Per crossing arm:
          1. collect keypoints within _ARM_ANCHOR_RADIUS of the arm centroid
             whose perpendicular offset from the axis is ≤ _ARM_PERP_MAX;
          2. split them into ANCHORS (warm-matched, t trusted) and ORPHANS
             (in crossing_kpt_ids, t discarded);
          3. require ≥ _ARM_MIN_ANCHORS_PER_SIDE anchors on both sides of the
             centroid, a t-span ≥ _ARM_MIN_T_SPAN, and t monotone in the axis
             coordinate for ≥ _ARM_MONOTONE_FRAC of consecutive anchor pairs;
          4. linearly interpolate each orphan's t from its axis projection,
             keeping only orphans strictly inside the anchor span.

        An arm failing any check contributes nothing — its keypoints stay
        dropped, as before.  Before an arm may claim an orphan, the orphan must
        also be _ARM_AMBIG_MARGIN closer to THIS arm's axis than to every
        SIBLING arm's axis of the same crossing.  That test is purely
        geometric, so it still fires when the sibling arm was itself skipped —
        otherwise a surviving arm would sweep up the failed arm's core-adjacent
        keypoints unopposed and stamp its own t-branch on them.

        Finally, every survivor must pass a 3-D χ²(3) gate against the EKF at
        the t it was just assigned (see the block near the end) — the geometry
        above proves only that a point lies along an axis, not that the thread
        is there.  The arm geometry itself stays in the IMAGE plane (the RANSAC
        axes come from 2-D intersection detection and have no depth); only the
        final EKF check is 3-D.  Without an `ekf` that check is skipped and the
        result is geometry-only.

        Returns {keypoint_id: t} for orphans only (never overwrites a matched
        keypoint's t, and never feeds the EKF update — see _make_ekf_obs).
        """
        if not intersection_segments or not crossing_kpt_ids:
            return {}

        matched_mask = np.asarray(matched_mask, dtype=bool)
        matched_t    = np.asarray(matched_t, dtype=float)
        claims       = {}          # kpt_id → [(perp, t, arm_id), ...]
        arm_id       = 0
        n_arms_used  = 0
        n_arms_skip  = 0
        n_ambig      = 0

        def _perp_to(axis_vec, centroid, ids):
            """Perpendicular distance (px) of kpts_2d[ids] from a line."""
            vec = kpts_2d[ids] - centroid
            prj = vec @ axis_vec
            return np.linalg.norm(vec - np.outer(prj, axis_vec), axis=1)

        for crossing in intersection_segments:
            # Normalise every arm's geometry FIRST: the sibling-axis ambiguity
            # test below needs all of them, including arms that go on to fail
            # their own anchor checks.
            arms = []
            for axis_seg in crossing:
                av  = np.asarray(axis_seg['axis_vec'], dtype=float).reshape(2)
                nrm = np.linalg.norm(av)
                if nrm < 1e-9:
                    continue
                arms.append((av / nrm,
                             np.asarray(axis_seg['centroid'],
                                        dtype=float).reshape(2)))

            for ai, (axis_vec, centroid) in enumerate(arms):
                arm_id += 1
                cand = np.asarray(
                    kd_tree.query_ball_point(centroid, self._ARM_ANCHOR_RADIUS),
                    dtype=int)
                if cand.size == 0:
                    n_arms_skip += 1
                    continue

                # Decompose each candidate into (along-axis, off-axis) parts.
                vec  = kpts_2d[cand] - centroid
                proj = vec @ axis_vec
                perp = np.linalg.norm(vec - np.outer(proj, axis_vec), axis=1)
                on   = perp <= self._ARM_PERP_MAX
                cand, proj, perp = cand[on], proj[on], perp[on]
                if cand.size == 0:
                    n_arms_skip += 1
                    continue

                is_anchor = matched_mask[cand] & np.isfinite(matched_t[cand])
                a_p, a_t  = proj[is_anchor], matched_t[cand][is_anchor]
                if (int((a_p < 0).sum()) < self._ARM_MIN_ANCHORS_PER_SIDE or
                        int((a_p > 0).sum()) < self._ARM_MIN_ANCHORS_PER_SIDE):
                    n_arms_skip += 1
                    continue

                # Walk the arm in the direction of increasing t, so np.interp
                # gets a strictly increasing x-array.
                srt      = np.argsort(a_p)
                a_p, a_t = a_p[srt], a_t[srt]
                sign     = 1.0 if a_t[-1] >= a_t[0] else -1.0
                a_x      = sign * a_p
                if sign < 0:
                    a_x, a_t = a_x[::-1], a_t[::-1]

                if float(a_t[-1] - a_t[0]) < self._ARM_MIN_T_SPAN:
                    n_arms_skip += 1
                    continue
                d = np.diff(a_t)
                if d.size == 0 or float((d >= 0).sum()) / d.size < self._ARM_MONOTONE_FRAC:
                    if not speedy:
                        print("  crossing arm: anchors not monotone in t "
                              "(straddles two branches); arm skipped.")
                    n_arms_skip += 1
                    continue
                a_t = np.maximum.accumulate(a_t)   # tolerate small noise

                # Orphans = the excluded crossing keypoints sitting on this arm.
                is_orphan = ~is_anchor & np.array(
                    [int(k) in crossing_kpt_ids for k in cand], dtype=bool)
                o_ids, o_x, o_perp = (cand[is_orphan],
                                      sign * proj[is_orphan],
                                      perp[is_orphan])
                # Strictly inside the anchor span — no extrapolation.
                inside = (o_x > a_x[0]) & (o_x < a_x[-1])
                o_ids, o_x, o_perp = o_ids[inside], o_x[inside], o_perp[inside]
                if o_ids.size == 0:
                    n_arms_used += 1
                    continue

                # Sibling-axis gate: on the crossing core an orphan is nearly
                # equidistant from both arms, and the arms' t-branches are far
                # apart — guessing there is exactly the error this routine is
                # meant to avoid, so drop rather than assign.
                if len(arms) > 1:
                    sib = np.min(np.column_stack(
                        [_perp_to(av, ct, o_ids)
                         for aj, (av, ct) in enumerate(arms) if aj != ai]),
                        axis=1)
                    clear = (sib - o_perp) >= self._ARM_AMBIG_MARGIN
                    n_ambig += int((~clear).sum())
                    o_ids, o_x, o_perp = (o_ids[clear], o_x[clear],
                                          o_perp[clear])
                n_arms_used += 1
                if o_ids.size == 0:
                    continue

                o_t = np.interp(o_x, a_x, a_t)
                for kid, tv, pd in zip(o_ids, o_t, o_perp):
                    claims.setdefault(int(kid), []).append(
                        (float(pd), float(tv), arm_id))

        # ── resolve per-keypoint arm claims ───────────────────────────────────
        # The sibling gate already leaves at most one claim per CROSSING, but
        # two nearby crossings can still both claim a keypoint; same rule.
        recovered = {}
        for kid, cl in claims.items():
            cl.sort(key=lambda c: c[0])           # nearest arm first
            if (len(cl) > 1 and cl[1][2] != cl[0][2]
                    and cl[1][0] - cl[0][0] < self._ARM_AMBIG_MARGIN):
                n_ambig += 1                      # on the crossing core — drop
                continue
            recovered[kid] = cl[0][1]

        # ── χ² gate on the ASSIGNED t ─────────────────────────────────────────
        # Everything above is geometry: it asks whether a keypoint lies along a
        # fitted axis, never whether the thread is actually THERE.  A 2-D blob
        # of mask noise — the instrument shaft merged into the thread by a bad
        # segmentation — projects onto an axis perfectly happily, which is how
        # 24 blob points once entered the order and made the optim QP
        # infeasible.  So ask the filter: at the t this arm just assigned you,
        # do I predict you at this 3-D position?  Note this is a DIFFERENT
        # question from warm_ordering's match gate, which judged these same
        # keypoints at their nearest-point projection — the ambiguous value
        # that got them excluded in the first place.  Here the t comes from the
        # arm, so the gate is finally being asked something meaningful.  The
        # caller passes kpts_3d with NaN rows for depthless keypoints, so those
        # can never be re-inserted (NaN innovation → inf maha² → rejected).
        n_gated = 0
        if recovered and ekf is not None and t_dense is not None and kpts_3d is not None:
            kids  = np.fromiter(recovered.keys(), dtype=int, count=len(recovered))
            tvals = np.fromiter(recovered.values(), dtype=float, count=len(recovered))
            idx_dense = np.clip(
                np.round(tvals * (len(t_dense) - 1)).astype(int),
                0, len(t_dense) - 1)
            gate, _ = ekf.mahalanobis_gate(
                kpts_3d[kids], idx_dense, t_dense,
                chi2_thresh=self._CROSSING_CHI2)
            gate      = np.asarray(gate, dtype=bool)
            n_gated   = int((~gate).sum())
            recovered = {int(k): float(t)
                         for k, t, g in zip(kids, tvals, gate) if g}
        elif recovered and not speedy:
            # Frame 0 has no filter to check against, so the geometric guards
            # are all there is — flagged rather than silently trusted.
            print("crossing t-recovery: no EKF available, re-inserted "
                  "keypoints are UNGATED this frame.")

        if not speedy or recovered or n_ambig or n_gated:
            print(f"crossing t-recovery: {len(recovered)} keypoint(s) "
                  f"re-inserted from {n_arms_used} arm(s) "
                  f"({n_arms_skip} arm(s) skipped, "
                  f"{n_ambig} core keypoint(s) dropped as ambiguous, "
                  f"{n_gated} rejected by the EKF gate at their assigned t)")
        return recovered

    # ══════════════════════════════════════════════════════════════════════
    #  warm_ordering  — with targeted debug visualisations
    # ══════════════════════════════════════════════════════════════════════

    def run_warm_ordering_with_ekf(self,
                                    mask, keypoints, P1,
                                    warm_thread,          # raw fitted spline from last frame
                                    curr_T, prev_T,
                                    speedy=False,
                                    update_ekf=True,
                                    **kwargs):
        """
        Drop-in wrapper around warm_ordering that manages the EKF lifecycle.
    
        On the very first frame, the raw warm_thread is used directly and the
        EKF is seeded from it.  From the second frame onward the EKF's
        predicted spline is used instead of the raw warm_thread, and the
        matched observations update the filter after the fact.
        """
        self._ekf_reseeded = False          # set True only if we re-seed below
        if warm_thread is None:
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None, None
        
        # ── Catch Invalid/NaN Splines Before They Crash SciPy ─────────────────
        # If warm_thread is a SciPy BSpline object:
        if hasattr(warm_thread, 'c') and np.isnan(warm_thread.c).any():
            print("Warning: warm_thread coefficients contain NaNs. Discarding to prevent solver crash.")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None, None
        # If warm_thread is just a raw NumPy array of points:
        elif isinstance(warm_thread, np.ndarray) and np.isnan(warm_thread).any():
            print("Warning: warm_thread array contains NaNs. Discarding to prevent solver crash.")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None, None
        
        t = Timer(enabled=self.timing or not speedy)
        t.start()
        if not self._ekf_initialized:
            # ── Frame 0: initialise from the first available warm_thread ──────
            self.ekf.initialize(warm_thread)   # P0_scale from ekf_params
            self._ekf_initialized = True
            # Anchor the t-direction to the grasped (near-gripper) end, which is
            # rigid and does not swing out — so a later end-crossing update can
            # be detected and reversed (see lock_orientation_to_grasp).
            # DISABLED: grasp-anchored direction locks are off while the
            # direction handling is being isolated (see the seed-anchor at
            # ros_thread_reconstruct's update_from_thread block).
            # self.ekf.lock_orientation_to_grasp(curr_T[:3, 3], curr_T[:3, :3])
            ekf_ref = None          # use raw warm_thread for the first match
        else:
            # ── Frame k>0: predict using tool kinematics, then match ──────────
            trans = np.identity(4)
            rot = curr_T[:3, :3] @ prev_T[:3, :3].T @ np.eye(3)
            translate = (curr_T[:3, 3] - prev_T[:3, 3])
            trans[:3, :3] = rot
            trans[:3, 3] = translate
            
            # Warp pivot = grasp position at the PREVIOUS frame (must match
            # warm_start's warp; pivoting on the post-move grasp is off by
            # (I-R)·t every frame).
            tool_pos_3d = prev_T[:3, 3]
            self.ekf.predict(trans, tool_pos_3d)
            # Divergence / lock-out recovery: if the last few frames' gate
            # rejected most keypoints while the tool is (now) nearly still, the
            # filter is confidently wrong and its shrunken P would keep rejecting
            # the good data.  Inflate P here — BEFORE this frame's gate — so the
            # gate is loose enough to accept the keypoints and re-acquire.
            # A fired recovery is the "reseed"-equivalent event now (the filter
            # is re-acquired from data), so flag it for the debug visualiser.
            self._ekf_reseeded = self.ekf.maybe_recover_from_divergence()
            ekf_ref = self.ekf                  # pass to warm_ordering for gating
        # Use the EKF-predicted spline as the reference (or raw on frame 0)
        filtered_warm_thread = self.ekf.get_spline() if self._ekf_initialized and ekf_ref is not None else warm_thread

        # ── Reject a scribbled warm spline before it scrambles the ordering ───
        # Both the raw prev_thread warp and the EKF prediction are checked so
        # the log pinpoints which stage produced the bad spline.
        # Raw is the trusted reference, so it is judged on bend radius only.
        # The filtered/EKF spline is additionally checked for ordering scramble
        # relative to raw (nearest-point projection of raw's ordered samples).
        raw_radius, _        = self._spline_quality(warm_thread)
        f_radius,  f_scramble = self._spline_quality(filtered_warm_thread,
                                                     ref=warm_thread)
        src = "ekf" if filtered_warm_thread is not warm_thread else "raw"
        print(f"warm-spline quality: raw(prev_thread) R={raw_radius:.2f} "
              f"| used({src}) R={f_radius:.2f} scramble={f_scramble:.2f}")

        t.stop("[order] ekf predict + quality check")
        t.start()
        # Warm-spline scramble feeds the ordering's ambiguity FLOOR: a spline
        # that already reverses some ordering (but not enough to be rejected as
        # a scribble) makes the ordering lean toward the raw keypoints.
        scramble_floor = (0.0 if np.isnan(f_scramble)
                          else float(np.clip(f_scramble / self._WARM_SCRAMBLE_MAX,
                                             0.0, 1.0)))
        new_keypoints, order, nwsk_full, ekf_obs = self.warm_ordering(
            mask, keypoints, P1,
            warm_thread=filtered_warm_thread,
            curr_T=curr_T,
            speedy=speedy,
            ekf=ekf_ref,        # None on frame 0 → falls back to fixed threshold
            ambiguity_floor=scramble_floor,
            **kwargs
        )
    
        t.stop("[order] warm_ordering")
        t.start()
        # ── Update filter with matched observations ───────────────────────────
        # Skipped in the KF↔optim loop (update_ekf=False): there the filter is
        # corrected from optim's robust thread AFTER optim runs, not from the raw
        # matched keypoints — so persistently-wrong keypoints can't bias it.
        if update_ekf and ekf_obs is not None:
            matched_t, matched_z_3d, ambiguity, denoise_rows, obs_conf = ekf_obs
            # Degrade (inflate covariance) BEFORE the update so an ambiguous
            # frame raises the Kalman gain — the filter leans on the raw matched
            # keypoints and unlocks from its stale prediction over a few frames.
            self.ekf.degrade(ambiguity, gain=self._AMB_P_GAIN,
                             max_diag=self._AMB_P_MAXDIAG)
            self.ekf.update(matched_t, matched_z_3d, conf=obs_conf)
            # Direction lock: if this update let the under-observed far end fold
            # past the grasp and reverse the spline, snap it back so the grasped
            # (rigid, non-swinging) end keeps its t — stops the thread flipping.
            # DISABLED (see the frame-0 anchor above): grasp-anchored direction
            # locks are off while direction handling is isolated.
            # self.ekf.lock_orientation_to_grasp(curr_T[:3, 3], curr_T[:3, :3])

            # ── EKF depth denoising of the keypoints ──────────────────────────
            # Feed the filter's smoothing back into the keypoints themselves:
            # the POSTERIOR spline at a matched keypoint's t is the MMSE fused
            # estimate of the thread there (prior + all measurements + bending
            # prior), so its z replaces the keypoint's noisy stereo z.  Lateral
            # row/col stay raw — the image measurement is accurate there; z is
            # the noisy axis.  Only the χ²-gated matched keypoints are touched
            # (wrong-arm t assignments never got into ekf_obs), so a crossing
            # can't drag a keypoint to the other branch's depth.  Downstream,
            # optim sees cleaner z and derives tighter, less noisy z-bounds.
            valid = denoise_rows >= 0
            if self.EKF_DENOISE_Z and valid.any() and new_keypoints is not None:
                rows   = denoise_rows[valid]
                post_z = np.asarray(
                    self.ekf.get_spline()(np.asarray(matched_t)[valid]))[:, 2]
                new_keypoints = np.asarray(new_keypoints, dtype=float)
                dz = np.abs(post_z - new_keypoints[rows, 2])
                new_keypoints[rows, 2] = post_z
                print(f"EKF z-denoise: {len(rows)} keypoints  "
                      f"|dz| mean={dz.mean():.2f} max={dz.max():.2f}")
        t.stop("[order] ekf update")
    
        return new_keypoints, order, nwsk_full


    def warm_ordering(self, mask, keypoints, P1,
                      warm_thread=None, prev_keypts=None, curr_T=None,
                      speedy=False, max_dist=200, T_curr_radius=25,
                      adjacents=None, intersection_segments=None,
                      dense_pts=None, keypt_conf=None,
                      ekf=None, ambiguity_floor=0.0):
        # keypt_conf: optional (len(keypoints),) per-keypoint stereo confidence
        # (keypt_selection full_conf, aligned with the full cluster means).
        # Carried into ekf_obs so the EKF update can inflate R for keypoints
        # whose stereo was ambiguous (see SplineEKF.update conf argument).

        # Cleared every call; set only on the successful KF-projection return
        # so the ros node can never pair a stale quality vector with a
        # different frame's keypoints.
        self._match_q_ordered = None

        if warm_thread is None:
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None, None, None

        # ── DEBUG FLAG ────────────────────────────────────────────────────────
        # Set to True to emit the four targeted diagnostic plots even in speedy
        # mode.  Each plot is labelled with its step number so you can compare
        # directly against the equivalent keypt_ordering output.
        if speedy:
            DEBUG = False
        else:
            DEBUG = True

        tw = Timer(enabled=self.timing or not speedy)
        tw.start()
        lin_keypts     = np.linspace(0, 1, 100)
        warm_keypoints = warm_thread(lin_keypts)

        old_keypoints = copy.copy(keypoints)
        keypoints     = np.asarray(keypoints)

        aug_curr_T  = np.append(curr_T[:3, 3], 1.0)
        proj_curr_T = P1 @ aug_curr_T
        proj_curr_T /= proj_curr_T[2] + 1e-7
        proj_curr_T  = proj_curr_T[[1, 0, 2]]

        aug_pts  = np.concatenate(
            (warm_keypoints, np.ones((len(warm_keypoints), 1))), axis=1)
        proj_pts = (P1 @ aug_pts.T).T
        proj_pts /= proj_pts[:, 2:] + 1e-7
        proj_pts[:, 2] = warm_keypoints[:, 2]
        proj_pts        = np.asarray(proj_pts[:, [1, 0, 2]])

        fx, fy = P1[0, 0], P1[1, 1]
        cx, cy = P1[0, 2], P1[1, 2]

        def unproject(kpts_yxz):
            z = kpts_yxz[:, 2]
            return np.stack([
                (kpts_yxz[:, 1] - cx) * z / fx,
                (kpts_yxz[:, 0] - cy) * z / fy,
                z,
            ], axis=1)

        # ── 3-D MATCHING SPACE ────────────────────────────────────────────────
        # All matching below happens in the CAMERA FRAME (unprojected keypoints
        # vs the warm/EKF spline itself): where the thread crosses itself in
        # the image, the two strands are still separated in depth, so the wrong
        # arm can no longer be the nearest one.  The cost is that stereo-depth
        # noise now enters the correspondence — handled by (a) the EKF gate's
        # anisotropic R (z is judged ~4x looser than lateral), (b) a z-noise
        # allowance folded into the absolute NN cap, and (c) depthless
        # keypoints being struck from the warm match entirely.
        kpts_3d = unproject(keypoints)

        # Depth validity: a keypoint with missing/non-positive stereo z has no
        # 3-D position to match.  For MATCHING it gets a placeholder at the
        # median depth (so cdist/Viterbi stay finite and strands are not
        # severed at every dropout); it is struck from matched_mask after the
        # gates, and its gate copy is NaN so no 3-D χ² test can ever pass it.
        z_col   = keypoints[:, 2].astype(float)
        z_valid = np.isfinite(z_col) & (z_col > 1e-6)
        z_med   = float(np.median(z_col[z_valid])) if z_valid.any() else 1.0
        kpts_3d_m = kpts_3d.copy()                   # matching copy (finite)
        if not z_valid.all():
            kpts_3d_m[~z_valid] = unproject(
                np.column_stack([keypoints[~z_valid, 0],
                                 keypoints[~z_valid, 1],
                                 np.full(int((~z_valid).sum()), z_med)]))
        kpts_3d_gate = kpts_3d.copy()                # gate copy (NaN = never)
        kpts_3d_gate[~z_valid] = np.nan

        # px → spline-units conversion at the working depth, used to carry the
        # image-tuned thresholds over to the 3-D metric.
        mm_per_px = z_med / fx

        N_DENSE       = 500
        t_dense       = np.linspace(0, 1, N_DENSE)
        warm_dense_3d = warm_thread(t_dense)

        # Project the dense warm spline into the left image → (row, col) px.
        # No longer used for matching — kept for the debug overlays and for the
        # IMAGE-PLANE nn-median diagnostic that feeds the optim data-trust
        # scaling (whose reference is tuned in px).
        aug_dense      = np.concatenate(
            (warm_dense_3d, np.ones((N_DENSE, 1))), axis=1)
        warm_dense_px  = (P1 @ aug_dense.T).T
        warm_dense_px  = warm_dense_px[:, :2] / (warm_dense_px[:, 2:3] + 1e-7)
        warm_dense_px  = warm_dense_px[:, [1, 0]]          # → (row, col)

        # ── Identify intersection keypoints (excluded from the warm match) ────
        # Built here (before matching) so the graph-consistent matcher can skip
        # them; the dedicated intersection code (Step C.5) rebuilds them into
        # oriented bridge segments and they keep their raw single-NN t.
        kpts_2d = keypoints[:, :2]
        kd_tree = cKDTree(kpts_2d)
        _internal_crossing_axes = []

        # ── Crossing CENTRES + core strike ────────────────────────────────────
        # Only keypoints within MATCH_Q_CROSS_CORE_PX of a crossing's CENTRE
        # (exact axis×axis intersection, same solve as keypt_selection's
        # merged_centroids) are struck from the NN match — right at the core
        # the two arms overlap and any t assignment is guesswork; those are
        # recovered by _crossing_t_by_axis along their own arm.  Keypoints out
        # on the ARMS stay matchable: the graph-consistent Viterbi sorts them
        # fine, and the continuous crossing-proximity term of the match-quality
        # score (below) down-weights them instead.  Previously EVERY keypoint
        # within 50 px of ANY arm pixel was struck, which discarded well-
        # matched arm keypoints wholesale.
        cross_centers = []
        if intersection_segments is not None and len(intersection_segments) > 0:
            for crossing in intersection_segments:
                c = None
                if len(crossing) >= 2:
                    p1 = np.asarray(crossing[0]['centroid'], dtype=float)
                    v1 = np.asarray(crossing[0]['axis_vec'], dtype=float)
                    p2 = np.asarray(crossing[1]['centroid'], dtype=float)
                    v2 = np.asarray(crossing[1]['axis_vec'], dtype=float)
                    try:
                        s = np.linalg.solve(np.column_stack((v1, -v2)), p2 - p1)
                        c = p1 + s[0] * v1
                    except np.linalg.LinAlgError:
                        c = None
                if c is None:
                    c = np.mean([np.asarray(ax['segment_center'], dtype=float)
                                 for ax in crossing], axis=0)
                cross_centers.append(c)

        crossing_kpt_ids = set()
        if cross_centers:
            d_cross = np.min(
                cdist(kpts_2d, np.asarray(cross_centers, dtype=float)), axis=1)
            crossing_kpt_ids = set(
                np.where(d_cross < ekf_params.MATCH_Q_CROSS_CORE_PX)[0].tolist())
        else:
            d_cross = np.full(len(kpts_2d), np.inf)

        # ── Graph-consistent NN matching (CAMERA FRAME, 3-D) ──────────────────
        # Linearise keypoints into strands along the mask-adjacency graph and
        # Viterbi-decode each strand onto ONE consistent spline arm — robust to
        # loops/self-crossings where a wrong arm is spatially nearer than the
        # right one.  Falls back to plain single-NN when no adjacency is given.
        # All distances (candidates, along-spline arc, physical spacing) are in
        # SPLINE UNITS: unprojected keypoints vs the warm spline itself, so
        # depth separates the arms at an image-space crossing and the arc term
        # measures true 3-D length instead of foreshortened pixels.
        #
        # crossing_kpt_ids is deliberately NOT passed: strands must run THROUGH
        # a crossing for the continuity term to keep both sides on one arm.
        # The crossing keypoints are struck from matched_mask below instead, so
        # they still never become warm matches or EKF observations.
        #
        # The metric is z-WHITENED (see _MATCH_Z_DOWNWEIGHT): matching decides
        # each keypoint's t, and t must not wobble with stereo depth noise.
        # Only these scaled copies see the down-weight — the caps and χ² gates
        # below judge the unscaled residuals.
        _wscale     = np.array([1.0, 1.0, 1.0 / self._MATCH_Z_DOWNWEIGHT])
        kpts_match  = kpts_3d_m * _wscale
        dense_match = warm_dense_3d * _wscale
        if adjacents is not None:
            nn_idxs, nn_dists, matched_t, cand_idx, cand_d = \
                self._graph_consistent_match(
                    kpts_match, dense_match, t_dense, adjacents,
                    cont_w=self._MATCH_CONT_W, n_cand=self._MATCH_N_CAND)
        else:
            # Same per-arm candidate search as the matcher (candidate 0 IS the
            # global nearest), so the quality score's ambiguity margin works on
            # this branch too.
            cand_idx, cand_d = self._candidate_matches(
                kpts_match, dense_match, self._MATCH_N_CAND)
            nn_idxs   = cand_idx[:, 0].copy()
            nn_dists  = cand_d[:, 0].copy()
            matched_t = t_dense[nn_idxs]

        # Ceiling (SPLINE UNITS) on how far a keypoint may sit from the warm
        # spline to be matched.  Applied in BOTH branches — the Mahalanobis gate
        # alone can accept far points when the predicted covariance is large, so
        # this caps absolute distance regardless.
        #
        # ANISOTROPIC: the residual is split into its LATERAL (camera x, y) and
        # DEPTH (z) components and each is capped on its own budget.  A single
        # Euclidean ball does not work here — its radius must be inflated to
        # ~3σ_z so ordinary depth noise cannot reject a good match, but that
        # same radius then leaks ~3σ_z (≈60px at the working depth) of LATERAL
        # slack, and the spline happily matched keypoints that were far too far
        # in the image.  Splitting restores the old px-tuned lateral strictness
        # while depth noise spends only its own 3σ_z allowance.
        #
        # The lateral base/hard values keep their image-plane tuning (20/120 px
        # — see the motion-adaptivity story below), converted to spline units
        # at the working depth via mm_per_px.
        #
        # MOTION-ADAPTIVE, because a FIXED cap breaks under ROTATION.  The warp
        # deliberately does not swing the far thread (w = exp(-d/deform_radius)),
        # so the prediction error grows as distance x angle — unbounded in the
        # rotation angle, unlike translation error which is bounded by |t|.  A
        # fixed 20px cap then vetoes EVERY keypoint (nn median ~70px) even though
        # the covariance-aware χ² gate accepts them all, matching collapses to
        # <4, warm_ordering returns None, and the cold fallback can come back
        # direction-flipped — which pins the warm source and stops recovery.
        # Scaling with the same motion signal the χ² gates use keeps the cap
        # tight at rest (rejects genuinely loose matches) and permissive exactly
        # when the prediction is known to be poor.  The depth cap gets the same
        # motion loosening (the warp errs in z too), with its own 9σ_z ceiling
        # mirroring the 6x lateral base→hard span.
        NN_MAX_DIST_BASE_PX = ekf_params.NN_MAX_DIST_BASE_PX  # cap, tool still
        NN_MAX_DIST_HARD_PX = ekf_params.NN_MAX_DIST_HARD_PX  # sanity ceiling
        _mot   = float(getattr(ekf, '_last_motion', 0.0)) if ekf is not None else 0.0
        _gain  = float(getattr(ekf, 'gate_motion_gain', 2.0)) if ekf is not None else 0.0
        _sig_z = float(getattr(ekf, 'sigma_meas_z', 8.0)) if ekf is not None else 8.0
        _lat_cap = min(NN_MAX_DIST_HARD_PX,
                       NN_MAX_DIST_BASE_PX * (1.0 + _gain * _mot)) * mm_per_px
        _z_cap   = min(9.0 * _sig_z, 3.0 * _sig_z * (1.0 + _gain * _mot))
        _res     = kpts_3d_m - warm_dense_3d[nn_idxs]
        d_lat    = np.linalg.norm(_res[:, :2], axis=1)
        d_z      = np.abs(_res[:, 2])
        cap_ok   = (d_lat < _lat_cap) & (d_z < _z_cap)

        # How far the PREDICTED (motion-model-warped) spline sits from this
        # frame's keypoints, in px — computed by PROJECTING the 3-D matches
        # back into the image, so it stays comparable to its px-tuned reference
        # even though matching itself is 3-D.  This is a direct, same-frame
        # measure of how much the motion model can be trusted: a bad warp
        # (wrong grasp offset, unmodelled rotation, stale warm source) shows up
        # here immediately.  The optim call scales its temporal prior by it, so
        # a prediction that disagrees with the data automatically loses
        # authority to the data.
        nn_px = np.linalg.norm(kpts_2d - warm_dense_px[nn_idxs], axis=1)
        self._last_nn_median = (float(np.median(nn_px))
                                if len(nn_px) else float('nan'))

        ambiguity = 0.0
        maha2     = None          # set in the EKF branch; _match_quality
                                  # tolerates None on frame 0
        if ekf is not None:
            CHI2_GATE_3D = ekf_params.CHI2_MATCH_3D  # χ²(3) 95% — 3-D match gate
                                         # (shared with _CROSSING_CHI2)
            matched_mask, maha2 = ekf.mahalanobis_gate(
                kpts_3d_m, nn_idxs, t_dense,
                chi2_thresh=CHI2_GATE_3D)
            # matched_mask = matched_mask & cap_ok & z_valid

            # ── Graceful EKF degradation ─────────────────────────────────────
            # Score how well the EKF prediction explains this frame's keypoints,
            # then DEMOTE the least-confident warm matches (highest maha²) to the
            # raw adjacency ordering path.  keep_bound shrinks from the full χ²
            # gate (amb=0: keep every gated match) toward 0 (amb→1: demote all →
            # <4 survive → cold keypt_ordering), so the ordering hands authority
            # from the EKF to the raw keypoints continuously as ambiguity rises.
            # ── coverage-aware acceptance fraction ────────────────────────
            # See _COV_BAND_MULT: measured against the keypoints the spline
            # CLAIMS (lateral band + z acceptance window), min-combined with
            # support along the spline's own t-span.  A detangle-clipped
            # spline that tracks its segment reads ~1.0 here even though it
            # matches a minority of ALL keypoints; a stale spline reads low
            # either through unclaimed-but-in-band keypoints (observed stall:
            # lat median ~25 inside a ~38 band) or through sparse t-support.
            _in_cov = (d_lat <= self._COV_BAND_MULT
                       * NN_MAX_DIST_HARD_PX * mm_per_px) & (d_z <= _z_cap)

            def _cov_frac(mask):
                # cov = min(mask.sum() / max(int(_in_cov.sum()), 1), 1.0)
                cov = min(mask.sum() / 1, 1.0)
                mt  = matched_t[mask]
                if mt.size:
                    bins    = np.clip((mt * self._COV_SUPPORT_BINS).astype(int),
                                      0, self._COV_SUPPORT_BINS - 1)
                    support = len(np.unique(bins)) / float(self._COV_SUPPORT_BINS)
                else:
                    support = 0.0
                return float(min(cov, support))

            pass_frac  = _cov_frac(matched_mask)
            passed     = maha2[np.isfinite(maha2) & matched_mask]
            maha_ratio = (float(np.median(passed)) / CHI2_GATE_3D
                          if passed.size else 1.0)
            ambiguity  = self._ambiguity(pass_frac, maha_ratio, floor=ambiguity_floor)

            # keep_bound is FLOORED at _AMB_KEEP_FLOOR of the gate.  Without the
            # floor, ambiguity=1.0 makes it exactly 0.0, and since maha² is
            # strictly positive that demotes EVERY match by construction — a
            # step function, not the continuous hand-off described above.  With
            # the floor, full ambiguity keeps only the most confident matches
            # (maha² ≤ ~1.2); if fewer than 4 clear that bar the frame still
            # falls back to cold ordering, so the hand-off becomes driven by the
            # evidence rather than by the threshold reaching zero.
            keep_bound   = CHI2_GATE_3D * max(1.0 - ambiguity,
                                              self._AMB_KEEP_FLOOR)
            n_pre        = int(matched_mask.sum())
            matched_mask = matched_mask & (maha2 <= keep_bound)
            n_demote     = n_pre - int(matched_mask.sum())

            # Divergence monitor gets the FINAL acceptance fraction (χ² gate
            # AND caps), coverage-normalised like the ambiguity above so a
            # detangle-clipped spline doesn't read as diverged — see
            # note_match_frac.
            ekf.note_match_frac(_cov_frac(matched_mask))

            print(f"NN 3-D matching (χ²(3) Mahalanobis gate + "
                    f"lat<{_lat_cap:.1f} [base {NN_MAX_DIST_BASE_PX:.0f}px] "
                    f"z<{_z_cap:.1f} [3σz, motion={_mot:.2f}] + "
                    f"{int((~z_valid).sum())} depthless): "
                    f"{matched_mask.sum()}/{len(keypoints)} keypoints pass  "
                    f"(lat-rej={int((d_lat >= _lat_cap).sum())} "
                    f"z-rej={int(((d_lat < _lat_cap) & (d_z >= _z_cap)).sum())} | "
                    f"lat: median={float(np.median(d_lat)):.1f} "
                    f"max={d_lat.max():.1f} | nn px median="
                    f"{self._last_nn_median:.1f})")
            print(f"EKF degrade: ambiguity={ambiguity:.2f} "
                    f"(pass_frac={pass_frac:.2f} "
                    # f"[coverage {int(_in_cov.sum())}/{len(keypoints)} kpts] "
                    f"[coverage commented kpts] "
                    f"maha_ratio={maha_ratio:.2f} "
                    f"floor={ambiguity_floor:.2f}) → demoted {n_demote} warm "
                    f"match(es) to raw ordering.")

            if matched_mask.sum() < 4:
                if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
                return None, None, None, None
        else:
            # Frame 0: no filter, so a pure distance test — px-tuned bounds
            # (max_dist param, 10px floor) converted to spline units and
            # applied to the LATERAL residual; depth gets its own 3σ_z cap so
            # z noise can neither reject a good match nor buy lateral slack.
            lat_median     = float(np.median(d_lat))
            dist_threshold = min(max_dist * mm_per_px,
                                 max(lat_median * 3.0, 10.0 * mm_per_px),
                                 _lat_cap)
            matched_mask   = (d_lat < dist_threshold) & (d_z < _z_cap) & z_valid
            print(f"NN 3-D matching: {matched_mask.sum()}/{len(keypoints)} "
                    f"within lat<{dist_threshold:.1f} z<{_z_cap:.1f} "
                    f"({int((~z_valid).sum())} depthless excluded; "
                    f"lat: min={d_lat.min():.1f}  "
                    f"median={lat_median:.1f}  max={d_lat.max():.1f})")

        # ── Per-keypoint match quality (continuous, [MATCH_Q_MIN, 1]) ─────────
        # Scores every keypoint on the SAME evidence the gates judged, plus the
        # arm-ambiguity margin and crossing proximity.  Consumed by optim to
        # soften the constraint boxes of poorly matched keypoints — see
        # _match_quality's docstring.  Stashed aligned to the RETURNED keypoint
        # array on the KF-projection path (_match_q_ordered).
        q_match = self._match_quality(
            d_lat, d_z, _lat_cap, _z_cap, z_valid, maha2,
            nn_dists, matched_t, cand_idx, cand_d, t_dense, d_cross)
        print(f"match quality: median={float(np.median(q_match)):.2f} "
              f"min={float(q_match.min()):.2f} "
              f"low(<0.5)={int((q_match < 0.5).sum())}/{len(q_match)}")

        tw.stop("[warm] projection + NN matching + gate")
        tw.start()
        best_warm_idx = np.clip(
            np.round(matched_t * (len(lin_keypts) - 1)).astype(int),
            0, len(lin_keypts) - 1)

        # ── Drop intersection keypoints from the warm match set ───────────────
        # crossing_kpt_ids was computed above (before matching); the dedicated
        # intersection code (Step C.5) rebuilds these into oriented segments.
        if crossing_kpt_ids:
            valid_ids = [idx for idx in crossing_kpt_ids if idx < len(matched_mask)]
            if valid_ids:
                matched_mask[valid_ids] = False
        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  DEBUG PLOT 1 – NN t-parameter assignments                      ║
        # ║                                                                  ║
        # ║  What to look for:                                               ║
        # ║  • Pink lines (keypoint → nearest spline point) should be short ║
        # ║    and roughly uniform.  Long or criss-crossing lines mean the  ║
        # ║    warm spline is misaligned or the depth scale is wrong.       ║
        # ║  • t-colormap should progress smoothly along the thread.  Any   ║
        # ║    non-monotone jump signals a bad NN assignment.               ║
        # ║  • Red X points are excluded by intersection logic.  Check that ║
        # ║    only the true crossing region is masked out.                 ║
        # ║  • Gray diamonds are distance-rejected (too far from spline).   ║
        # ║    If most keypoints are gray the warm spline is stale / wrong. ║
        # ╚══════════════════════════════════════════════════════════════════╝
        # Stash the DEBUG-1 inputs so the ROS node can re-render this same view
        # into its per-frame debug PNG sequence without needing DEBUG on (these
        # are references/small arrays, so the capture is essentially free).
        self._dbg1 = {
            'mask':             mask,
            'proj_pts':         np.asarray(proj_pts),
            'keypoints':        np.asarray(keypoints),
            'matched_mask':     np.asarray(matched_mask, dtype=bool).copy(),
            'matched_t':        np.asarray(matched_t),
            'best_warm_idx':    np.asarray(best_warm_idx),
            'crossing_kpt_ids': list(crossing_kpt_ids) if crossing_kpt_ids else [],
        }

        if DEBUG:
            fig, axes_dbg = plt.subplots(1, 2, figsize=(16, 7))
            fig.suptitle("DEBUG 1 – NN t-parameter assignments", fontsize=11,
                         fontweight='bold')

            for ax, use_3d in zip(axes_dbg, [False, True]):
                ax.imshow(mask, cmap='gray')

                # Warm spline in 2-D projection
                ax.plot(proj_pts[:, 1], proj_pts[:, 0],
                        c='red', lw=1.2, alpha=0.5, label='warm spline (2D)')
                # Mark t=0 and t=1 ends
                ax.scatter(proj_pts[0,  1], proj_pts[0,  0],
                           c='red', s=80, marker='^', zorder=6, label='t=0')
                ax.scatter(proj_pts[-1, 1], proj_pts[-1, 0],
                           c='red', s=80, marker='v', zorder=6, label='t=1')

                excl_ids  = list(crossing_kpt_ids) if crossing_kpt_ids else []
                dist_fail = np.where(~matched_mask)[0]
                matched   = np.where(matched_mask)[0]

                # Distance-rejected keypoints
                if len(dist_fail):
                    ax.scatter(keypoints[dist_fail, 1], keypoints[dist_fail, 0],
                               c='gray', s=20, marker='D', alpha=0.6,
                               zorder=3, label=f'dist-rejected ({len(dist_fail)})')

                # Intersection-excluded keypoints
                if excl_ids:
                    ax.scatter(keypoints[excl_ids, 1], keypoints[excl_ids, 0],
                               c='red', s=40, marker='x', linewidths=2,
                               zorder=5, label=f'excl. intersection ({len(excl_ids)})')

                # Matched keypoints coloured by t-value
                if len(matched):
                    sc = ax.scatter(keypoints[matched, 1], keypoints[matched, 0],
                                    c=matched_t[matched], cmap='plasma',
                                    vmin=0, vmax=1, s=35, zorder=4,
                                    edgecolors='white', linewidths=0.5,
                                    label=f'matched ({len(matched)})')
                    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label='t-param')

                    # Lines: keypoint → nearest warm spline point
                    for ki in matched:
                        wi  = best_warm_idx[ki]
                        ax.plot([keypoints[ki, 1],   proj_pts[wi, 1]],
                                [keypoints[ki, 0],   proj_pts[wi, 0]],
                                c='pink', lw=0.7, alpha=0.6)

                ax.set_title('2D projection' if not use_3d else
                             '2D projection (zoomed — use 3D subplot separately)')
                ax.legend(fontsize=7, loc='upper right')

            plt.tight_layout()
            plt.savefig("debug1_nn_assignment.png", dpi=150, bbox_inches='tight')
            print("Saved debug1_nn_assignment.png")
            plt.show()

        # ── Sort matched keypoints by t-parameter ─────────────────────────────
        matched_kpt_ids  = np.where(matched_mask)[0]
        matched_t_subset = matched_t[matched_kpt_ids]
        # Primary key: t.  Secondary key: distance to the spline, so when two
        # keypoints share the same t (both nearest to the same spline point, as
        # happens at loops/crowded regions) the one CLOSER to the spline is
        # ordered first — instead of an arbitrary by-index tie-break that looks
        # like an earlier point "stealing" a closer point's match.
        nn_dist_subset   = nn_dists[matched_kpt_ids]
        sort_ord         = np.lexsort((nn_dist_subset, matched_t_subset))
        matched_kpt_ids_dedup   = matched_kpt_ids[sort_ord]
        matched_warm_idxs_dedup = best_warm_idx[matched_kpt_ids_dedup]
        matched_t_dedup         = matched_t_subset[sort_ord]

        warm_segment_kpts     = keypoints[matched_kpt_ids_dedup]
        new_warm_start_keypts = list(matched_t_dedup)
        warm_start_ids        = list(matched_warm_idxs_dedup)

        used = np.zeros(len(keypoints), dtype=bool)
        used[matched_kpt_ids_dedup] = True
        warm_segment_kpts = np.asarray(warm_segment_kpts)
        n_warm            = len(warm_segment_kpts)

        # print(f"NN matched {n_warm} keypoints covering "
        #       f"t=[{matched_t_dedup.min():.3f}, {matched_t_dedup.max():.3f}] "
        #       f"of warm thread.")

        if n_warm < 4:
            print(f"\nNot enough NN matches ({n_warm}), falling back to None.\n")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None, None, None

        def _make_ekf_obs(final_idx):
            """Build ekf_obs = (matched_t, z_3d, ambiguity, denoise_rows,
            obs_conf) for the caller's EKF update, given the FINAL ordered
            keypoint indices.  Shared by the KF-projection early-return and the
            assembly return so the EKF observation logic lives in one place."""
            if ekf is None:
                return None
            z3    = unproject(keypoints[matched_kpt_ids_dedup])
            t_arr = np.asarray(matched_t_dedup, dtype=float)
            ok = np.all(np.isfinite(z3), axis=1) & (z3[:, 2] > 1e-6)
            if ok.any():
                idx_dense = np.clip(
                    np.round(t_arr * (len(t_dense) - 1)).astype(int),
                    0, len(t_dense) - 1)
                gate3d, _ = ekf.mahalanobis_gate(z3[ok], idx_dense[ok], t_dense)
                keep = np.zeros(len(z3), dtype=bool)
                keep[np.nonzero(ok)[0][gate3d]] = True
            else:
                keep = ok
            n_drop = int(len(z3) - keep.sum())
            if n_drop:
                print(f"EKF obs: dropped {n_drop}/{len(z3)} matched keypoints "
                      "with bad/gated 3-D depth.")
            kpt_ids_arr = np.asarray(matched_kpt_ids_dedup, dtype=int)[keep]
            # Per-observation confidence, same keep-filtered indexing as the
            # observations themselves → SplineEKF.update inflates R as
            # R_i = R / max(conf_i, EKF_CONF_R_FLOOR).
            #
            # TWO factors, multiplied:
            #   stereo conf  — was the DEPTH measured well?  (keypt_selection)
            #   match  q     — was this keypoint matched to the right PLACE on
            #                  the spline?  (_match_quality: lateral/z residual,
            #                  maha², arm-ambiguity margin, crossing proximity)
            #
            # q used to reach optim only, as keypt_quality.  In
            # EKF_OUTPUT_MODE='ekf_thread' optim never runs, so the entire
            # crossing-proximity term (MATCH_Q_CROSS_SIGMA_PX / _CROSS_PEN) was
            # computed every frame and discarded — and stereo confidence alone
            # is exactly the wrong weight at a crossing, where two overlapping
            # strands give MORE texture and a sharper SSD minimum.  A keypoint
            # matched onto the wrong arm therefore arrived with the wrong depth
            # AND an un-inflated R, and pulled the filter to that depth.
            # Folding q in means arm-ambiguous and crossing-adjacent keypoints
            # inflate R instead of being trusted equally.
            obs_conf = None
            if keypt_conf is not None and len(keypt_conf) == len(keypoints):
                obs_conf = np.asarray(keypt_conf, dtype=float)[kpt_ids_arr]
            if q_match is not None and len(q_match) == len(keypoints):
                q_sel = np.clip(np.asarray(q_match, dtype=float)[kpt_ids_arr],
                                0.0, 1.0)
                obs_conf = q_sel if obs_conf is None else obs_conf * q_sel
            row_of_id   = {int(g): i for i, g in enumerate(final_idx)}
            denoise_rows = np.array(
                [row_of_id.get(int(g), -1) for g in kpt_ids_arr], dtype=int)
            return (t_arr[keep], z3[keep], ambiguity, denoise_rows, obs_conf)

        # ── KF-projection ordering (skip the segment-split + greedy assembly) ──
        # The keypoints are already sorted by their graph-consistent, crossing-
        # robust t-projection onto the KF spline (matched_kpt_ids_dedup is the
        # lexsort by t).  That t comes from the temporally-stable KF, so this
        # order is stable frame-to-frame — the assembly re-joins/flips (Steps
        # A-F) are exactly what made the order jump completely.
        #
        # The crossing keypoints were struck from matched_mask above, so they
        # are absent from matched_kpt_ids_dedup.  Steps C.5/D/F (which rebuild
        # them into oriented bridge segments) are skipped on this path, so
        # without the re-insertion below they would be dropped from the order
        # entirely.  _crossing_t_by_axis gives each one a t interpolated ALONG
        # ITS OWN crossing arm, which is exactly the disambiguation the raw NN
        # t lacks — then they merge back into the t-sort like any other point.
        if self.KF_PROJECTION_ORDER:
            final_idx = list(matched_kpt_ids_dedup)
            nwsk_full = list(matched_t_dedup)
            recovered = self._crossing_t_by_axis(
                kpts_2d, kd_tree, intersection_segments, crossing_kpt_ids,
                matched_mask, matched_t,
                ekf=ekf, t_dense=t_dense, kpts_3d=kpts_3d_gate, speedy=speedy)
            if recovered:
                ins_ids = np.fromiter(recovered.keys(), dtype=int,
                                      count=len(recovered))
                ins_t   = np.fromiter(recovered.values(), dtype=float,
                                      count=len(recovered))
                all_ids = np.concatenate([np.asarray(matched_kpt_ids_dedup,
                                                     dtype=int), ins_ids])
                all_t   = np.concatenate([np.asarray(matched_t_dedup,
                                                     dtype=float), ins_t])
                # Same tie-break as the matched-keypoint sort above: equal t →
                # the keypoint closer to the spline comes first.
                srt       = np.lexsort((nn_dists[all_ids], all_t))
                final_idx = list(all_ids[srt])
                nwsk_full = list(all_t[srt])
            tw.stop("[warm] KF-projection t-sort ordering")
            new_keypoints = keypoints[final_idx]
            order         = np.arange(len(final_idx))
            # Quality vector re-indexed to the returned (t-sorted) keypoints —
            # crossing re-insertions included (q was computed for ALL
            # keypoints; theirs carries the core-proximity penalty).
            self._match_q_ordered = q_match[np.asarray(final_idx, dtype=int)]
            print(f"KF-projection order: {len(final_idx)} keypoints t-sorted "
                  f"({len(matched_kpt_ids_dedup)} matched + {len(recovered)} "
                  f"crossing re-inserted) "
                  f"(t=[{float(nwsk_full[0]):.3f}, "
                  f"{float(nwsk_full[-1]):.3f}]); assembly skipped.")
            return new_keypoints, order, nwsk_full, _make_ekf_obs(final_idx)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  DEBUG PLOT 2 – t-ordered warm segment                          ║
        # ║                                                                  ║
        # ║  What to look for:                                               ║
        # ║  • Numbered arrows should trace a smooth, plausible path along  ║
        # ║    the thread.  A zigzag or U-turn means t-ordering is locally  ║
        # ║    wrong — likely two keypoints swapping near a high-curvature  ║
        # ║    region or near the intersection.                             ║
        # ║  • The sequence numbers should agree with the left-to-right /   ║
        # ║    top-to-bottom flow seen in the keypt_ordering output.        ║
        # ║  • Large index jumps between spatially adjacent points indicate ║
        # ║    that the warm spline has a loop or reversal in that region.  ║
        # ╚══════════════════════════════════════════════════════════════════╝
        if DEBUG:
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.suptitle(
                f"DEBUG 2 – t-ordered warm segment  ({n_warm} pts, "
                f"t=[{matched_t_dedup.min():.3f}, {matched_t_dedup.max():.3f}])",
                fontsize=11, fontweight='bold')
            ax.imshow(mask, cmap='gray')

            # Warm spline for reference
            ax.plot(proj_pts[:, 1], proj_pts[:, 0],
                    c='red', lw=1, alpha=0.3, label='warm spline')
            ax.scatter(proj_pts[0,  1], proj_pts[0,  0],
                       c='red', s=60, marker='^', zorder=5, label='t=0')
            ax.scatter(proj_pts[-1, 1], proj_pts[-1, 0],
                       c='red', s=60, marker='v', zorder=5, label='t=1')

            n_w = len(warm_segment_kpts)
            cmap_w = plt.cm.cool
            for k in range(n_w - 1):
                color = cmap_w(k / max(n_w - 1, 1))
                p0 = warm_segment_kpts[k,     :2]   # (y, x)
                p1 = warm_segment_kpts[k + 1, :2]
                # Arrow along the segment direction
                ax.annotate('',
                    xy=(p1[1], p1[0]), xytext=(p0[1], p0[0]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

            sc = ax.scatter(warm_segment_kpts[:, 1], warm_segment_kpts[:, 0],
                            c=np.arange(n_w), cmap='cool',
                            s=40, zorder=4, edgecolors='white', linewidths=0.5)
            plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02,
                         label='t-order index')

            # Annotate every 3rd point with its sequence index and t-value
            for k in range(n_w):
                if k % 3 == 0:
                    ax.text(warm_segment_kpts[k, 1] + 3,
                            warm_segment_kpts[k, 0] - 3,
                            f"{k}\n{matched_t_dedup[k]:.2f}",
                            fontsize=6, color='white',
                            bbox=dict(boxstyle='round,pad=0.1',
                                      fc='black', alpha=0.5, ec='none'))

            ax.legend(fontsize=8, loc='upper right')
            plt.tight_layout()
            plt.savefig("debug2_t_ordered_warm_seg.png", dpi=150, bbox_inches='tight')
            print("Saved debug2_t_ordered_warm_seg.png")
            plt.show()

        # ── Cleaning toggles ──────────────────────────────────────────────────
        WARM_SPIKE_REMOVAL = False
        WARM_SEGMENT_SPLIT = False
        # Split a warm segment only where the t-jump between consecutive
        # (t-ordered) matched keypoints exceeds this multiple of the median
        # t-step.  Larger → fewer, longer warm segments.
        WARM_T_GAP_FACTOR  = 5.0
        UNMATCHED_SEGMENTS = True
        POST_DEDUP         = True
        POST_SPIKE_REMOVAL = True

        # ── Step A: optional warm spike removal ───────────────────────────────
        if WARM_SPIKE_REMOVAL:
            wn = len(warm_segment_kpts)
            changed = True
            while changed and wn > 3:
                changed = False
                spike_free = [0]; k = 1
                while k < wn - 1:
                    pp = warm_segment_kpts[spike_free[-1], :2]
                    cp = warm_segment_kpts[k, :2]
                    np_ = warm_segment_kpts[k + 1, :2]
                    if (np.linalg.norm(np_ - pp) < np.linalg.norm(cp - pp) and
                            np.linalg.norm(np_ - pp) < np.linalg.norm(np_ - cp)):
                        changed = True; k += 1; continue
                    spike_free.append(k); k += 1
                spike_free.append(wn - 1)
                if changed:
                    print(f"Warm spike removal: {wn - len(spike_free)} point(s) removed.")
                    matched_kpt_ids_dedup = matched_kpt_ids_dedup[spike_free]
                    warm_segment_kpts     = warm_segment_kpts[spike_free]
                    new_warm_start_keypts = [new_warm_start_keypts[k] for k in spike_free]
                    matched_t_dedup = matched_t_dedup[spike_free]
                    wn = len(warm_segment_kpts)
            used = np.zeros(len(keypoints), dtype=bool)
            used[matched_kpt_ids_dedup] = True
            n_warm = len(warm_segment_kpts)

        # ── Step B: combined keypoint array ───────────────────────────────────
        unmatched_ids  = np.where(~used)[0]
        unmatched_kpts = keypoints[unmatched_ids]
        all_kpts       = np.concatenate([warm_segment_kpts, unmatched_kpts], axis=0)
        warm_idx_set   = set(range(n_warm))

        t_lookup = np.full(len(all_kpts), np.nan)
        for i in range(n_warm):
            t_lookup[i] = matched_t_dedup[i]              # exact, already sorted
        for j, orig_idx in enumerate(unmatched_ids):
            t_lookup[n_warm + j] = matched_t[orig_idx]    # NN approximation

        visited = np.zeros(len(keypoints), dtype=int)
        visited[matched_kpt_ids_dedup] = 1

        # ── Step C: warm sub-segments ─────────────────────────────────────────
        if WARM_SEGMENT_SPLIT:
            # Split where the t-ordered warm keypoints have a large JUMP in t
            # (a genuine gap in warm-spline coverage) — NOT where their original
            # keypoint indices are non-consecutive.  Index order follows cluster
            # creation, not the thread, so with dense keypoints (matched and
            # unmatched interleaved in index space) an index-gap split shatters
            # the segment into singletons.  t is monotone along the thread, so a
            # t-gap is the meaningful discontinuity; real spatial gaps are also
            # handled later by the D.5 spatial-gap split.
            t_ord = np.asarray(matched_t_dedup, dtype=float)
            dt    = np.diff(t_ord)
            pos_dt = dt[dt > 0]
            t_gap_thr = float(np.median(pos_dt) * WARM_T_GAP_FACTOR) if pos_dt.size else np.inf
            warm_segs = []; cur = [0]
            for pos in range(1, len(matched_kpt_ids_dedup)):
                if t_ord[pos] - t_ord[pos - 1] > t_gap_thr:
                    warm_segs.append(cur); cur = [pos]
                else:
                    cur.append(pos)
            warm_segs.append(cur)
        else:
            warm_segs = [list(range(n_warm))]

        print(f"Warm pass: {len(warm_segs)} sub-segment(s) "
              f"from {n_warm} matched keypoints.")

        segments    = list(warm_segs)
        is_warm_seg = [True] * len(warm_segs)

        # ── Step C.5: build intersection prebuilt_segs (Optimized) ────────────
        CORE_EXCLUSION_RADIUS = 20.0
        intersection_radius   = 100.0   # px — max search radius for intersection bridging
        ANGLE_MARGIN_DEG      = 30.0    # Degrees of tolerance for axis alignment
        angle_cos_margin      = np.cos(np.deg2rad(ANGLE_MARGIN_DEG))
        POINTS_PER_SIDE       = 4       # keypoints to collect on each side of a crossing
        prebuilt_segs = []

        def _kpt_to_all_idx(kpt_idx):
            if used[kpt_idx]:
                hits = np.where(matched_kpt_ids_dedup == kpt_idx)[0]
                return int(hits[0]) if len(hits) else None
            else:
                hits = np.where(unmatched_ids == kpt_idx)[0]
                return (n_warm + int(hits[0])) if len(hits) else None

        def _build_axis_seg(centroid, axis_vec, kpt_indices):
            mapped = []
            seen   = set()
            for kpt_idx in kpt_indices:
                all_idx = _kpt_to_all_idx(int(kpt_idx))
                if all_idx is None or all_idx in seen:
                    continue
                seen.add(all_idx)
                dist_to_center = np.linalg.norm(all_kpts[all_idx, :2] - centroid)
                if dist_to_center <= CORE_EXCLUSION_RADIUS:
                    continue
                mapped.append(all_idx)
            if len(mapped) < 2:
                return []
            proj   = (all_kpts[mapped, :2] - centroid) @ axis_vec
            mapped = [mapped[k] for k in np.argsort(proj)]
            return mapped

        # We reuse `kd_tree` directly since it was already initialized earlier in warm_ordering!
        
        if intersection_segments is not None and len(intersection_segments) > 0:
            for crossing in intersection_segments:
                for axis_seg in crossing:
                    axis_vec = np.array(axis_seg['axis_vec'])
                    centroid = np.array(axis_seg['centroid'])
                    
                    candidate_kpt_ids = []
                    n_positive = 0          # keypoints found on the +axis side
                    n_negative = 0          # keypoints found on the -axis side

                    # 1. Fast O(log N) Spatial Query using the existing KD-Tree
                    # k=40 ensures we grab enough neighbors to bridge the gap
                    # It returns arrays already sorted by distance!
                    dists, indices = kd_tree.query(centroid, k=40, distance_upper_bound=intersection_radius)

                    for dist, idx in zip(dists, indices):
                        if dist == float('inf'):
                            break

                        # Handle core-zone points: Lock them out but don't assign them to an arm
                        if dist <= CORE_EXCLUSION_RADIUS:
                            visited[idx] = 1 # Mark as visited to lock out standard segment growers
                            continue

                        # 2. Compute vector math ONLY on the local neighborhood
                        vec = kpts_2d[idx] - centroid
                        v_norm = vec / dist
                        dot_prod = np.dot(v_norm, axis_vec)

                        # Collect up to POINTS_PER_SIDE nearest keypoints on each
                        # side of the crossing (a longer, better-conditioned arm
                        # than a single point per side).
                        if dot_prod > angle_cos_margin and n_positive < POINTS_PER_SIDE:
                            candidate_kpt_ids.append(idx)
                            n_positive += 1

                        elif dot_prod < -angle_cos_margin and n_negative < POINTS_PER_SIDE:
                            candidate_kpt_ids.append(idx)
                            n_negative += 1

                        # 3. Stop once both sides have enough points to bridge the crossing
                        if n_positive >= POINTS_PER_SIDE and n_negative >= POINTS_PER_SIDE:
                            break
                            
                    # Lock out the selected candidate points globally
                    for c_idx in candidate_kpt_ids:
                        visited[c_idx] = 1
                        
                    seg = _build_axis_seg(centroid, axis_vec, candidate_kpt_ids)
                    if len(seg) >= 2:
                        prebuilt_segs.append(seg)
                        segments.append(seg)
                        is_warm_seg.append(False)
                        
            print(f"warm_ordering: {len(prebuilt_segs)} intersection prebuilt "
                  f"segment(s) built (CORE_EXCLUSION_RADIUS={CORE_EXCLUSION_RADIUS}px)")
                  
        elif _internal_crossing_axes:
            for centroid, axis_vec, global_ids in _internal_crossing_axes:
                seg = _build_axis_seg(centroid, axis_vec, global_ids)
                if len(seg) >= 2:
                    prebuilt_segs.append(seg)
                    segments.append(seg)
                    is_warm_seg.append(False)
            print(f"warm_ordering: {len(prebuilt_segs)} intersection prebuilt "
                  f"segment(s) built from internal RANSAC axes "
                  f"(CORE_EXCLUSION_RADIUS={CORE_EXCLUSION_RADIUS}px)")

        # Ambiguity widens the raw-segment join thresholds (Step D / D.5): when
        # the EKF is untrustworthy the adjacency graph should bridge more freely
        # so the demoted keypoints form coherent raw segments instead of shards.
        amb_widen = 1.0 + self._AMB_WIDEN_GAIN * ambiguity

        # ── Step D: grow unmatched segments via adjacency ─────────────────────
        if UNMATCHED_SEGMENTS and adjacents is not None and len(unmatched_ids) > 0:
            warm_dists = [
                np.linalg.norm((warm_segment_kpts[p] - warm_segment_kpts[p-1])[:2])
                for p in range(1, len(matched_kpt_ids_dedup))
                if matched_kpt_ids_dedup[p] - matched_kpt_ids_dedup[p-1] == 1
            ]
            split_thr = np.median(warm_dists) * 3.0 * amb_widen if warm_dists else np.inf
            n_kpts = len(keypoints); n_adj = len(adjacents)

            def safe_nb(node):
                if int(node) >= n_adj: return []
                return [int(nb) for nb in adjacents[int(node)] if int(nb) < n_kpts]

            outer_front = [
                int(uid) for uid in unmatched_ids
                if sum(1 for nb in safe_nb(uid) if visited[nb] == 0) <= 1
            ]
            while True:
                source = None
                while outer_front:
                    c_id = outer_front.pop()
                    if visited[c_id] == 1: continue
                    if sum(1 for nb in safe_nb(c_id) if visited[nb] == 0) <= 1:
                        source = c_id; break
                if source is None:
                    rem = [int(u) for u in unmatched_ids if visited[u] == 0]
                    if not rem: break
                    source = rem[0]
                frontier = [source]; visited[source] = 1; seg_orig = []
                while frontier:
                    curr = frontier.pop(); seg_orig.append(curr)
                    mn_dist, mn_nb = np.inf, None
                    for nb in safe_nb(curr):
                        if visited[nb] == 0:
                            d = np.linalg.norm((keypoints[nb] - keypoints[curr])[:2])
                            if d < mn_dist: mn_dist, mn_nb = d, nb
                    if mn_nb is not None:
                        if mn_dist <= split_thr:
                            visited[mn_nb] = 1; frontier.append(mn_nb)
                        else:
                            outer_front.append(mn_nb)
                if seg_orig:
                    seg_new = [n_warm + int(np.where(unmatched_ids == k)[0][0])
                               for k in seg_orig]
                    if len(seg_new) > 2:
                        segments.append(seg_new); is_warm_seg.append(False)
                    else:
                        print(f"Ignoring unmatched segment of {len(seg_new)} keypoint(s).")

        # ── Step D.5: split large spatial gaps ───────────────────────────────
        all_dists = [
            np.linalg.norm((all_kpts[seg[k]] - all_kpts[seg[k-1]])[:2])
            for seg in segments for k in range(1, len(seg))
        ]
        spatial_gap_thr = (min(np.median(all_dists) * 15.0 * amb_widen, max_dist) #og 3.0 max dist 40
                           if all_dists else max_dist)

        split_segs, split_warm = [], []
        for seg, warm_flag in zip(segments, is_warm_seg):
            cur = [seg[0]]
            for k in range(1, len(seg)):
                d = np.linalg.norm((all_kpts[seg[k]] - all_kpts[seg[k-1]])[:2])
                if d > spatial_gap_thr:
                    split_segs.append(cur); split_warm.append(warm_flag)
                    cur = [seg[k]]
                else:
                    cur.append(seg[k])
            if cur: split_segs.append(cur); split_warm.append(warm_flag)
        segments    = split_segs
        is_warm_seg = split_warm
        print(f"Spatial gap split (thresh={spatial_gap_thr:.1f}px) → "
              f"{len(segments)} segments.")
        
        # ══════════════════════════════════════════════════════════════════════
        # Step D.6 — split warm segments at non-warm t-band boundaries
        # (runs after D.5 so every segment that will ever exist is present)
        # ══════════════════════════════════════════════════════════════════════
        #
        # After D.5 the segment list is final.  Non-warm segments now include
        # intersection prebuilt segs from C.5 AND any unmatched segs from D.
        # A warm segment spanning t∈[0, 0.8] would sort before a non-warm
        # segment at t∈[0.44, 0.56] in Step F, double-covering the middle.
        #
        # For each non-warm segment: find its t-band [t_lo, t_hi] from t_lookup.
        # For each warm segment: drop keypoints inside any such band, then
        # re-split the remainder into contiguous sub-segments.
        # ─────────────────────────────────────────────────────────────────────
 
        # assign each warm keypoint a "region index" that increments at
        # every band boundary:
        #
        #   region 0  →  before band 0         (keep, even)
        #   region 1  →  inside band 0         (drop, odd)
        #   region 2  →  between band 0 and 1  (keep, even)
        #   region 3  →  inside band 1         (drop, odd)
        #   region 4  →  after band 1          (keep, even)
        #   ...
        #
        # A segment is split whenever adjacent points have different region
        # indices.  "Keep" (even) contiguous runs of ≥ 2 points become
        # sub-segments; "drop" (odd) runs are discarded.  Crucially, a
        # transition from region 0 to region 2 (no warm points in the band)
        # still produces a split with two kept sub-segments.
        # ─────────────────────────────────────────────────────────────────────
 
        # A non-warm segment's drop-band is the t-range it already covers.  But
        # t_lookup for unmatched/intersection keypoints is the ambiguous
        # nearest-t projection onto the warm spline: near a loop/crossing two
        # physically distant points project to very different t, so the raw
        # (min,max) envelope can span most of [0,1] and swallow every warm
        # keypoint.  A non-warm segment whose t-values are spread wider than
        # this cap is exactly that ambiguous case — skip it rather than let it
        # define a giant drop-band.
        _MAX_NONWARM_BAND = 0.25
        non_warm_t_bands = []
        for seg, seg_is_warm in zip(segments, is_warm_seg):
            if seg_is_warm:
                continue
            t_vals = t_lookup[seg]
            valid  = t_vals[~np.isnan(t_vals)]
            if len(valid) == 0:
                continue
            lo, hi = float(valid.min()), float(valid.max())
            if hi - lo > _MAX_NONWARM_BAND:
                print(f"  non-warm seg len={len(seg)}: t-span "
                      f"[{lo:.3f},{hi:.3f}] > {_MAX_NONWARM_BAND} "
                      "(ambiguous NN-t near intersection); not forming a "
                      "drop-band.")
                continue
            non_warm_t_bands.append((lo, hi))
 
        if non_warm_t_bands:
            sorted_bands = sorted(non_warm_t_bands)   # ascending t_lo order
            if not speedy:
                print(f"Step D.6: {len(sorted_bands)} non-warm t-band(s): "
                  + "  ".join(f"[{lo:.3f},{hi:.3f}]"
                              for lo, hi in sorted_bands))
 
            n_warm_in  = sum(is_warm_seg)
            new_segments, new_is_warm = [], []
 
            for seg, seg_is_warm in zip(segments, is_warm_seg):
                if not seg_is_warm:
                    new_segments.append(seg); new_is_warm.append(False)
                    continue
 
                seg_t = t_lookup[seg]   # (len(seg),) — NaN for any missing
 
                # ── assign region index per point ─────────────────────────
                # Each t-boundary increments the region counter by 1.
                # Bands contribute two boundaries: t_lo (enter) and t_hi (exit).
                regions = np.zeros(len(seg), dtype=int)
                for t_lo, t_hi in sorted_bands:
                    valid    = ~np.isnan(seg_t)
                    regions[valid & (seg_t >= t_lo)] += 1  # entered band or beyond
                    regions[valid & (seg_t >  t_hi)] += 1  # exited band
 
                # NaN points inherit the region of the previous valid point
                # (handles any gaps without creating spurious splits)
                last_region = 0
                for k in range(len(regions)):
                    if np.isnan(seg_t[k]):
                        regions[k] = last_region
                    else:
                        last_region = regions[k]
 
                # ── collect contiguous runs of even (keep) regions ────────
                run, run_region, run_segs = [], regions[0], []
                for all_idx, reg in zip(seg, regions):
                    if reg == run_region:
                        run.append(all_idx)
                    else:
                        if run_region % 2 == 0 and len(run) >= 2:
                            run_segs.append(list(run))
                        run        = [all_idx]
                        run_region = reg
                # flush final run
                if run_region % 2 == 0 and len(run) >= 2:
                    run_segs.append(list(run))
 
                t_finite = seg_t[~np.isnan(seg_t)]
                t_rng    = (f"t=[{t_finite.min():.3f},"
                            f"{t_finite.max():.3f}]") if len(t_finite) else "t=?"
 
                if not run_segs:
                    print(f"  warm seg len={len(seg)} {t_rng}: "
                          f"fully inside non-warm band(s), dropped.")
                elif len(run_segs) == 1 and run_segs[0] == list(seg):
                    new_segments.append(seg); new_is_warm.append(True)
                else:
                    new_segments.extend(run_segs)
                    new_is_warm.extend([True] * len(run_segs))
                    print(f"  warm seg len={len(seg)} {t_rng}: "
                          f"split → {len(run_segs)} sub-seg(s) "
                          + " ".join(f"len={len(r)}" for r in run_segs))
 
            segments    = new_segments
            is_warm_seg = new_is_warm
            print(f"Step D.6 done: {n_warm_in} warm → {sum(new_is_warm)} | "
                  f"{len(segments)} total segments.")
        else:
            print("Step D.6: no non-warm segments, nothing to split.")
             
        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  DEBUG PLOT 3 – segments entering two-phase assembly            ║
        # ║                                                                  ║
        # ║  What to look for:                                               ║
        # ║  • Each warm segment (blue) should show a sensible local order  ║
        # ║    (▶ head → ■ tail arrow should point along the thread).       ║
        # ║  • Each intersection prebuilt seg (cyan) should span exactly    ║
        # ║    one arm of the crossing, with the axis arrow aligned with    ║
        # ║    the physical thread direction through the crossing.          ║
        # ║  • Unmatched segs (orange) should be in regions the warm spline ║
        # ║    didn't cover (e.g. newly visible thread portions).           ║
        # ║  • If a warm seg has its head/tail reversed relative to the     ║
        # ║    keypt_ordering output, the warm spline t=0 end is flipped.  ║
        # ╚══════════════════════════════════════════════════════════════════╝
        if DEBUG:
            fig, ax = plt.subplots(figsize=(11, 9))
            fig.suptitle(
                f"DEBUG 3 – segments entering assembly  "
                f"(W={len(warm_segs)}  I={len(prebuilt_segs)}  "
                f"U={len(segments)-len(warm_segs)-len(prebuilt_segs)})",
                fontsize=11, fontweight='bold')
            ax.imshow(mask, cmap='gray')

            cmap20 = plt.cm.get_cmap('tab20')
            legend_handles = []

            for si, (seg, is_warm) in enumerate(zip(segments, is_warm_seg)):
                pts   = all_kpts[seg]
                n_pts = len(pts)

                if is_warm:
                    base_color = np.array([0.2, 0.5, 1.0])   # blue family
                    label_char = 'W'
                else:
                    base_color = np.array([0.0, 0.85, 0.85]) # cyan
                    label_char = 'I'

                # Draw the segment with a fading alpha to show direction
                for k in range(n_pts - 1):
                    alpha = 0.35 + 0.65 * k / max(n_pts - 1, 1)
                    ax.plot([pts[k,1], pts[k+1,1]], [pts[k,0], pts[k+1,0]],
                            color=base_color, alpha=alpha, lw=2.0)

                # Scatter points coloured by within-segment order
                sc_c = plt.cm.Greys(np.linspace(0.3, 1.0, n_pts))
                ax.scatter(pts[:, 1], pts[:, 0], color=sc_c, s=18,
                           zorder=3, edgecolors=base_color, linewidths=1.0)

                # ▶ head and ■ tail markers
                ax.scatter(pts[0, 1],  pts[0, 0],
                           marker='>', color=base_color, s=60, zorder=5)
                ax.scatter(pts[-1, 1], pts[-1, 0],
                           marker='s', color=base_color, s=60, zorder=5)

                # Label at midpoint
                mid = n_pts // 2
                ax.text(pts[mid, 1] + 2, pts[mid, 0] - 2,
                        f"{label_char}{si}",
                        fontsize=8, fontweight='bold',
                        color=base_color,
                        bbox=dict(boxstyle='round,pad=0.15',
                                  fc='black', alpha=0.55, ec='none'))

                # For intersection segs, draw the axis arrow
                if not is_warm and len(pts) >= 2:
                    p0 = pts[0,  :2]
                    p1 = pts[-1, :2]
                    ax.annotate('',
                        xy=(p1[1], p1[0]), xytext=(p0[1], p0[0]),
                        arrowprops=dict(arrowstyle='->', color='yellow',
                                        lw=2.5))

            legend_handles = [
                plt.Line2D([0],[0], color=[0.2,0.5,1.0], lw=2,
                           label='Warm segments (W)'),
                plt.Line2D([0],[0], color=[0.0,0.85,0.85], lw=2,
                           label='Intersection prebuilt (I)'),
                plt.Line2D([0],[0], marker='>', color='gray', linestyle='None',
                           markersize=8, label='Segment head (▶)'),
                plt.Line2D([0],[0], marker='s', color='gray', linestyle='None',
                           markersize=8, label='Segment tail (■)'),
                plt.Line2D([0],[0], color='yellow', lw=2,
                           label='Intersection axis direction'),
            ]
            ax.legend(handles=legend_handles, fontsize=8, loc='upper right')

            # Also annotate each keypoint with its original index for cross-referencing
            for i, kpt in enumerate(keypoints):
                ax.text(kpt[1] + 1, kpt[0] + 1, str(i),
                        fontsize=5, color='white', alpha=0.7)

            plt.tight_layout()
            plt.savefig("debug3_segments_pre_assembly.png", dpi=150,
                        bbox_inches='tight')
            print("Saved debug3_segments_pre_assembly.png")
            plt.show()

# ── Step F: T-ordered segment assembly ────────────────────────────────
        #
        # The warm start t-values are the authority for global segment ordering.
        # Every keypoint in all_kpts has a meaningful t-value:
        #
        #   • Warm-matched  (all_kpts idx < n_warm):
        #         exact t from matched_t_dedup — perfectly reliable.
        #
        #   • Intersection / unmatched (all_kpts idx ≥ n_warm):
        #         NN-projected t from matched_t[orig_idx].  These keypoints
        #         were excluded from the warm gate, but matched_t was computed
        #         for ALL keypoints before any exclusion, so the value still
        #         gives a valid spline-position estimate for ordering purposes.
        #
        # Algorithm
        # ─────────
        #   F.1  Build t_lookup[all_idx] for every entry in all_kpts.
        #   F.2  For each segment compute t_mean (sort key) and t_head / t_tail
        #        (orientation key) using the first and last valid t in the seg.
        #   F.3  Sort segments by t_mean (ascending).
        #   F.4  Orient each segment so t increases head → tail (flip if not).
        #   F.5  Concatenate.  At each boundary, if the tail of the previous
        #        segment and the head of the next are within SNAP_DIST, accept
        #        the join directly.  If the tail end is actually closer, flip
        #        the incoming segment — this handles intersection inner-endpoints
        #        that sit very close to an adjacent warm segment mouth and whose
        #        t-value estimate is slightly noisy.
        # ─────────────────────────────────────────────────────────────────────
        tw.stop("[warm] segment building (steps A-D)")
        tw.start()
        SNAP_DIST = 15.0   # px — below this, spatial adjacency overrides t-orientation
        # Minimum t-span for a segment's t-orientation to be "trusted": at or
        # above this, the spatial-proximity flip (F.5) may NOT reverse it.
        # Short intersection bridges fall below and can still be flipped.
        WARM_T_ORIENT_MIN = 0.10
        # Assembly join cap (px): during F.5 concatenation, a segment whose
        # connecting end sits farther than this from the running thread tail is
        # SKIPPED rather than joined — otherwise two t-adjacent but spatially
        # distant segments would be bridged by an implausible jump in the
        # ordered thread.  Raise to allow longer bridges; lower to be stricter.
        ASSEMBLY_MAX_JOIN_PX = 120.0

        # ── F.2: per-segment t statistics ─────────────────────────────────────
        def _seg_t_info(seg):
            """
            Returns (t_mean, t_head, t_tail) for one segment.
            t_head / t_tail use the FIRST and LAST valid t-value in the
            segment (not necessarily the first and last position index) so
            that a few NaN-padded endpoints do not mislead the orientation.
            """
            t_vals  = t_lookup[seg]
            valid   = np.where(~np.isnan(t_vals))[0]
            if len(valid) == 0:
                return np.nan, np.nan, np.nan
            return (float(t_vals[valid].mean()),
                    float(t_vals[valid[0]]),
                    float(t_vals[valid[-1]]))

        seg_info = [_seg_t_info(s) for s in segments]

        # ── F.3: sort segments by t_mean ──────────────────────────────────────
        # Segments with no t-values (NaN) sort to the end.
        sort_key  = [ti[0] if not np.isnan(ti[0]) else 1e9 for ti in seg_info]
        sort_order = np.argsort(sort_key, kind='stable')
        if not speedy:
            print("Segment t-order (idx → t_mean  t_head  t_tail  warm?):")
            for rank, si in enumerate(sort_order):
                tm, th, tt = seg_info[si]
                print(f"  rank {rank:2d}  seg {si:2d}  "
                    f"t_mean={tm:.3f}  t_head={th:.3f}  t_tail={tt:.3f}  "
                    f"warm={is_warm_seg[si]}  len={len(segments[si])}")

        # ── F.4: orient each segment so t increases head → tail ───────────────
        # t_strength = |t_tail - t_head| measures how trustworthy the segment's
        # own t-orientation is.  A long warm segment spans a wide t-range
        # (strong); a short intersection segment barely spans any t (weak,
        # ambiguous).  Used in F.5 to decide whether the spatial-proximity flip
        # is allowed to override the t-orientation.
        oriented, t_strength = [], []
        for si in sort_order:
            seg = list(segments[si])
            _,  t_head, t_tail = seg_info[si]
            if (not np.isnan(t_head) and not np.isnan(t_tail)):
                if t_head > t_tail:
                    seg = seg[::-1]
                t_strength.append(abs(t_tail - t_head))
            else:
                t_strength.append(0.0)
            oriented.append(seg)

        # ── F.5: concatenate with boundary snap ───────────────────────────────
        final_idx = list(oriented[0])
        for k in range(1, len(oriented)):
            seg      = oriented[k]
            tail_pt  = all_kpts[final_idx[-1], :2]
            head_pt  = all_kpts[seg[0],  :2]
            tail2_pt = all_kpts[seg[-1], :2]
            d_head   = np.linalg.norm(tail_pt - head_pt)
            d_tail   = np.linalg.norm(tail_pt - tail2_pt)

            if d_head <= SNAP_DIST:
                # Endpoint is right next to us — connect directly without flip
                oriented_seg, join_gap = seg, d_head
            elif d_tail < d_head and t_strength[k] < WARM_T_ORIENT_MIN:
                # Tail end is closer AND this segment's own t-direction is weak
                # (short/ambiguous, e.g. an intersection bridge) — trust spatial
                # continuity and flip.  A segment with a strong t-gradient keeps
                # its t-orientation: at a crossing both its ends sit near the
                # core, so endpoint distance is a bad tiebreaker and would
                # reverse a whole warm run backward through the spline.
                if not speedy:
                    print(f"  boundary flip at rank {k}: "
                        f"d_head={d_head:.1f}  d_tail={d_tail:.1f}  "
                        f"t_strength={t_strength[k]:.3f}")
                oriented_seg, join_gap = seg[::-1], d_tail
            else:
                oriented_seg, join_gap = seg, d_head

            # Assembly join cap: skip a segment whose connecting end sits farther
            # than ASSEMBLY_MAX_JOIN_PX from the running tail rather than bridge
            # an implausible spatial jump in the ordered thread.
            if join_gap > ASSEMBLY_MAX_JOIN_PX:
                if not speedy:
                    print(f"  assembly: skipping seg rank {k} — join gap "
                          f"{join_gap:.1f}px > {ASSEMBLY_MAX_JOIN_PX:.0f}px cap.")
                continue
            final_idx.extend(oriented_seg)

        new_keypoints = all_kpts[final_idx]
        n = len(new_keypoints)
        if n < 10:
            print(f"less than 10 keypoints, num of keypoints={n}, enabling debug mode")
            speedy = False
            DEBUG = True
        # ── Step G: post-join deduplication ───────────────────────────────────
        if POST_DEDUP:
            keep = [0]
            for k in range(1, n):
                if np.linalg.norm(new_keypoints[k,:2] - new_keypoints[keep[-1],:2]) > 1.0:
                    keep.append(k)
            if len(keep) < n:
                print(f"Dedup: {n-len(keep)} near-duplicate keypoints removed.")
                final_idx     = [final_idx[k] for k in keep]
                new_keypoints = all_kpts[final_idx]
                n = len(new_keypoints)
            if n < 10:
                print(f"less than 10 keypoints, num of keypoints={n}, enabling debug mode")
                speedy = False
                DEBUG = True

        # ── Step H: cosine-angle spike removal ────────────────────────────────
        if POST_SPIKE_REMOVAL:
            SHARP_TURN_THRESH = 0.0
            changed = True
            while changed and n > 2:
                changed = False
                cleaned = [final_idx[0]]
                k = 1
                while k < n - 1:
                    pp  = all_kpts[cleaned[-1],    :2]
                    cp  = all_kpts[final_idx[k],   :2]
                    np_ = all_kpts[final_idx[k+1], :2]
                    v_in  = cp  - pp
                    v_out = np_ - cp
                    norm_in  = np.linalg.norm(v_in)
                    norm_out = np.linalg.norm(v_out)
                    if norm_in > 1e-5 and norm_out > 1e-5:
                        cos_angle = np.dot(v_in, v_out) / (norm_in * norm_out)
                        if cos_angle < SHARP_TURN_THRESH:
                            changed = True; k += 1; continue
                    cleaned.append(final_idx[k])
                    k += 1
                cleaned.append(final_idx[-1])
                if changed:
                    if not speedy:
                        print(f"Post-spike (cosine): {n - len(cleaned)} point(s) removed.")
                    final_idx     = cleaned
                    new_keypoints = all_kpts[final_idx]
                    n = len(new_keypoints)
            if n < 10:
                print(f"less than 10 keypoints, num of keypoints={n}, enabling debug mode")
                speedy = False
                DEBUG = True

        # ══════════════════════════════════════════════════════════════════════
        # Step I — slowly grow the thread ENDPOINTS into unmatched keypoints
        # ══════════════════════════════════════════════════════════════════════
        # When the thread moves, freshly-exposed thread at either END projects
        # far from the (previous-frame) warm spline, fails the NN gate, and would
        # otherwise be dropped.  Here we walk OUTWARD from each end of the
        # assembled order along the mask-adjacency graph, appending unused
        # keypoints that continue the thread's local direction.  Growth is capped
        # per frame ("slow"): optim refits the warm spline to include the new
        # points, so the next frame matches them normally and the end can grow
        # further — the thread recovers its moved ends over a few frames instead
        # of grabbing (possibly cluttered) unmatched regions all at once.
        GROW_ENDPOINTS   = True
        MAX_GROW_PER_END = 3      # keypoints added per end per frame — the "slow" cap
        GROW_GAP_MAX     = spatial_gap_thr   # px ceiling on a growth step (reuse
                                             # the assembly spatial-gap threshold)
        GROW_COS_MIN     = 0.3    # min cos(step, local end-direction): keeps growth
                                  # running ALONG the thread, not sideways into clutter
        if (GROW_ENDPOINTS and adjacents is not None and len(final_idx) >= 2):
            n_kpts_g = len(keypoints)
            n_adj_g  = len(adjacents)
            # all_kpts index → original keypoint index, and its inverse.  Every
            # keypoint is in all_kpts exactly once (matched ∪ unmatched), so the
            # inverse is fully defined.
            all_to_orig = np.concatenate(
                [matched_kpt_ids_dedup, unmatched_ids]).astype(int)
            orig_to_all = np.full(n_kpts_g, -1, dtype=int)
            orig_to_all[all_to_orig] = np.arange(len(all_to_orig))

            used_orig = set(int(all_to_orig[i]) for i in final_idx)

            def _grow_nbrs(o):
                if o >= n_adj_g:
                    return []
                return [int(u) for u in adjacents[o]
                        if int(u) < n_kpts_g
                        and int(u) not in used_orig
                        and int(u) not in crossing_kpt_ids]

            def _grow_from(end_all, prev_all):
                """Walk outward from keypoint `end_all`; `prev_all` (one step
                inward) sets the initial outward direction.  Returns the grown
                original-keypoint indices, nearest-end first."""
                grown = []
                curr  = int(all_to_orig[end_all])
                tangent = (keypoints[curr, :2]
                           - keypoints[int(all_to_orig[prev_all]), :2]).astype(float)
                tn = float(np.linalg.norm(tangent))
                tangent = tangent / tn if tn > 1e-6 else tangent
                for _ in range(MAX_GROW_PER_END):
                    best, best_gap = None, np.inf
                    for nb in _grow_nbrs(curr):
                        step = (keypoints[nb, :2] - keypoints[curr, :2]).astype(float)
                        gap  = float(np.linalg.norm(step))
                        if gap > GROW_GAP_MAX or gap < 1e-6:
                            continue
                        cos = float(np.dot(step, tangent) / gap) if tn > 1e-6 else 1.0
                        if cos < GROW_COS_MIN:
                            continue
                        if gap < best_gap:            # nearest aligned neighbour
                            best, best_gap = nb, gap
                    if best is None:
                        break
                    step    = (keypoints[best, :2] - keypoints[curr, :2]).astype(float)
                    tangent = step / best_gap         # follow the thread's curve
                    tn      = best_gap
                    grown.append(best)
                    used_orig.add(best)
                    curr = best
                return grown

            head_grow = _grow_from(final_idx[0],  final_idx[1])
            tail_grow = _grow_from(final_idx[-1], final_idx[-2])

            if head_grow or tail_grow:
                # head growth is prepended farthest-first (it becomes the new start)
                pre  = [int(orig_to_all[o]) for o in reversed(head_grow)]
                post = [int(orig_to_all[o]) for o in tail_grow]
                final_idx     = pre + list(final_idx) + post
                new_keypoints = all_kpts[final_idx]
                n = len(new_keypoints)
                if not speedy:
                    print(f"Step I endpoint growth: +{len(head_grow)} head / "
                          f"+{len(tail_grow)} tail → {n} keypoints.")

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  DEBUG PLOT 4 – final ordering (post-assembly, post-spike)      ║
        # ║                                                                  ║
        # ║  What to look for:                                               ║
        # ║  • This should match the keypt_ordering scatter plot exactly    ║
        # ║    in terms of which physical position gets which index.        ║
        # ║  • The "hot" colourmap runs black (0) → red → yellow (N-1).    ║
        # ║    If the colouring is inverted vs keypt_ordering, the entire   ║
        # ║    warm spline is running in the wrong direction (flip t).      ║
        # ║  • Any island of wrong-coloured points surrounded by correct    ║
        # ║    ones marks the exact segment boundary where a join went bad. ║
        # ║  • The 2D vs 3D panels should agree — a 3D crossing that looks ║
        # ║    correct in 2D but wrong in 3D means z-ordering is the issue.║
        # ╚══════════════════════════════════════════════════════════════════╝
        if DEBUG:
            fig = plt.figure(figsize=(16, 7))
            fig.suptitle(
                f"DEBUG 4 – final ordering  ({n} pts, "
                f"{n_warm} warm + {n-n_warm} unmatched)",
                fontsize=11, fontweight='bold')

            # ── Left: 2D numbered scatter (mirrors keypt_ordering final plot) ──
            ax_l = fig.add_subplot(1, 3, 1)
            ax_l.imshow(mask, cmap='gray')
            sc = ax_l.scatter(new_keypoints[:, 1], new_keypoints[:, 0],
                              c=np.arange(n), cmap='hot', s=30, zorder=3)
            plt.colorbar(sc, ax=ax_l, fraction=0.03, pad=0.02, label='order index')
            for k in range(n - 1):
                p0 = new_keypoints[k,   :2]
                p1 = new_keypoints[k+1, :2]
                ax_l.annotate('',
                    xy=(p1[1], p1[0]), xytext=(p0[1], p0[0]),
                    arrowprops=dict(arrowstyle='->', color='white',
                                   lw=0.8, alpha=0.5))
            # Label every 3rd point
            for k in range(n):
                if k % 3 == 0:
                    ax_l.text(new_keypoints[k, 1] + 2, new_keypoints[k, 0] - 2,
                              str(k), fontsize=6, color='cyan')
            ax_l.set_title('Final order (hot cmap = 0→N-1)')
            ax_l.scatter(new_keypoints[0,  1], new_keypoints[0,  0],
                         c='lime',  s=80, marker='^', zorder=6, label='start (0)')
            ax_l.scatter(new_keypoints[-1, 1], new_keypoints[-1, 0],
                         c='white', s=80, marker='v', zorder=6, label=f'end ({n-1})')
            ax_l.legend(fontsize=7)

            # ── Middle: warm vs unmatched colour-coded ────────────────────────
            ax_m = fig.add_subplot(1, 3, 2)
            ax_m.imshow(mask, cmap='gray')
            warm_in_final    = [i for i, idx in enumerate(final_idx)
                                if idx in warm_idx_set]
            unmatched_in_final = [i for i, idx in enumerate(final_idx)
                                   if idx not in warm_idx_set]
            if warm_in_final:
                ax_m.scatter(new_keypoints[warm_in_final, 1],
                             new_keypoints[warm_in_final, 0],
                             c=[warm_in_final/np.max(warm_in_final)], cmap='Blues',
                             s=30, zorder=4, label='warm-matched')
            if unmatched_in_final:
                ax_m.scatter(new_keypoints[unmatched_in_final, 1],
                             new_keypoints[unmatched_in_final, 0],
                             c='orange', s=30, zorder=4,
                             marker='D', label='unmatched')
            ax_m.plot(proj_pts[:, 1], proj_pts[:, 0],
                      c='red', lw=1, alpha=0.4, label='warm spline')
            ax_m.legend(fontsize=7); ax_m.set_title('Warm vs unmatched origin')

            # ── Right: 3D camera-space ordering ──────────────────────────────
            ax_r = fig.add_subplot(1, 3, 3, projection='3d')
            nk_3d = unproject(new_keypoints)
            ax_r.scatter(nk_3d[:, 0], nk_3d[:, 1], nk_3d[:, 2],
                         c=np.arange(n), cmap='hot', s=15, zorder=3)
            ax_r.plot(nk_3d[:, 0], nk_3d[:, 1], nk_3d[:, 2],
                      c='steelblue', lw=0.8, alpha=0.5)
            wp = warm_keypoints
            ax_r.plot(wp[:, 0], wp[:, 1], wp[:, 2],
                      c='red', lw=1, alpha=0.4, label='warm spline (3D)')
            ct = curr_T[:3, 3]
            ax_r.scatter(ct[0], ct[1], ct[2],
                         c='red', s=60, marker='x', zorder=5, label='curr_T')
            ax_r.set_xlabel('X'); ax_r.set_ylabel('Y'); ax_r.set_zlabel('Z')
            ax_r.set_title('3D camera space')
            ax_r.legend(fontsize=7)

            plt.tight_layout()
            plt.savefig("debug4_final_ordering.png", dpi=150, bbox_inches='tight')
            print("Saved debug4_final_ordering.png")
            plt.show()

        # ── Build u-params for next warm start ────────────────────────────────
        # 1. Calculate cumulative physical distance along the ordered keypoints
        ordered_pts = all_kpts[final_idx, :2]
        dists = np.linalg.norm(ordered_pts[1:] - ordered_pts[:-1], axis=1)
        cum_dists = np.insert(np.cumsum(dists), 0, 0.0)

        # 2. Extract the known (distance, s-parameter) pairs for the warm points
        known_d = []
        known_s = []
        for i, idx in enumerate(final_idx):
            if idx in warm_idx_set:
                known_d.append(cum_dists[i])
                known_s.append(new_warm_start_keypts[idx])

        # interp1d requires STRICTLY increasing x. Coincident ordered keypoints
        # give duplicate cumulative distances → division-by-zero → NaN s-values
        # that later crash refit_spline. Keep the first sample at each distance.
        if len(known_d) >= 2:
            known_d = np.asarray(known_d, dtype=float)
            known_s = np.asarray(known_s, dtype=float)
            keep = np.concatenate(([True], np.diff(known_d) > 1e-9))
            known_d, known_s = known_d[keep], known_s[keep]

        # 3. Interpolate (and extrapolate) s-values for all non-warm points
        if len(known_d) >= 2:
            s_interp = interp1d(known_d, known_s, fill_value="extrapolate")
            nwsk_full = s_interp(cum_dists).tolist()
        elif len(known_d) == 1:
            # Fallback if only 1 warm point survives
            nwsk_full = [known_s[0]] * len(final_idx)
        else:
            # Ultimate fallback if no warm points survive
            nwsk_full = np.linspace(0.0, 1.0, len(final_idx)).tolist()

        if nwsk_full[0] == None:
            pdb.set_trace()
        known = [(i, v) for i, v in enumerate(nwsk_full) if v is not None]
        if known:
            n_full  = len(nwsk_full)
            v_min   = known[0][1];  v_max   = known[-1][1]
            i_first = known[0][0];  i_last  = known[-1][0]
            t_lo = i_first / n_full
            t_hi = (i_last + 1) / n_full
            v_span = v_max - v_min if v_max != v_min else 1.0
            for i, v in known:
                nwsk_full[i] = float(t_lo + (v - v_min) / v_span * (t_hi - t_lo))

            known   = [(i, v) for i, v in enumerate(nwsk_full) if v is not None]
            anchors = list(known)
            if anchors[0][0] > 0:           anchors = [(0, 0.0)] + anchors
            if anchors[-1][0] < n_full - 1: anchors = anchors + [(n_full-1, 1.0)]
            for (i0, v0), (i1, v1) in zip(anchors[:-1], anchors[1:]):
                for i in range(i0, i1 + 1):
                    t = (i - i0) / (i1 - i0) if i1 != i0 else 0.0
                    nwsk_full[i] = float(v0 + t * (v1 - v0))

        n     = len(new_keypoints)
        order = np.arange(n)
        print(f"Combined ordering: {n_warm} warm + {n-n_warm} unmatched = {n} total.")

        if not speedy:
            fig = plt.figure(figsize=(12, 6))
            ax2d = fig.add_subplot(1, 2, 1)
            ax2d.imshow(mask, cmap="gray")
            for wi, ki in zip(warm_start_ids, range(len(warm_segment_kpts))):
                ax2d.plot([proj_pts[wi,1], warm_segment_kpts[ki,1]],
                          [proj_pts[wi,0], warm_segment_kpts[ki,0]],
                          c='pink', lw=1)
            ax2d.scatter(proj_pts[:,1], proj_pts[:,0], c='r', s=3, label='warm start')
            ax2d.scatter(old_keypoints[:,1], old_keypoints[:,0],
                         c=np.arange(len(old_keypoints)), cmap='Greens', s=10,
                         label='original keypoints', zorder=2)
            ax2d.scatter(new_keypoints[:,1], new_keypoints[:,0],
                         c=np.arange(len(new_keypoints)), cmap='Blues', s=20,
                         label='matched keypoints', zorder=3)
            ax2d.scatter(proj_curr_T[1], proj_curr_T[0],
                         c='r', s=30, marker='x', label='curr_T')
            ax2d.set_title('2D image'); ax2d.legend(fontsize=7)

            ax3d  = fig.add_subplot(1, 2, 2, projection='3d')
            nk_3d = unproject(new_keypoints)
            ok_3d = unproject(old_keypoints)
            wm_3d = unproject(warm_segment_kpts)
            wp    = warm_keypoints
            ax3d.scatter(wp[:,0], wp[:,1], wp[:,2], c='red', s=5, alpha=0.6,
                         label='warm proj pts')
            ax3d.plot(wp[:,0], wp[:,1], wp[:,2], c='red', lw=0.8, alpha=0.4)
            ax3d.scatter(nk_3d[:,0], nk_3d[:,1], nk_3d[:,2],
                         c=plt.cm.Blues(np.linspace(0.3, 1.0, len(nk_3d))),
                         s=15, zorder=3, label='matched keypoints')
            ax3d.plot(nk_3d[:,0], nk_3d[:,1], nk_3d[:,2],
                      c='steelblue', lw=1, alpha=0.5)
            ax3d.scatter(ok_3d[:,0], ok_3d[:,1], ok_3d[:,2],
                         c=plt.cm.Greens(np.linspace(0.3, 1.0, len(ok_3d))),
                         s=8, alpha=0.5, label='original keypoints')
            for wi, ki in zip(warm_start_ids, range(len(wm_3d))):
                w_pt = warm_keypoints[wi]; k_pt = wm_3d[ki]
                ax3d.plot([w_pt[0], k_pt[0]], [w_pt[1], k_pt[1]],
                          [w_pt[2], k_pt[2]], c='pink', lw=0.8, alpha=0.5)
            ct = curr_T[:3, 3]
            ax3d.scatter(ct[0], ct[1], ct[2], c='red', s=60, marker='x',
                         zorder=5, label='curr_T')
            ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
            ax3d.set_title('3D camera space'); ax3d.legend(fontsize=7)
            plt.tight_layout(); plt.show()

        # print(f"warm start matched keypoints {new_keypoints}\n")
        tw.stop("[warm] assembly + finalize")
        # EKF observations for the caller's update (shared with the KF-
        # projection path); final_idx is the assembled order here.
        ekf_obs = _make_ekf_obs(final_idx)
        if nwsk_full[0] == None:
            print("last nwsk")
            pdb.set_trace()

        return new_keypoints, order, nwsk_full, ekf_obs
    # ══════════════════════════════════════════════════════════════════════
    #  keypt_ordering
    # ══════════════════════════════════════════════════════════════════════
    # Minimum segment length that may seed the greedy assembly on reliability
    # alone.  The seed anchors the entire chain — every other segment is joined
    # onto it — so a 2-point fragment that happens to score 1.0 must not win
    # over a long, well-measured run.  If no segment reaches this length the
    # seed falls back to the longest, i.e. the old behaviour.
    # Turn angle (deg) between consecutive steps of a crawled segment above
    # which the segment is split.  Measured in the IMAGE plane only — stereo
    # depth is far too noisy to judge a corner by.  A thread cannot bend this
    # far within one keypoint spacing, so anything above it means the crawl
    # walked onto a different strand.  Lower to split more aggressively; a
    # tight loop's real curvature per hop is well under 45°, so there is a
    # wide safe margin below the default.
    COLD_SPLIT_TURN_DEG = 70.0
    # Sub-segments shorter than this after a split are discarded — a 1-2 point
    # stub carries no direction and only adds noise to the greedy assembly.
    COLD_SPLIT_MIN_LEN  = 3

    COLD_SEED_MIN_LEN = 5
    # Decimal places the segment confidence is rounded to before seeds are
    # compared, so confidences that are equal in substance tie and the length
    # tie-break decides.  Lower → more segments treated as equally reliable.
    _SEED_CONF_DECIMALS = 3

    def _split_sharp_turns(self, keypoints, segments, speedy=False):
        """Split segments wherever consecutive steps turn more than
        COLD_SPLIT_TURN_DEG, and drop the stubs.

        The turn at interior point j uses the steps j-1→j and j→j+1; the cut is
        made AFTER j, so j stays with the run leading into the corner and j+1
        starts a new one.  Zero-length steps (duplicate keypoints) carry no
        direction and never trigger a split.

        2-D only: (row, col).  Stereo depth is the noisiest axis by far, and a
        single bad z would otherwise manufacture corners that aren't there.
        """
        cos_thresh = float(np.cos(np.deg2rad(self.COLD_SPLIT_TURN_DEG)))
        kp = np.asarray(keypoints, dtype=float)
        out, n_split, n_cuts, n_dropped = [], 0, 0, 0

        for seg in segments:
            if len(seg) < 3:
                out.append(seg)
                continue
            idx = np.asarray(seg, dtype=int)
            if idx.max() >= len(kp) or idx.min() < 0:
                out.append(seg)                     # out-of-range → leave alone
                continue
            pts = kp[idx][:, :2]
            v   = np.diff(pts, axis=0)
            nrm = np.linalg.norm(v, axis=1)
            u   = v / np.maximum(nrm, 1e-9)[:, None]
            cosang = (u[:-1] * u[1:]).sum(axis=1)   # turn at seg index j = i+1
            real   = (nrm[:-1] > 1e-6) & (nrm[1:] > 1e-6)
            brk    = np.where(real & (cosang < cos_thresh))[0] + 1
            if brk.size == 0:
                out.append(seg)
                continue

            n_split += 1
            n_cuts  += int(brk.size)
            edges = [0, *(int(b) + 1 for b in brk), len(seg)]
            for a, b in zip(edges[:-1], edges[1:]):
                sub = seg[a:b]
                if len(sub) >= self.COLD_SPLIT_MIN_LEN:
                    out.append(sub)
                else:
                    n_dropped += len(sub)

        if n_split and not speedy:
            worst = self.COLD_SPLIT_TURN_DEG
            print(f"cold split: {n_split} segment(s) had turns > {worst:.0f}° "
                  f"→ {n_cuts} cut(s), {n_dropped} keypoint(s) dropped as "
                  f"stubs; {len(segments)} → {len(out)} segments")
        return out

    def keypt_ordering(self, img1, img_3D, cluster_map, keypoints,
                       grow_paths, adjacents, intersection_segments=None,
                       speedy=False, keypt_conf=None):
        """keypt_conf : optional (len(keypoints),) per-keypoint stereo
        confidence, aligned with `keypoints`.  When given, the greedy assembly
        seeds from the most RELIABLE segment instead of merely the longest —
        see the seed block in the assembly section.  None → seed by length."""
        # # ── Partition grow-paths into disjoint parts (BFS) ───────────────────
        # # Using deque + popleft() (O(1)) instead of list + pop(0) (O(N))
        # grow_parts = [[] for _ in grow_paths]
        # part_adjs  = [[] for _ in grow_paths]
        # DIRECTIONS = np.array([[1, 0],  [-1, 0],  [0, 1],  [0, -1],
        #                         [1, 1],  [-1, -1], [-1, 1], [1, -1]])
        # min_size = 2
        # # pdb.set_trace()
        # for c_id, path in enumerate(grow_paths):
        #     path_list   = list(path)
        #     visited     = [False] * len(path)
        #     pix2idx     = {pix: k for k, pix in enumerate(path)}
        #     num_visited = 0
        #     source      = 0
 
        #     while num_visited < len(visited):
        #         while visited[source]:
        #             source += 1
        #         # Start new BFS component
        #         frontier = deque([np.array(path_list[source])])
        #         visited[source] = True
        #         num_visited += 1
        #         part, part_adj = [], set()
 
        #         while frontier:
        #             curr = frontier.popleft()   # O(1) with deque
        #             part.append(curr)
        #             for d in DIRECTIONS:
        #                 neigh   = curr + d
        #                 t_neigh = tuple(neigh)
        #                 if t_neigh not in path:
        #                     # Out-of-path: check bounds then cluster membership
        #                     if (neigh[0] < 0 or neigh[0] >= cluster_map.shape[0] or
        #                             neigh[1] < 0 or neigh[1] >= cluster_map.shape[1]):
        #                         continue
        #                     nc = cluster_map[t_neigh]
        #                     if nc != 0 and nc != c_id + 1:
        #                         part_adj.add(nc - 1)
        #                     continue
        #                 n_idx = pix2idx[t_neigh]
        #                 if not visited[n_idx]:
        #                     frontier.append(neigh)
        #                     visited[n_idx] = True
        #                     num_visited += 1
 
        #         if len(part) > min_size:
        #             grow_parts[c_id].append(part)
        #             part_adjs[c_id].append(part_adj)

        # fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # colors = ['blue', 'orange', 'green', 'red']

        # # Plot 1: Original
        # ax = axes[0]
        # ax.set_title("Original 'grow_paths'")
        # ax.invert_yaxis()
        # for i, path in enumerate(grow_paths):
        #     rs = [p[0] for p in path]
        #     cs = [p[1] for p in path]
        #     ax.scatter(cs, rs, c=colors[i%4], label=f'Path {i}', s=40)
        # ax.legend()
        # ax.grid(True, linestyle='--', alpha=0.6)

        # # Plot 2: Processed
        # ax = axes[1]
        # ax.set_title("Processed 'grow_parts' (>2 px) & Adjacencies")
        # ax.invert_yaxis()
        # for i, parts in enumerate(grow_parts):
        #     for j, part in enumerate(parts):
        #         rs = [p[0] for p in part]
        #         cs = [p[1] for p in part]
        #         ax.scatter(cs, rs, c=colors[i%4], marker='o', s=40, label=f'Path {i} Part {j}' if j==0 else "")
                
        #         # Draw adjacencies
        #         if part_adjs[i][j]:
        #             centroid_r, centroid_c = np.mean(rs), np.mean(cs)
        #             for adj in part_adjs[i][j]:
        #                 if grow_parts[adj]:
        #                     a_rs = [p[0] for p in grow_parts[adj][0]]
        #                     a_cs = [p[1] for p in grow_parts[adj][0]]
        #                     a_cr, a_cc = np.mean(a_rs), np.mean(a_cs)
        #                     ax.plot([centroid_c, a_cc], [centroid_r, a_cr], 'k--', linewidth=1.5)
        #                     # ax.text((centroid_c+a_cc)/2, (centroid_r+a_cr)/2, "Touches!", color='black', fontsize=10, ha='center', weight='bold')

        # # ax.legend()
        # ax.grid(True, linestyle='--', alpha=0.6)

        # plt.tight_layout()
        # plt.show()
        # ── Intersection handling ─────────────────────────────────────────────
        tc = Timer(enabled=self.timing or not speedy)
        tc.start()
        kpts_2d          = keypoints[:, :2]
        prebuilt_segs    = []
        visited_crossing = set()
        plot_core_nodes  = set() # Tracks the removed center points for the visualizer
        
        CORE_EXCLUSION_RADIUS = 15.0   # px — dead-centre exclusion (matches warm_ordering)
        intersection_radius   = 100     # px — max search radius
        ANGLE_MARGIN_DEG      = 15.0   # Degrees of tolerance for axis alignment
        angle_cos_margin      = np.cos(np.deg2rad(ANGLE_MARGIN_DEG))
        POINTS_PER_SIDE       = 3      # keypoints to collect on each side of a crossing
 
        def _build_axis_seg(centroid, axis_vec, kpt_indices):
            """
            From a list of candidate keypoint indices belonging to one intersection
            arm, drop core-zone points, deduplicate, then sort the survivors along
            axis_vec. Mirrors warm_ordering's C.5 helper exactly.
            """
            mapped = []
            seen   = set()
            for kpt_idx in kpt_indices:
                if kpt_idx is None or kpt_idx in seen:
                    continue
                seen.add(kpt_idx)
                if np.linalg.norm(kpts_2d[kpt_idx] - centroid) <= CORE_EXCLUSION_RADIUS:
                    visited_crossing.add(kpt_idx)   # lock out crawler
                    plot_core_nodes.add(kpt_idx)    # track for visualisation
                    continue
                mapped.append(kpt_idx)
            if len(mapped) < 2:
                return []
            proj = (kpts_2d[mapped] - centroid) @ axis_vec
            return [mapped[k] for k in np.argsort(proj)]
 
        if intersection_segments is not None and len(intersection_segments) > 0:
            for crossing in intersection_segments:
                for axis_seg in crossing:
                    axis_vec = np.array(axis_seg['axis_vec'])
                    centroid = np.array(axis_seg['centroid'])
 
                    candidate_kpt_ids = []
                    n_positive = 0          # keypoints found on the +axis side
                    n_negative = 0          # keypoints found on the -axis side

                    # 1. Calculate vectors and distances from centroid to ALL sparse keypoints
                    vecs  = kpts_2d - centroid
                    dists = np.linalg.norm(vecs, axis=1)

                    # 2. Sort indices by distance to search radially outward
                    sorted_kpt_indices = np.argsort(dists)

                    for idx in sorted_kpt_indices:
                        dist = dists[idx]

                        # Stop searching if we exceed the maximum allowed intersection radius
                        if dist > intersection_radius:
                            break

                        # Handle core-zone points: Lock them out but don't assign them to an arm
                        if dist <= CORE_EXCLUSION_RADIUS:
                            visited_crossing.add(idx)
                            plot_core_nodes.add(idx)
                            continue

                        # 3. Check angular alignment using the dot product
                        v_norm   = vecs[idx] / dist
                        dot_prod = np.dot(v_norm, axis_vec)

                        # Collect up to POINTS_PER_SIDE nearest keypoints on each
                        # side of the crossing (a longer, better-conditioned arm
                        # than a single point per side).
                        if dot_prod > angle_cos_margin and n_positive < POINTS_PER_SIDE:
                            candidate_kpt_ids.append(idx)
                            n_positive += 1

                        elif dot_prod < -angle_cos_margin and n_negative < POINTS_PER_SIDE:
                            candidate_kpt_ids.append(idx)
                            n_negative += 1

                        # 4. Stop once both sides have enough points to bridge the crossing
                        if n_positive >= POINTS_PER_SIDE and n_negative >= POINTS_PER_SIDE:
                            break
 
                    # Lock out the selected candidate points globally so other segments don't grab them
                    for c_idx in candidate_kpt_ids:
                        visited_crossing.add(c_idx)
 
                    # Build and sort the segment
                    seg = _build_axis_seg(centroid, axis_vec, candidate_kpt_ids)
                    if len(seg) >= 2:
                        prebuilt_segs.append(seg) 
            # Final lock-out: ensure every surviving endpoint is also blocked
            # (core-excluded points were already added inside _build_axis_seg)
            for seg in prebuilt_segs:
                visited_crossing.update(seg)
 
            print(f"keypt_ordering: {len(prebuilt_segs)} intersection prebuilt segment(s) built "
                  f"(CORE_EXCLUSION_RADIUS={CORE_EXCLUSION_RADIUS}px, "
                  f"{len(plot_core_nodes)} core nodes excluded)")
 
        # ── Extract curve segments ────────────────────────────────────────────
        # adjacents may be longer than keypoints (phantom nodes from keypt_selection);
        # every neighbour access is clamped to n_kpts to stay in bounds.
        tc.stop("[cold] intersection prebuild")
        tc.start()
        n_kpts = len(keypoints)
 
        crawled_segs = []
        visited = [0] * n_kpts
        for c_id in visited_crossing:
            if c_id < n_kpts:
                visited[c_id] = 1
        # Seed from all degree-≤-2 nodes within keypoints range.
        # Do NOT exclude visited_crossing here: some arms have only intersection-
        # adjacent nodes as their sole endpoint; if excluded they get no crawl
        # source and that section of thread is silently dropped.  The
        # `visited[c_id] == 1` check in the selection loop skips them safely.
        outer_frontier = [
            c_id for c_id, neighs in enumerate(adjacents)
            if c_id < n_kpts and len(neighs) <= 2
        ]

        while True:
            source = None
            while outer_frontier:
                c_id = outer_frontier.pop()
                paths = 0
                for n_id in adjacents[c_id]:
                    n_id = int(n_id)
                    if n_id < n_kpts and visited[n_id] != 1:
                        paths += 1
                if paths == 1:
                    source = c_id
                    break
            if source is None:
                break
 
            frontier = [source]; visited[source] = 1; segment = []
            while frontier:
                curr = frontier.pop()
                # if len(segment) > 1:
                    # dist = np.linalg.norm(keypoints[segment[-1]]-keypoints[curr])
                    # if dist > max_dist:
                    #     print(f"new point {curr} is {dist} px away from previous point {segment[-1]}, not joining")
                    #     outer_frontier.append(curr)
                    #     continue
                segment.append(curr)
                mn_d, mn_nb = np.inf, None
                for neigh in adjacents[curr]:
                    neigh = int(neigh)
                    if neigh >= n_kpts:
                        continue
                    d = np.linalg.norm((keypoints[neigh] - keypoints[curr])[:2])
                    if visited[neigh] != 1 and d < mn_d:
                        mn_d, mn_nb = d, neigh
                if mn_nb is not None:
                    visited[mn_nb] = 1; frontier.append(mn_nb)
            crawled_segs.append(segment)
 
        tc.stop("[cold] segment crawl")
        tc.start()
        # ── Flat segment list (warm_ordering style) ───────────────────────────
        # Prebuilt intersection segments and crawled segments are collected into
        # one flat list.  No explicit stitching is performed here; directional
        # greedy assembly (below) is responsible for joining them in order.
        segments = ([list(s) for s in prebuilt_segs] +
                    [list(s) for s in crawled_segs if len(s) > 0])

        # A crawled segment is grown by nearest-unvisited-neighbour hops, so
        # where two strands pass close it can turn a corner and continue down
        # the WRONG one.  A real thread cannot double back within one keypoint
        # spacing, so a sharp turn is proof the segment spans two strands.
        # Split there and let the greedy assembly re-decide the join, instead
        # of carrying a corner that no downstream stage can undo.
        segments = self._split_sharp_turns(keypoints, segments, speedy=speedy)

        # # ── Extend endpoints ──────────────────────────────────────────────────
        # # Check by content membership rather than object-id since structures merged
        # for segment in segments:
        #     if any(idx in visited_crossing for idx in segment):
        #         continue  # Skip extending segments that are linked to intersections
                
        #     for side, endpt in enumerate([segment[0], segment[-1]]):
        #         for k, adjs in enumerate(part_adjs[endpt]):
        #             if len(adjs) == 0:
        #                 part    = np.array(grow_parts[endpt][k])
        #                 dists   = np.linalg.norm(
        #                     keypoints[endpt:endpt+1, :2] - part, axis=1)
        #                 new_end = part[np.argmax(dists)]
        #                 new_end = np.array([new_end[0], new_end[1],
        #                                     keypoints[endpt, 2]])
        #                 idx = keypoints.shape[0]
        #                 if side == 0:
        #                     segment.insert(0, idx)
        #                 else:
        #                     segment.append(idx)
        #                 keypoints  = np.concatenate(
        #                     (keypoints, np.expand_dims(new_end, 0)), axis=0)
        #                 grow_paths.append({tuple(pix) for pix in part})
 
        # ╔══════════════════════════════════════════════════════════════════════╗
        # ║  DEBUG PLOT A — segments before greedy assembly                     ║
        # ║                                                                     ║
        # ║  What to look for:                                                  ║
        # ║  • Cyan segs = intersection prebuilt; orange = crawled.            ║
        # ║  • ▶ marks the head (index 0) and ■ the tail of each segment.     ║
        # ║  • Segment labels (P0, C1 …) match stdout print for cross-ref.    ║
        # ║  • Check that every visible thread section has at least one seg.   ║
        # ║  • Segments that overlap or duplicate a region signal that the     ║
        # ║    intersection exclusion radius is too small / large.             ║
        # ╚══════════════════════════════════════════════════════════════════════╝
        n_prebuilt = len(prebuilt_segs)
 
        if not speedy:
            fig, ax = plt.subplots(figsize=(12, 10))
            fig.suptitle(
                f"DEBUG A — {n_prebuilt} prebuilt (cyan) + "
                f"{len(segments)-n_prebuilt} crawled (orange) segments",
                fontsize=11, fontweight='bold')
            ax.imshow(img1)
 
            print("\n── Segments before assembly ──────────────────────────────────")
            for si, seg in enumerate(segments):
                is_pre  = si < n_prebuilt
                label   = f"P{si}" if is_pre else f"C{si}"
                color   = np.array([0.0, 0.85, 0.85]) if is_pre \
                          else np.array([1.0, 0.55, 0.0])
 
                pts   = keypoints[seg, :2]  # (row, col)
                n_pts = len(pts)
 
                # Draw segment line with fading alpha to show direction
                for k in range(n_pts - 1):
                    alpha = 0.35 + 0.65 * k / max(n_pts - 1, 1)
                    ax.plot([pts[k, 1], pts[k+1, 1]],
                            [pts[k, 0], pts[k+1, 0]],
                            color=color, alpha=alpha, lw=2.0)
 
                # Scatter points in grey-scale by within-segment order
                sc_c = plt.cm.Greys(np.linspace(0.3, 1.0, n_pts))
                ax.scatter(pts[:, 1], pts[:, 0],
                           color=sc_c, s=22, zorder=3,
                           edgecolors=color, linewidths=1.0)
 
                # Head ▶ and tail ■
                ax.scatter(pts[0,  1], pts[0,  0],
                           marker='>', color=color, s=70, zorder=5)
                ax.scatter(pts[-1, 1], pts[-1, 0],
                           marker='s', color=color, s=70, zorder=5)
 
                # Centroid label
                mid = n_pts // 2
                ax.text(pts[mid, 1] + 3, pts[mid, 0] - 3,
                        label, fontsize=8, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.15',
                                  fc='black', alpha=0.55, ec='none'))
 
                kpt_ids_str = str(seg[:4])[:-1] + ('…]' if len(seg) > 4 else ']')
                print(f"  {label:4s}  len={n_pts:3d}  "
                      f"head_kpt={seg[0]:3d}  tail_kpt={seg[-1]:3d}  "
                      f"kpts={kpt_ids_str}")
 
            ax.legend(handles=[
                plt.Line2D([0],[0], color=[0.0,0.85,0.85], lw=2,
                           label='Prebuilt (intersection)'),
                plt.Line2D([0],[0], color=[1.0,0.55,0.0],  lw=2,
                           label='Crawled'),
                plt.Line2D([0],[0], marker='>', color='gray',
                           linestyle='None', markersize=8, label='Head ▶'),
                plt.Line2D([0],[0], marker='s', color='gray',
                           linestyle='None', markersize=8, label='Tail ■'),
            ], fontsize=8, loc='upper right')
 
            # Also annotate every keypoint with its original index
            for i, kpt in enumerate(keypoints):
                ax.text(kpt[1] + 1, kpt[0] + 1, str(i),
                        fontsize=5, color='white', alpha=0.6)
 
            plt.tight_layout()
            plt.savefig("dbg_A_segments_pre_assembly.png",
                        dpi=150, bbox_inches='tight')
            print("Saved dbg_A_segments_pre_assembly.png")
            plt.show()
 
        # ── Connect segments via Directional Greedy Endpoint Assembly ─────────
        if not speedy:
            print("\n── Greedy assembly joins ─────────────────────────────────────")
 
        if len(segments) > 1:
            # ── Seed selection ────────────────────────────────────────────────
            # The seed anchors the whole chain: every later segment is appended
            # or prepended to it, and its direction fixes the direction of the
            # result.  Seeding by LENGTH alone can anchor on a long run whose
            # stereo depth is mush, and every join then inherits that.  Prefer
            # the most RELIABLE segment among those long enough to be a credible
            # anchor (COLD_SEED_MIN_LEN); mean confidence is the score, segment
            # length the tie-break.  Only the seed changes — the join loop below
            # rescans all remaining segments each step regardless of order.
            conf = None
            if keypt_conf is not None:
                conf = np.asarray(keypt_conf, dtype=float)
                if conf.size < len(keypoints):
                    conf = None          # stale/short → fall back to length

            def _seg_conf(seg):
                idx = [i for i in seg if 0 <= i < conf.size]
                return float(np.mean(conf[idx])) if idx else 0.0

            # ROUNDED before comparison.  np.mean accumulates differently for
            # different-length segments, so two runs of identical confidence
            # can differ by 1 ULP — and with the median confidence often at
            # exactly 1.00, that float noise would decide the seed instead of
            # length.  Quantising makes near-equal confidences tie, so the
            # length tie-break does its job.
            def _seed_key(seg):
                return (round(_seg_conf(seg), self._SEED_CONF_DECIMALS),
                        len(seg))

            if conf is not None:
                eligible = [s for s in segments
                            if len(s) >= self.COLD_SEED_MIN_LEN] or segments
                seed = max(eligible, key=_seed_key)
                segments.remove(seed)
                order = seed
                if not speedy:
                    longest = max(segments + [seed], key=len)
                    print(f"  seed: len={len(seed)} conf={_seg_conf(seed):.2f} "
                          f"(longest available: len={len(longest)} "
                          f"conf={_seg_conf(longest):.2f})")
            else:
                segments.sort(key=len, reverse=True)
                order = segments.pop(0)

            # ── DEBUG: track join sequence for Plot B ─────────────────────────
            _join_log = []   # list of (step, action, direction, s_idx_orig, cost,
                             #          join_tail_pt, join_head_pt)
 
            join_step = 0
            while segments:
                best_cost = np.inf
                best_match = None
                nearest_cand = np.inf   # closest rejected gap, for the log

                o_start_pt = keypoints[order[0], :2]
                o_end_pt   = keypoints[order[-1], :2]
                
                # Macro-line orientations
                o_vec_start = keypoints[order[min(3, len(order)-1)], :2] - o_start_pt
                o_dir_start = o_vec_start / (np.linalg.norm(o_vec_start) + 1e-8)
                
                o_vec_end = o_end_pt - keypoints[order[max(-4, -len(order))], :2]
                o_dir_end = o_vec_end / (np.linalg.norm(o_vec_end) + 1e-8)
 
                # Print all candidates for this step
                if not speedy:
                    print(f"\n  Step {join_step}  chain len={len(order)}"
                          f"  start_kpt={order[0]}  end_kpt={order[-1]}")
 
                for s_idx, seg in enumerate(segments):
                    s_start_pt = keypoints[seg[0], :2]
                    s_end_pt   = keypoints[seg[-1], :2]
                    
                    s_vec_start = keypoints[seg[min(3, len(seg)-1)], :2] - s_start_pt
                    s_dir_start = s_vec_start / (np.linalg.norm(s_vec_start) + 1e-8)
                    
                    s_vec_end = s_end_pt - keypoints[seg[max(-4, -len(seg))], :2]
                    s_dir_end = s_vec_end / (np.linalg.norm(s_vec_end) + 1e-8)
                    
                    d_es = np.linalg.norm(o_end_pt - s_start_pt)
                    d_ee = np.linalg.norm(o_end_pt - s_end_pt)
                    d_ss = np.linalg.norm(o_start_pt - s_start_pt)
                    d_se = np.linalg.norm(o_start_pt - s_end_pt)
 
                    align_es = abs(np.dot(o_dir_end,   s_dir_start))
                    align_ee = abs(np.dot(o_dir_end,   s_dir_end))
                    align_ss = abs(np.dot(o_dir_start, s_dir_start))
                    align_se = abs(np.dot(o_dir_start, s_dir_end))
 
                    # Raw-distance cap: a pairing farther than the cap is not a
                    # candidate at all, whatever its alignment (see
                    # COLD_ASSEMBLY_MAX_JOIN_PX).
                    _cap = self.COLD_ASSEMBLY_MAX_JOIN_PX
                    nearest_cand = min(nearest_cand, d_es, d_ee, d_ss, d_se)
                    cost_es = (d_es * (1.0 + 3.0 * (1.0 - align_es))
                               if d_es <= _cap else np.inf)
                    cost_ee = (d_ee * (1.0 + 3.0 * (1.0 - align_ee))
                               if d_ee <= _cap else np.inf)
                    cost_ss = (d_ss * (1.0 + 3.0 * (1.0 - align_ss))
                               if d_ss <= _cap else np.inf)
                    cost_se = (d_se * (1.0 + 3.0 * (1.0 - align_se))
                               if d_se <= _cap else np.inf)
 
                    if not speedy:
                        print(f"    seg[{s_idx}] kpts={seg[0]}…{seg[-1]} len={len(seg)}"
                              f"  cost ES={cost_es:6.1f}(d={d_es:.1f} a={align_es:.2f})"
                              f"  EE={cost_ee:6.1f}(d={d_ee:.1f} a={align_ee:.2f})"
                              f"  SS={cost_ss:6.1f}(d={d_ss:.1f} a={align_ss:.2f})"
                              f"  SE={cost_se:6.1f}(d={d_se:.1f} a={align_se:.2f})")
 
                    min_cost = min(cost_es, cost_ee, cost_ss, cost_se)
                    if min_cost < best_cost:
                        best_cost = min_cost
                        if min_cost == cost_es:
                            best_match = (s_idx, 'append',  'forward',
                                          o_end_pt,   s_start_pt)
                        elif min_cost == cost_ee:
                            best_match = (s_idx, 'append',  'reverse',
                                          o_end_pt,   s_end_pt)
                        elif min_cost == cost_ss:
                            best_match = (s_idx, 'prepend', 'reverse',
                                          o_start_pt, s_start_pt)
                        elif min_cost == cost_se:
                            best_match = (s_idx, 'prepend', 'forward',
                                          o_start_pt, s_end_pt)
 
                if best_match is None:
                    # Nothing left within reach — stop rather than bridge an
                    # implausible gap.  The unjoined segments are discarded;
                    # a truncated thread is recoverable, a thread with a
                    # 500px teleport in the middle is not (it makes the optim
                    # QP infeasible and poisons the next frame's warm start).
                    n_left = len(segments)
                    n_pts  = sum(len(s) for s in segments)
                    print(f"cold assembly: stopping with {n_left} segment(s) "
                          f"({n_pts} keypoints) unjoined — nearest gap "
                          f"{nearest_cand:.1f}px > "
                          f"{self.COLD_ASSEMBLY_MAX_JOIN_PX:.0f}px cap; "
                          f"kept a chain of {len(order)}")
                    break

                s_idx, action, direction, join_order_pt, join_seg_pt = best_match

                if not speedy:
                    print(f"  → CHOSEN: seg[{s_idx}]  action={action}"
                          f"  direction={direction}  cost={best_cost:.1f}")
                    _join_log.append((join_step, action, direction,
                                      s_idx, best_cost,
                                      join_order_pt, join_seg_pt))
 
                seg = segments.pop(s_idx)
                if direction == 'reverse':
                    seg = seg[::-1]
                if action == 'append':
                    order = order + seg
                else:
                    order = seg + order
 
                join_step += 1
        else:
            order = segments[0] if segments else []
            _join_log = []
 
        # ╔══════════════════════════════════════════════════════════════════════╗
        # ║  DEBUG PLOT B — greedy join decisions                               ║
        # ║                                                                     ║
        # ║  What to look for:                                                  ║
        # ║  • Each dashed red line shows where two segments were bridged.     ║
        # ║  • A very long dashed line means the greedy chose a distant pair   ║
        # ║    because the angular alignment with the closer option was poor.  ║
        # ║  • Numbered circles show join order (0 = first join).             ║
        # ║  • The final chain (blue line) should trace the thread smoothly.  ║
        # ╚══════════════════════════════════════════════════════════════════════╝
        if not speedy:
            fig, ax = plt.subplots(figsize=(12, 10))
            fig.suptitle("DEBUG B — greedy join decisions", fontsize=11,
                         fontweight='bold')
            ax.imshow(img1)
 
            # Draw the final assembled chain
            pts_order = keypoints[order, :2]
            ax.plot(pts_order[:, 1], pts_order[:, 0],
                    c='steelblue', lw=1.5, alpha=0.6, label='assembled chain')
            ax.scatter(pts_order[:, 1], pts_order[:, 0],
                       c=np.arange(len(order)), cmap='Blues',
                       s=15, zorder=3)
            ax.scatter(pts_order[0,  1], pts_order[0,  0],
                       c='lime',  s=100, marker='^', zorder=6, label='chain start')
            ax.scatter(pts_order[-1, 1], pts_order[-1, 0],
                       c='white', s=100, marker='v', zorder=6, label='chain end')
 
            # Overlay each join as a dashed red bridge
            for step, action, direction, s_idx_orig, cost, p_order, p_seg in _join_log:
                ax.plot([p_order[1], p_seg[1]], [p_order[0], p_seg[0]],
                        color='red', linestyle='--', lw=2, zorder=5)
                ax.scatter([p_order[1], p_seg[1]], [p_order[0], p_seg[0]],
                           color='red', s=50, zorder=6)
                mid_col = (p_order[1] + p_seg[1]) / 2
                mid_row = (p_order[0] + p_seg[0]) / 2
                ax.text(mid_col, mid_row,
                        f"#{step}\n{action[:3]}/{direction[:3]}\nc={cost:.0f}",
                        fontsize=6, color='yellow', ha='center',
                        bbox=dict(boxstyle='round,pad=0.1',
                                  fc='black', alpha=0.6, ec='none'))
 
            ax.legend(fontsize=8, loc='upper right')
            plt.tight_layout()
            plt.savefig("dbg_B_greedy_joins.png", dpi=150, bbox_inches='tight')
            print("\nSaved dbg_B_greedy_joins.png")
            plt.show()
 
        if not speedy:
            fig, ax = plt.subplots(figsize=(12, 10))
            fig.suptitle("DEBUG C — assembled order (pre-spike-removal)",
                         fontsize=11, fontweight='bold')
            ax.imshow(img1)
            pts_order = keypoints[order, :2]
 
            # Connecting lines between consecutive points
            for k in range(len(order) - 1):
                p0 = pts_order[k];  p1 = pts_order[k + 1]
                ax.plot([p0[1], p1[1]], [p0[0], p1[0]],
                        c='white', lw=0.8, alpha=0.5)
 
            sc = ax.scatter(pts_order[:, 1], pts_order[:, 0],
                            c=np.arange(len(order)), cmap='hot',
                            s=35, zorder=4, edgecolors='gray', linewidths=0.4)
            plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label='order index')
 
            # Label every point with its order index and keypoint id
            for k, kpt_id in enumerate(order):
                ax.text(keypoints[kpt_id, 1] + 2, keypoints[kpt_id, 0] - 2,
                        f"{k}\n({kpt_id})",
                        fontsize=5, color='cyan',
                        bbox=dict(boxstyle='round,pad=0.1',
                                  fc='black', alpha=0.4, ec='none'))
 
            ax.scatter(pts_order[0,  1], pts_order[0,  0],
                       c='lime',  s=100, marker='^', zorder=6, label='start')
            ax.scatter(pts_order[-1, 1], pts_order[-1, 0],
                       c='white', s=100, marker='v', zorder=6, label='end')
            ax.legend(fontsize=8, loc='upper right')
            plt.tight_layout()
            plt.savefig("dbg_C_assembled_order.png", dpi=150, bbox_inches='tight')
            print("Saved dbg_C_assembled_order.png")
            plt.show()
 
        # ── Post-Processing: Remove Sharp Directional Spikes ──────────────────
        # Iteratively removes points that force the path to fold back or zig-zag
        if len(order) > 2:
            # Tunable Parameter: The dot-product threshold for a "sharp turn"
            #  1.0 = Perfectly straight line
            #  0.0 = 90 degree turn
            # -0.5 = 120 degree turn (sharp)
            # -1.0 = 180 degree U-turn (folding back on itself)
            SHARP_TURN_THRESH = 0.0  # Drops anything 90 degrees or sharper
            
            changed = True
            while changed and len(order) > 2:
                changed = False
                cleaned_order = [order[0]]
                
                k = 1
                while k < len(order) - 1:
                    pp  = keypoints[cleaned_order[-1], :2] # Previous point
                    cp  = keypoints[order[k], :2]          # Current point
                    np_ = keypoints[order[k+1], :2]        # Next point
                    
                    v_in  = cp - pp
                    v_out = np_ - cp
                    
                    norm_in  = np.linalg.norm(v_in)
                    norm_out = np.linalg.norm(v_out)
                    
                    # Prevent division by zero if keypoints are duplicates
                    if norm_in > 1e-5 and norm_out > 1e-5:
                        cos_angle = np.dot(v_in, v_out) / (norm_in * norm_out)
                        
                        if cos_angle < SHARP_TURN_THRESH:
                            # Sharp spike detected! Drop the current point.
                            changed = True
                            k += 1
                            continue
                            
                    cleaned_order.append(order[k])
                    k += 1
                    
                cleaned_order.append(order[-1])
                
                # Update the order for the next pass
                order = cleaned_order
 
        # ── Final Output & Plotting ───────────────────────────────────────────
        if not speedy:
            fig, ax = plt.subplots(figsize=(12, 10))
            fig.suptitle(f"DEBUG D — final order after spike removal  ({len(order)} pts)",
                         fontsize=11, fontweight='bold')
            ax.imshow(img1)
            if len(order) > 0:
                pts_final = keypoints[order, :2]
                for k in range(len(order) - 1):
                    p0 = pts_final[k]; p1 = pts_final[k + 1]
                    ax.plot([p0[1], p1[1]], [p0[0], p1[0]],
                            c='white', lw=0.8, alpha=0.5)
                sc = ax.scatter(pts_final[:, 1], pts_final[:, 0],
                                c=np.arange(len(order)), cmap='hot',
                                s=35, zorder=4, edgecolors='gray', linewidths=0.4)
                plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label='order index')
                for k, kpt_id in enumerate(order):
                    ax.text(keypoints[kpt_id, 1] + 2, keypoints[kpt_id, 0] - 2,
                            f"{k}({kpt_id})", fontsize=5, color='cyan',
                            bbox=dict(boxstyle='round,pad=0.1',
                                      fc='black', alpha=0.4, ec='none'))
                ax.scatter(pts_final[0,  1], pts_final[0,  0],
                           c='lime',  s=100, marker='^', zorder=6, label='start')
                ax.scatter(pts_final[-1, 1], pts_final[-1, 0],
                           c='white', s=100, marker='v', zorder=6, label='end')
                ax.legend(fontsize=8, loc='upper right')
            plt.tight_layout()
            plt.savefig("dbg_D_final_order.png", dpi=150, bbox_inches='tight')
            print("Saved dbg_D_final_order.png")
            plt.show()
 
        tc.stop("[cold] assembly + finalize")
        return img_3D, keypoints, grow_paths, order


    # ══════════════════════════════════════════════════════════════════════
    def hand_ordering(self, img1, img_3D, keypoints, needle_pos_file, P1):
        def gen_embedded(image, img_3D, keypoints):
            print("Generating keypoint hand ordering embedded")
            save_points, order, show_order = [], [], []
            accuracy = 10
            w_name   = ("Add (L-click) and Remove (R-click) keypoint. "
                        "Esc to Quit")

            def select_point(event, x, y, flags, param):
                nonlocal save_points, order, show_order
                if event == cv2.EVENT_LBUTTONDOWN:
                    print(f"img_3D value at {x, y} is {img_3D[y, x, 2]}")
                    candidates = [
                        [np.linalg.norm(pt - (y, x)), idx, pt[0], pt[1]]
                        for idx, pt in enumerate(keypoints[:, :2])
                        if np.linalg.norm(pt - (y, x)) <= accuracy
                           and idx not in show_order
                    ]
                    if candidates:
                        best = np.argmin(np.array(candidates)[:, 0])
                        show_order.append(int(candidates[best][1]))
                        order.append(len(order))
                        save_points.append(keypoints[int(candidates[best][1])])
                        print(f"point {save_points[-1]} is added")
                    clone = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
                    for idx, pt in enumerate(keypoints[:, :2]):
                        r = 5 if idx in show_order else 3
                        c = (0, 1, 0) if idx in show_order else (0, 0, 1)
                        cv2.circle(clone, (int(pt[1]), int(pt[0])),
                                   radius=r, color=c, thickness=-1)
                    cv2.imshow(w_name, clone)

                elif event == cv2.EVENT_RBUTTONDOWN:
                    if order:
                        order.pop(); show_order.pop(); save_points.pop()
                        print("last point removed")
                    clone = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
                    for idx, pt in enumerate(keypoints[:, :2]):
                        if idx in show_order:
                            cg = int((show_order.index(idx)+1)/len(show_order)*225)
                            cv2.circle(clone, (int(pt[1]), int(pt[0])),
                                       radius=5, color=(0, cg, 0), thickness=-1)
                        else:
                            cv2.circle(clone, (int(pt[1]), int(pt[0])),
                                       radius=3, color=(0, 0, 1), thickness=-1)
                    cv2.imshow(w_name, clone)

            cv2.namedWindow(w_name)
            cv2.setMouseCallback(w_name, select_point)
            clone = image.copy()
            for idx, pt in enumerate(keypoints[:, :2]):
                cv2.circle(clone, (int(pt[1]), int(pt[0])),
                           radius=3, color=(0, 0, 1), thickness=-1)
            cv2.imshow(w_name, clone)
            print("Embedding generated!")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            new_keypoints = np.array(save_points)
            print(f"new keypoints {new_keypoints}")
            return new_keypoints, order

        if needle_pos_file is not None:
            img1 = self.add_needle_point(img1, needle_pos_file, P1)
        return gen_embedded(img1, img_3D, keypoints)

    # ══════════════════════════════════════════════════════════════════════
    def add_needle_point(self, img1, needle_pos_file, P1):
        from scipy.spatial.transform import Rotation
        r = 8.2761
        with open(needle_pos_file, 'rb') as f:
            data = pickle.load(f)
        needle_pos = (np.array([data.get('x'), data.get('y'), data.get('z'),
                                 data.get('qw'), data.get('qx'),
                                 data.get('qy'), data.get('qz')]) * 1000)
        theta   = np.pi * 3 / 2
        conn_pt = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
        rot     = Rotation.from_quat(needle_pos[3:]).as_matrix()
        conn_pt = rot @ conn_pt + needle_pos[:3]

        aug     = np.append(conn_pt, 1.0)
        proj    = (P1 @ aug)
        proj   /= proj[2] + 1e-7
        rc      = np.round([proj[1], proj[0]]).astype(int)
        img1[rc[0]][rc[1]] = np.array([225, 225, 225])
        return img1