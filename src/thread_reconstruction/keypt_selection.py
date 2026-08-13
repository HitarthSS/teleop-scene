import enum
import pdb
import numpy as np
import copy
from collections import deque, defaultdict

import matplotlib.pyplot as plt

from scipy.spatial  import cKDTree
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.linear_model import RANSACRegressor
from skimage.measure import ransac, LineModelND
import scipy.ndimage as ndi
import time

# stereo matching vectorizing
from scipy.ndimage import uniform_filter

# gemini parallel
import os
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed

# intersection optimizing
import networkx as nx

global_executor = ProcessPoolExecutor(max_workers=os.cpu_count() - 1)

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def stop(self, name=""):
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"{name} function took: {elapsed_time:0.4f} seconds")

# ── Optional fast-path imports ────────────────────────────────────────────────
try:
    from numba import njit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False

try:
    import cupy as cp
    _CUPY = True
except ImportError:
    _CUPY = False

OG_BFS = True

# Density multiplier for the reliability-agnostic `full_clusters` used by
# intersection detection: their BFS size cap is max_size // FULL_CAP_DIVISOR,
# so a larger divisor → smaller clusters → denser cluster means (more points
# per crossing arm).  Raise if intersection detection still finds too few
# points; lower toward 1 to match the core `clusters` density.
FULL_CAP_DIVISOR = 1

# A full cluster mean is tagged "reliable" (kept after ordering, passed to
# optim) when at least this fraction of its pixels had reliable stereo
# (reliab > 0.5).  Lower to keep more points; raise to keep only high-quality
# depth points.
REL_FRAC_THRESH = 0.7

# Expected camera-frame workspace box (X, Y, Z — same units as the Q stereo
# reprojection, mm in this pipeline).  Segmented pixels whose reprojected 3-D
# point falls outside this box come from wrong disparities and are rejected
# BEFORE clustering, so they never become keypoints / dense points / adjacency
# nodes.  The rejection log prints the observed XYZ ranges each time it fires —
# use those to calibrate the box.  Widen if valid thread is being dropped.
# Only Z bites: it is really a disparity band filter (Z = f·B/d = 6996.6/d mm
# with this calibration).  Measured on left/right_thread_image: Z p50 = 155 mm,
# p99 = 237 mm — so a 160 mm ceiling deleted 17% of the thread in contiguous
# runs and broke the ordering graph there.  300 keeps 99.4% while still killing
# the d = 0 no-match sentinel (Z ≈ 1e10); 400 gains nothing further.  The near
# bound is inert — max_disp = 80 caps d at 79, so nothing can reproject nearer
# than 88.6 mm.  X/Y are inert too (reachable extents are only ±~140).
XYZ_EXPECTED_MIN = np.array([-300.0, -300.0,   0.0])
XYZ_EXPECTED_MAX = np.array([480.0,  640.0,  300.0])
# Fail-open guard: if the box would reject more than this fraction of all
# pixels it is assumed misconfigured (or the scene shifted) and the filter is
# skipped for the frame.
XYZ_MAX_REJECT_FRAC = 0.4

# Stereo plateau handling: disparities whose SSD is within this relative
# margin of the best are treated as ties.  On thread sections parallel to the
# epipolar lines (horizontal), the energy curve has a flat valley instead of a
# peak; a raw argmin picks the valley's FIRST (smallest) disparity, biasing
# depth consistently too far.  The match is instead reported at the CENTER of
# the contiguous tie plateau around the argmin, and reliability is damped by
# the plateau width.  Raise to treat more disparities as ties (wider plateaus,
# stronger damping); lower toward 0 to recover plain argmin behaviour.
PLATEAU_REL_EPS = 0.05

# Interpolated-depth confidence degradation: a cluster whose Z was
# interpolated from graph anchors (no reliable pixels of its own) is only as
# trustworthy as those anchors are close.  Its published stereo confidence
# (full_conf → last_keypt_conf) is scaled by exp(-d / INTERP_CONF_DECAY),
# d = graph distance (px) to the NEAREST reliable anchor: adjacent to an
# anchor ≈ unchanged, deep inside a long ambiguous run → near zero.  Raise to
# trust interpolated depth farther from its anchors.
INTERP_CONF_DECAY = 40.0

if _NUMBA:
    @njit(cache=True, parallel=True, fastmath=True)
    def _stereo_numba(segpix1, img1, img2, max_disp, rad, ignore_rad,
                            c_data, c_slope, c_shift, plateau_eps):
        num_pixels = segpix1.shape[0]
        reliab = np.zeros(num_pixels, dtype=np.float64)
        depth_calc = np.ones((4, num_pixels), dtype=np.float64)
        H, W, _ = img1.shape

        for i in prange(num_pixels):
            r = segpix1[i, 0]
            c = segpix1[i, 1]
            curr_max_disp = min(c, max_disp)

            r_min, r_max = max(0, r - rad), min(H, r + rad + 1)
            c_min, c_max = max(0, c - rad), min(W, c + rad + 1)

            seg_count = 0
            for wr in range(r_min, r_max):
                for wc in range(c_min, c_max):
                    if img1[wr, wc, 0] > 0 or img1[wr, wc, 1] > 0 or img1[wr, wc, 2] > 0:
                        seg_count += 1

            if seg_count == 0 or curr_max_disp == 0:
                reliab[i] = 0.0
                depth_calc[0, i] = r
                depth_calc[1, i] = c
                depth_calc[2, i] = 0.0
                depth_calc[3, i] = 1.0
                continue

            worst = 255.0**2 * 3 * seg_count
            energy = np.full(curr_max_disp, worst, dtype=np.float64)

            for d in range(1, curr_max_disp):
                # Right-mask constraint: the window centre must land on the
                # right segmentation.  Thread-vs-background comparisons carve
                # spurious energy valleys, especially on epipolar-parallel
                # (horizontal) sections.  (c - d >= 1 since d < curr_max_disp
                # <= c.)
                if (img2[r, c - d, 0] < 1.0 and img2[r, c - d, 1] < 1.0
                        and img2[r, c - d, 2] < 1.0):
                    continue
                ssd = 0.0
                all_zero = True
                truncated = False
                for wr in range(r_min, r_max):
                    if truncated:
                        break
                    for wc in range(c_min, c_max):
                        if img1[wr, wc, 0] > 0 or img1[wr, wc, 1] > 0 or img1[wr, wc, 2] > 0:
                            right_c = wc - d
                            if right_c < 0:
                                # Truncated window: summing fewer terms makes
                                # off-edge disparities artificially cheap
                                # (biased-close depth near the left border).
                                truncated = True
                                break
                            if img2[wr, right_c, 0] >= 1.0 or img2[wr, right_c, 1] >= 1.0 or img2[wr, right_c, 2] >= 1.0:
                                all_zero = False
                            diff_r = img1[wr, wc, 0] - img2[wr, right_c, 0]
                            diff_g = img1[wr, wc, 1] - img2[wr, right_c, 1]
                            diff_b = img1[wr, wc, 2] - img2[wr, right_c, 2]
                            ssd += (diff_r**2 + diff_g**2 + diff_b**2)
                if not truncated and not all_zero:
                    energy[d] = ssd

            best = np.min(energy)
            disp_i = np.argmin(energy)

            if best >= worst:
                # No disparity was actually evaluated (all skipped/off-mask).
                reliab[i] = 0.0
                depth_calc[0, i] = r
                depth_calc[1, i] = c
                depth_calc[2, i] = 0.0
                depth_calc[3, i] = 1.0
                continue

            # ── Plateau-aware disparity ───────────────────────────────────
            # Contiguous run of near-ties (within plateau_eps of best) around
            # the argmin.  argmin alone returns the run's FIRST index, so the
            # flat valleys of the aperture problem bias depth consistently
            # too far; the run centre is unbiased under symmetric ambiguity.
            tie_thresh = best * (1.0 + plateau_eps)
            lo = disp_i
            while lo > 1 and energy[lo - 1] <= tie_thresh:
                lo -= 1
            hi = disp_i
            while hi + 1 < curr_max_disp and energy[hi + 1] <= tie_thresh:
                hi += 1
            disp = 0.5 * (lo + hi)

            e_msk = energy.copy()
            ignore_start = max(disp_i - ignore_rad, 0)
            ignore_end = min(disp_i + ignore_rad + 1, curr_max_disp)
            for d in range(ignore_start, ignore_end):
                e_msk[d] = np.inf

            next_best = np.min(e_msk)
            x = (next_best - best) / ((best + 1e-7) * c_data)
            exponent = -c_slope * (x - c_shift)
            if exponent < -87: exponent = -87.0

            rel = 1.0 / (1.0 + np.exp(exponent))
            # Damp confidence by plateau width beyond the ignore window: a
            # wide tie plateau means the epipolar search was ambiguous even
            # when the next-best margin (taken outside the ignore window)
            # has not fully collapsed.
            plateau_w = hi - lo + 1
            ignore_w = 2 * ignore_rad + 1
            if plateau_w > ignore_w:
                rel *= ignore_w / plateau_w
            reliab[i] = rel
            depth_calc[0, i] = r
            depth_calc[1, i] = c
            depth_calc[2, i] = disp
            depth_calc[3, i] = 1.0

        return reliab, depth_calc
  
def _stereo_numpy_original(segpix1, img1, img2, max_disp, rad, ignore_rad,
                            c_data, c_slope, c_shift):
    N   = len(segpix1)
    H, W = img1.shape[:2]
    worst = 255.0**2 * (2*rad + 1)**2 * 3.0
 
    img2_row_sums = img2.sum(axis=2)
    img2_nz_cols  = {
        r: np.flatnonzero(img2_row_sums[r])
        for r in np.unique(segpix1[:, 0])
    }
 
    reliab     = np.zeros(N, dtype=np.float64)
    best_disps = np.zeros(N, dtype=np.int64)
 
    for i in range(N):
        pix = segpix1[i]
        try:
            curr_max_disp = min(int(pix[1]), max_disp)
            chunk = img1[pix[0]-rad : pix[0]+rad+1, pix[1]-rad : pix[1]+rad+1]
            seg   = (np.argwhere(chunk.sum(-1) > 0) + pix - rad)                              
 
            energy  = np.full(curr_max_disp, worst)
            nz_cols = img2_nz_cols.get(int(pix[0]), np.empty(0, np.int64))
            offsets = pix[1] - nz_cols
            valid   = offsets[(offsets > 0) & (offsets < curr_max_disp)]
 
            if len(valid) > 0 and seg.shape[0] > 0:
                g_l     = img1[seg[:, 0], seg[:, 1]].astype(np.float32)
                col_r   = seg[:, 1, None] - valid[None, :]
                col_r_c = np.clip(col_r, 0, W - 1).astype(np.int64)
                g_r_all = img2[seg[:, 0, None], col_r_c].astype(np.float32)
                g_r_all[col_r < 0] = 0.0
                all_zero = (g_r_all < 1.0).all(axis=(0, 2))
                ssd      = ((g_l[:, None, :] - g_r_all)**2).sum(axis=(0, 2))
                energy[valid] = np.where(all_zero, worst, ssd)
 
            best  = energy.min()
            disp  = int(energy.argmin())
            e_msk = energy.copy()
            e_msk[max(disp - ignore_rad, 0) : disp + ignore_rad + 1] = np.inf
            next_best = e_msk.min()
 
            x           = (next_best - best) / ((best + 1e-7) * c_data)
            reliab[i]   = 1.0 / (1.0 + np.exp(np.clip(-c_slope * (x - c_shift), -87, None)))
            best_disps[i] = disp
 
        except (ValueError, IndexError):
            pass
 
    return reliab, best_disps

def _run_stereo(segpix1, img1, img2, max_disp, rad, ignore_rad,
                c_data, c_slope, c_shift):
    if _NUMBA:
        print("~~~using numba")
        return _stereo_numba(segpix1, img1, img2, max_disp, rad, ignore_rad,
                             c_data, c_slope, c_shift, PLATEAU_REL_EPS)
    print("~~~using modified original")
    return _stereo_numpy_original(segpix1, img1, img2, max_disp, rad, ignore_rad, c_data, c_slope, c_shift)

# 12-connectivity used by the BFS clustering (8-neighbourhood + 2-px jumps).
_DIRS12 = np.array([[1, 0],  [-1, 0],  [0, 1],   [0, -1],
                    [1, 1],  [-1, -1], [-1, 1],  [1, -1],
                    [2, 0],  [-2, 0],  [0, 2],   [0, -2]], dtype=np.int64)
# 8-connectivity used by the adjacency grow BFS.
_DIRS8 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                   [1, 1], [-1, -1], [-1, 1], [1, -1]], dtype=np.int64)

if _NUMBA:
    @njit(cache=True)
    def _bfs_cluster_nb(seed_r, seed_c, H, W, cap, min_size, dirs):
        """Numba port of the size-capped BFS clustering (identical semantics
        to the Python version in keypt_selection): grow FIFO-BFS clusters from
        the seed pixels in order, chop at `cap`+1 pixels (leftover frontier is
        re-seeded), drop clusters smaller than `min_size`.

        Returns (rows, cols, labels, n_clusters) — pixels grouped contiguously
        by cluster label in [0, n_clusters).
        """
        n = seed_r.shape[0]
        vmap = np.ones((H, W), np.uint8)     # 0 = unvisited seed pixel
        for i in range(n):
            vmap[seed_r[i], seed_c[i]] = 0
        out_r   = np.empty(n, np.int64)
        out_c   = np.empty(n, np.int64)
        out_lab = np.empty(n, np.int64)
        out_n   = 0
        qr = np.empty(n, np.int64)
        qc = np.empty(n, np.int64)
        n_clusters = 0
        nd    = dirs.shape[0]
        src_i = 0
        while True:
            while src_i < n and vmap[seed_r[src_i], seed_c[src_i]] == 1:
                src_i += 1
            if src_i >= n:
                break
            qr[0] = seed_r[src_i]
            qc[0] = seed_c[src_i]
            head, tail = 0, 1
            vmap[seed_r[src_i], seed_c[src_i]] = 1
            csize  = 0
            cstart = out_n
            # size check BEFORE each pop (mirrors `len(cluster) <= cap`), so a
            # cluster may reach cap+1 pixels exactly like the Python version.
            while head < tail and csize <= cap:
                r = qr[head]; c = qc[head]; head += 1
                out_r[out_n] = r; out_c[out_n] = c
                out_lab[out_n] = n_clusters
                out_n += 1; csize += 1
                for d in range(nd):
                    nr = r + dirs[d, 0]; nc = c + dirs[d, 1]
                    if 0 <= nr < H and 0 <= nc < W and vmap[nr, nc] == 0:
                        vmap[nr, nc] = 1
                        qr[tail] = nr; qc[tail] = nc; tail += 1
            # leftover frontier of a cap-chopped cluster → future seeds
            for j in range(head, tail):
                vmap[qr[j], qc[j]] = 0
            if csize >= min_size:
                n_clusters += 1
            else:
                out_n = cstart                 # drop the undersized cluster
        return out_r[:out_n], out_c[:out_n], out_lab[:out_n], n_clusters

    @njit(cache=True)
    def _adjacency_nb(solid_map, mask_on, starts, pix_r, pix_c, H, W, dirs):
        """Numba port of the adjacency grow-BFS: for each cluster (CSR layout
        starts/pix_r/pix_c), flood outward through on-mask pixels that belong
        to no cluster (solid_map == 0) and record which other cluster labels
        are reached.  Returns an (n_edges, 2) array of directed (cluster,
        neighbour) index pairs (duplicates included — dedup outside).
        """
        n_clusters = starts.shape[0] - 1
        visited_gen = np.zeros((H, W), np.int32)
        # per-cluster dedup of recorded neighbours (gen-stamped), so each
        # (cluster, neighbour) pair is emitted once no matter how many pixels
        # touch — keeps the fixed edge buffer from overflowing.
        seen = np.zeros(n_clusters + 1, np.int32)
        qr = np.empty(H * W, np.int64)
        qc = np.empty(H * W, np.int64)
        max_edges = 64 * n_clusters + 64
        edges  = np.empty((max_edges, 2), np.int64)
        n_edges = 0
        nd = dirs.shape[0]
        for cid in range(n_clusters):
            gen = cid + 1
            target = cid + 1
            head, tail = 0, 0
            for k in range(starts[cid], starts[cid + 1]):
                r = pix_r[k]; c = pix_c[k]
                visited_gen[r, c] = gen
                qr[tail] = r; qc[tail] = c; tail += 1
            while head < tail:
                r = qr[head]; c = qc[head]; head += 1
                for d in range(nd):
                    nr = r + dirs[d, 0]; nc = c + dirs[d, 1]
                    if (0 <= nr < H and 0 <= nc < W
                            and visited_gen[nr, nc] != gen):
                        visited_gen[nr, nc] = gen
                        if mask_on[nr, nc]:
                            neigh = solid_map[nr, nc]
                            if neigh != target:
                                if neigh != 0:
                                    if (seen[neigh] != gen
                                            and n_edges < max_edges):
                                        seen[neigh] = gen
                                        edges[n_edges, 0] = cid
                                        edges[n_edges, 1] = neigh - 1
                                        n_edges += 1
                                else:
                                    qr[tail] = nr; qc[tail] = nc; tail += 1
        return edges[:n_edges]

print("Warming up Numba functions...")
dummy_pix = np.zeros((10, 2), dtype=np.int64, order='F')
dummy_img = np.zeros((480, 640, 3), dtype=np.uint32, order='C')
_run_stereo(dummy_pix, dummy_img, dummy_img, 10, 2, 2, 5, 8, 0.8)
if _NUMBA:
    _bfs_cluster_nb(np.array([1, 2], dtype=np.int64), np.array([1, 2], dtype=np.int64),
                    8, 8, 5, 1, _DIRS12)
    _adjacency_nb(np.zeros((8, 8), np.int32), np.zeros((8, 8), np.bool_),
                  np.array([0, 1], np.int64), np.array([1], np.int64),
                  np.array([1], np.int64), 8, 8, _DIRS8)
print("Warm-up complete. Starting live pipeline.")

class Select():
    def __init__(self, args):
        # --time: print per-stage Timer breakdowns even in speedy mode
        self.timing = getattr(args, 'time', False)
        # Continuous stereo confidence per full cluster mean, set by
        # keypt_selection() and read by the caller for the published
        # per-point reliability.  None until the first frame.
        self.last_keypt_conf = None

    def keypt_selection(self, img1, img2, mask1, Q, speedy=False):
        # Timers print when NOT speedy (as before) or when --time is passed.
        timing = (not speedy) or self.timing
        if timing:
            t_full = Timer()
            t_full.start()

            t_setup = Timer()
            t_setup.start()
            
        # ── Segmented pixel coordinates ───────────────────────────────────────
        segpix1 = np.argwhere(mask1 > 0)
        img_3D  = np.zeros((img1.shape[0], img1.shape[1], 3))

        max_disp   = int(80 * img1.shape[1] / 640)
        rad        = int(8  * img1.shape[1] / 640)
        c_data     = 5
        c_slope    = 4
        c_shift    = 0.3 # TODO tune if needed
        ignore_rad = int(1  * img1.shape[1] / 640)
        
        if timing:
            t_setup.stop("[select] setup")
        
        # ── Stereo matching ───────────────────────────────────────────────────
        if timing:
            t_disp = Timer()
            t_disp.start()
            
        reliab, best_disps = _run_stereo(
            segpix1, img1, img2, max_disp, rad, ignore_rad,
            c_data, c_slope, c_shift)
            
        if timing:
            t_disp.stop("[select] run stereo 1")
            t_setup.start()

        # ── Reproject to 3-D ─────────────────────────────────────────────────
        if OG_BFS:
            depth_calc = best_disps
            depth_calc = np.matmul(Q, depth_calc)
        else:
            depth_calc = best_disps
            depth_calc = np.matmul(Q, depth_calc)
            
        depth_calc /= np.clip(depth_calc[3], a_min=1e-7, a_max=None)

        # ── Expected-range (workspace) filter ─────────────────────────────────
        # Reject pixels whose reprojected 3-D position is outside the expected
        # XYZ box — wrong disparities land far outside the physical workspace,
        # and once such a point becomes a keypoint it corrupts ordering and can
        # make the optim QP infeasible.  Filtering here removes it from every
        # downstream structure at once (clusters, dense_pts, adjacency).
        X, Y, Z  = depth_calc[0], depth_calc[1], depth_calc[2]
        finite   = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
        in_range = (finite &
                    (X >= XYZ_EXPECTED_MIN[0]) & (X <= XYZ_EXPECTED_MAX[0]) &
                    (Y >= XYZ_EXPECTED_MIN[1]) & (Y <= XYZ_EXPECTED_MAX[1]) &
                    (Z >= XYZ_EXPECTED_MIN[2]) & (Z <= XYZ_EXPECTED_MAX[2]))
        n_total = len(segpix1)
        n_rej   = int(n_total - in_range.sum())
        if n_rej and n_rej > XYZ_MAX_REJECT_FRAC * n_total:
            fx = lambda a: f"[{np.nanmin(a):.1f}, {np.nanmax(a):.1f}]"
            print(f"[select] range filter SKIPPED: would reject "
                  f"{n_rej}/{n_total} px — box likely misconfigured.  "
                  f"Observed X{fx(X)} Y{fx(Y)} Z{fx(Z)} vs expected "
                  f"X[{XYZ_EXPECTED_MIN[0]}, {XYZ_EXPECTED_MAX[0]}] "
                  f"Y[{XYZ_EXPECTED_MIN[1]}, {XYZ_EXPECTED_MAX[1]}] "
                  f"Z[{XYZ_EXPECTED_MIN[2]}, {XYZ_EXPECTED_MAX[2]}]")
        elif n_rej:
            fx = lambda a: f"[{np.nanmin(a):.1f}, {np.nanmax(a):.1f}]"
            print(f"[select] range filter: rejected {n_rej}/{n_total} px "
                  f"outside expected XYZ box "
                  f"(observed X{fx(X)} Y{fx(Y)} Z{fx(Z)})")
            segpix1    = segpix1[in_range]
            reliab     = reliab[in_range]
            depth_calc = depth_calc[:, in_range]

        img_3D[segpix1[:, 0], segpix1[:, 1], 2] = depth_calc[2]

        # ── Prune unreliable points ───────────────────────────────────────────
        relidx = np.argwhere(reliab > 0.3)
        relpts = segpix1[relidx[:, 0]]
        dense_pts = segpix1.astype(float)
        
        H, W = mask1.shape
        # max_size  = segpix1.shape[0] // 150
        # min_size  = segpix1.shape[0] // 3000
        max_size  = segpix1.shape[0] // 100
        min_size  = segpix1.shape[0] // 400
        print(f"segpix1[0]: {segpix1.shape[0]}")
        print(f"max size: {max_size}\n min size: {min_size}")

        if timing:
            t_setup.stop("[select] setup 2")
            t_loop = Timer()
            t_loop.start()

        if OG_BFS:
            # ── BFS clustering ────────────────────────────────────────────────
            DIRECTIONS = np.array([[1, 0],  [-1, 0],  [0, 1],   [0, -1],
                                    [1, 1],  [-1, -1], [-1, 1],  [1, -1],
                                    [2, 0],  [-2, 0],  [0, 2],   [0, -2]])

            def _bfs_cluster(seed_pts, cap=max_size):
                """Grow size-capped clusters from seed_pts.  `cap` bounds a
                cluster's pixel count: a smaller cap yields more, smaller
                clusters (denser cluster means).  Seeding from relpts prunes
                unreliable points; seeding from all segpix1 keeps them."""
                vlist = deque(seed_pts.copy())
                vmap  = np.ones_like(mask1)
                vmap[seed_pts[:, 0], seed_pts[:, 1]] = 0
                clusters = []
                escape   = False

                while vlist:
                    cluster = []
                    source  = vlist.popleft()
                    while vmap[source[0], source[1]] == 1:
                        if not vlist:
                            escape = True
                            break
                        source = vlist.popleft()
                    if escape:
                        break

                    frontier = deque([source])
                    vmap[source[0], source[1]] = 1
                    while frontier and len(cluster) <= cap:
                        curr = frontier.popleft()
                        cluster.append(curr)

                        nbrs     = curr + DIRECTIONS
                        in_bnds  = ((nbrs[:, 0] >= 0) & (nbrs[:, 0] < H) &
                                    (nbrs[:, 1] >= 0) & (nbrs[:, 1] < W))
                        nbrs     = nbrs[in_bnds]
                        if len(nbrs):
                            new_pts = nbrs[vmap[nbrs[:, 0], nbrs[:, 1]] == 0]
                            if len(new_pts):
                                vmap[new_pts[:, 0], new_pts[:, 1]] = 1
                                frontier.extend(new_pts)

                    while frontier:
                        curr = frontier.popleft()
                        vmap[curr[0], curr[1]] = 0

                    if len(cluster) >= min_size:
                        clusters.append(cluster)
                return clusters

            # Single BFS over ALL segmentation pixels, with a SMALLER cap so
            # the reliability-agnostic cluster means are denser (more points
            # for ordering + intersection detection).  Reliability is applied
            # later as a per-cluster flag, so no separate reliable-only BFS is
            # needed.
            full_cap = max(min_size + 1, max_size // FULL_CAP_DIVISOR)
            print(f"full cap: {full_cap}")
            if _NUMBA:
                # compiled port of _bfs_cluster (same semantics, ~50x faster)
                out_r, out_c, out_lab, n_cl = _bfs_cluster_nb(
                    np.ascontiguousarray(segpix1[:, 0], dtype=np.int64),
                    np.ascontiguousarray(segpix1[:, 1], dtype=np.int64),
                    H, W, full_cap, min_size, _DIRS12)
                pts = np.column_stack((out_r, out_c))
                # pixels are emitted grouped by label → split at label changes
                bounds = np.flatnonzero(np.diff(out_lab)) + 1
                full_clusters = [np.asarray(c) for c in np.split(pts, bounds)] \
                    if len(pts) else []
            else:
                full_clusters = _bfs_cluster(segpix1, cap=full_cap)
        else:
            # ── Clustering via scipy.ndimage.label ───────────────────────────
            rel_mask = np.zeros((H, W), dtype=np.uint8)
            rel_mask[relpts[:, 0], relpts[:, 1]] = 1

            cross_3x3  = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
            dilated    = ndi.binary_dilation(rel_mask, structure=cross_3x3)
            struct     = np.ones((3, 3), dtype=bool)          
            labeled_d, n_comp = ndi.label(dilated, structure=struct)
            
            labeled    = np.where(rel_mask, labeled_d, 0)
            n_comp     = int(labeled.max())
            comp_sizes = ndi.sum(rel_mask, labeled, range(1, n_comp + 1))
            
            clusters = []
            for lbl, sz in enumerate(comp_sizes, start=1):
                if sz < min_size:
                    continue
                pts = np.argwhere(labeled == lbl)
                if sz <= max_size:
                    clusters.append(pts)
                else:
                    sub_mask          = (labeled == lbl).astype(np.uint8)
                    sub_lbl, sub_n    = ndi.label(sub_mask)
                    for sl in range(1, sub_n + 1):
                        sub_pts = np.argwhere(sub_lbl == sl)
                        if len(sub_pts) >= min_size:
                            clusters.append(sub_pts)
            # ndimage path already clusters all reliable pixels; reuse it as the
            # full set (this branch is not the active OG_BFS path).
            full_clusters = clusters

        # ── Per-pixel reliability lookups ─────────────────────────────────────
        # Built BEFORE the Z filter / cluster means so cluster depth can be
        # computed from reliable pixels only (below).  segpix1 and reliab were
        # filtered together by the range filter above, so they stay aligned.
        # rel_lookup: boolean reliab > 0.3 flag.  conf_lookup: the CONTINUOUS
        # stereo confidence — sigmoid of the best/second-best SSD margin, i.e.
        # genuine MEASUREMENT quality; its cluster mean is published as the
        # per-point reliability (see optim.RELIABILITY_MODE).
        rel_lookup = np.zeros((H, W), dtype=bool)
        rel_lookup[relpts[:, 0], relpts[:, 1]] = True
        conf_lookup = np.zeros((H, W), dtype=np.float64)
        conf_lookup[segpix1[:, 0], segpix1[:, 1]] = reliab

        # ── Z-Depth Outlier Removal (applied to BOTH cluster sets) ────────────

        def _cluster_z(c):
            """Cluster depth = median Z over the cluster's RELIABLE pixels
            when it has any.  The matcher's own confidence already flags
            ambiguous matches (flat epipolar valleys on horizontal sections);
            including them drags the median toward correlated-wrong depths.
            Falls back to the all-pixel median when no pixel is reliable."""
            z   = img_3D[c[:, 0], c[:, 1], 2]
            rel = rel_lookup[c[:, 0], c[:, 1]]
            return np.median(z[rel]) if rel.any() else np.median(z)

        def _z_filter(cl):
            """3σ Z-outlier rejection.  Statistics and rejection consider only
            clusters whose Z came from reliable pixels: fallback (no-reliable-
            pixel) clusters have known-ambiguous Z, so they would poison the
            mean/std, and they are kept regardless because the depth
            interpolation below re-derives their Z from the graph anchors."""
            cl = [np.asarray(c) for c in cl]
            for n_std in (3, 3):
                if not cl:
                    break
                zc     = np.array([_cluster_z(c) for c in cl])
                hasrel = np.array([bool(rel_lookup[c[:, 0], c[:, 1]].any())
                                   for c in cl])
                if hasrel.any():
                    m, s = zc[hasrel].mean(), zc[hasrel].std() + 1e-9
                    cl = [c for i, c in enumerate(cl)
                          if (not hasrel[i]) or abs(zc[i] - m) / s < n_std]
                else:
                    m, s = zc.mean(), zc.std() + 1e-9
                    cl = [c for i, c in enumerate(cl)
                          if abs(zc[i] - m) / s < n_std]
            return cl

        def _means(cl):
            means = np.zeros((len(cl), 3))
            for idx, c in enumerate(cl):
                c = np.asarray(c)
                means[idx, :2] = c.mean(axis=0)
                means[idx, 2]  = _cluster_z(c)
            return means

        full_clusters = _z_filter(full_clusters)

        # ── Full (reliability-agnostic, denser) cluster means — the ORDERING
        # substrate.  cluster_map / solidify / adjacency below are all built
        # from full_clusters so the cold graph-ordering path orders these same
        # points.  full_clusters covers low-reliability regions the reliable
        # set misses, giving a more complete ordering.
        full_cluster_means = _means(full_clusters)

        # ── Per-point reliability flag ────────────────────────────────────────
        # Rather than keep a separate reliable cluster_means array, tag each
        # full cluster mean with whether it is "reliable" (enough of its pixels
        # had reliable stereo).  Ordering runs on ALL full means; after ordering
        # the sequence is filtered to the reliable ones before optimisation.
        # (rel_lookup / conf_lookup are built above, before the Z filter.)
        reliable_flag = np.zeros(len(full_clusters), dtype=bool)
        full_conf     = np.zeros(len(full_clusters), dtype=np.float64)
        # Whether each cluster has ANY reliable pixel — i.e. whether its
        # _cluster_z median came from reliable stereo or from the all-pixel
        # fallback.  Fallback clusters get their Z re-derived from the
        # adjacency graph once it exists (depth interpolation below).
        has_rel_z     = np.zeros(len(full_clusters), dtype=bool)
        for idx, c in enumerate(full_clusters):
            c = np.asarray(c)
            relvals  = rel_lookup[c[:, 0], c[:, 1]]
            rel_frac = relvals.mean() if len(c) else 0.0
            reliable_flag[idx] = rel_frac >= REL_FRAC_THRESH
            has_rel_z[idx] = bool(relvals.any())
            full_conf[idx] = (conf_lookup[c[:, 0], c[:, 1]].mean()
                              if len(c) else 0.0)
        # Published via an attribute rather than the return tuple: callers
        # unpack that tuple with differing arities.
        self.last_keypt_conf = full_conf
        if not speedy:
            print(f"[select] stereo confidence per cluster mean: "
                  f"min={full_conf.min():.2f} median={np.median(full_conf):.2f} "
                  f"max={full_conf.max():.2f}")

        # int32, NOT zeros_like(mask1): a boolean mask (e.g. from clip_mask)
        # would make this a bool array and collapse every cluster id to True.
        cluster_map = np.zeros(mask1.shape[:2], dtype=np.int32)
        for idx, c in enumerate(full_clusters):
            c = np.asarray(c)
            cluster_map[c[:, 0], c[:, 1]] = idx + 1

        # From here on the graph substrate (solidify, adjacency) uses the full
        # clusters, so cold ordering operates on full_cluster_means.
        clusters = [np.asarray(c) for c in full_clusters]

        if not speedy:
            # Save as an overlay plot (like the intersection plot) rather than
            # a raw array.
            fig_fc, ax_fc = plt.subplots(1, 1, figsize=(8, 8))
            ax_fc.imshow(mask1, cmap='gray')
            ax_fc.scatter(full_cluster_means[:, 1], full_cluster_means[:, 0],
                          c='lime', s=12, alpha=0.8)
            ax_fc.set_title(f"full (reliability-agnostic) cluster means "
                            f"(n={len(full_cluster_means)})")
            plt.tight_layout()
            plt.savefig("debug_select_full_cluster_means.png", dpi=150,
                        bbox_inches='tight')
            plt.close(fig_fc)

        if not speedy:
            # SAVED, not shown: under the Agg backend plt.show() is a no-op
            # (the "FigureCanvasAgg is non-interactive" warning), so this
            # figure — the only view of WHERE the reliability filter drops
            # keypoints — was being built and discarded every frame.
            rel_means  = full_cluster_means[reliable_flag]
            drop_means = full_cluster_means[~reliable_flag]
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            ax[0].set_title(f"cluster means — kept {int(reliable_flag.sum())}"
                            f"/{len(full_cluster_means)} "
                            f"(REL_FRAC_THRESH={REL_FRAC_THRESH})")
            ax[0].imshow(mask1, cmap='gray')
            ax[0].scatter(dense_pts[:, 1], dense_pts[:, 0], c="red", s=4,
                          alpha=0.25, label="dense px")
            # DROPPED drawn on top and large: these are the ones that never
            # reach optim, so their spatial distribution is the whole question.
            ax[0].scatter(rel_means[:, 1], rel_means[:, 0], c="lime", s=18,
                          label=f"reliable ({len(rel_means)})", zorder=3)
            ax[0].scatter(drop_means[:, 1], drop_means[:, 0], c="magenta",
                          s=40, marker="x", zorder=4,
                          label=f"DROPPED ({len(drop_means)})")
            ax[0].legend(fontsize=8, loc="upper right")
            ax[1].set_title("stereo confidence per cluster mean "
                            "(sorted; dashed = threshold)")
            _conf_sorted = np.sort(full_conf)
            ax[1].plot(_conf_sorted, marker='.', lw=1)
            ax[1].axhline(REL_FRAC_THRESH, ls='--', c='r')
            ax[1].set_xlabel("cluster (sorted by confidence)")
            ax[1].set_ylabel("mean stereo confidence")
            ax[1].set_ylim(-0.02, 1.02)
            plt.tight_layout()
            plt.savefig("debug_select_reliability.png", dpi=150,
                        bbox_inches='tight')
            plt.close(fig)
            print("Saved debug_select_reliability.png")
            print(f"full cluster means: {len(full_cluster_means)}  "
                  f"reliable: {int(reliable_flag.sum())}")

        if timing:
            t_loop.stop("[select] remove z-depth outliers")
            t_loop.start()
        
        # ── Solidify (Optimized via Maximum Filter) ───────────────────────────
        solid_clusters = [(c.tolist() if isinstance(c, np.ndarray) else c) for c in clusters]
        solid_map      = cluster_map.copy()
        
        # Dilate cluster IDs to instantly identify assignment bounds
        dilated_map = ndi.maximum_filter(solid_map, size=3)
        
        # Identify non-assigned thread pixels
        unassigned_mask = solid_map[segpix1[:, 0], segpix1[:, 1]] == 0
        unassigned_pix = segpix1[unassigned_mask]
        
        # Fast map lookup directly over unassigned indices
        new_ids = dilated_map[unassigned_pix[:, 0], unassigned_pix[:, 1]]
        valid_mask = new_ids > 0
        
        valid_pix = unassigned_pix[valid_mask]
        valid_ids = new_ids[valid_mask].astype(int)
        
        # Matrix assignment mapping
        solid_map[valid_pix[:, 0], valid_pix[:, 1]] = valid_ids
        for p, cid in zip(valid_pix, valid_ids):
            solid_clusters[cid - 1].append(p.tolist())
            
        if timing:
            t_loop.stop("[select] solidify")
            t_loop.start()
        
        # ── Adjacency via grow-path BFS ───────────────────────────────────────
        # For each cluster, flood outward through on-mask pixels that belong to
        # no cluster (solid_map == 0) and record which other clusters are
        # reached.  Compiled (numba) when available; Python fallback otherwise.
        # NOTE: grow_paths pixel sets are only consumed by legacy/commented
        # code — the numba path returns them empty (adjacency is unaffected).
        adjacents  = [set() for _ in solid_clusters]
        grow_paths = [set() for _ in solid_clusters]

        if _NUMBA and solid_clusters:
            # CSR layout of cluster pixels for the compiled kernel
            sizes  = np.array([len(c) for c in solid_clusters], dtype=np.int64)
            starts = np.concatenate(([0], np.cumsum(sizes)))
            allpix = np.concatenate(
                [np.asarray(c, dtype=np.int64).reshape(-1, 2)
                 for c in solid_clusters])
            edges = _adjacency_nb(
                np.ascontiguousarray(solid_map, dtype=np.int32),
                np.ascontiguousarray(mask1 > 0),
                starts, np.ascontiguousarray(allpix[:, 0]),
                np.ascontiguousarray(allpix[:, 1]), H, W, _DIRS8)
            for a, b in edges:
                adjacents[a].add(int(b))
        else:
            DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1))

            # Generation-stamped visited array: a cell counts as "visited for this
            # cluster" iff its stamp equals the cluster's unique generation id.
            visited_gen = np.zeros((H, W), dtype=np.int32)
            for c_id, cluster in enumerate(solid_clusters):
                local_adj = adjacents[c_id]
                local_grow = grow_paths[c_id]
                target_clust_id = c_id + 1
                gen = c_id + 1                      # strictly increasing → unique

                cluster_arr = np.asarray(cluster)
                visited_gen[cluster_arr[:, 0], cluster_arr[:, 1]] = gen
                frontier = [tuple(p) for p in cluster]

                while frontier:
                    r, c = frontier.pop()
                    for dr, dc in DIRS:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < H and 0 <= nc < W
                                and visited_gen[nr, nc] != gen):
                            visited_gen[nr, nc] = gen
                            if mask1.item((nr, nc)) > 0:
                                neigh_clust = solid_map.item((nr, nc))
                                if neigh_clust != target_clust_id:
                                    if neigh_clust != 0:
                                        local_adj.add(neigh_clust - 1)
                                    else:
                                        frontier.append((nr, nc))
                                        local_grow.add((nr, nc))
                                    
        if timing:
            t_loop.stop("[select] adjacency")
            t_loop.start()

        # ── Depth interpolation for no-reliable-Z clusters ────────────────────
        # A cluster with NO reliable pixels (epipolar-parallel run: the whole
        # disparity valley tied, confidence collapsed) has no trustworthy Z of
        # its own — _cluster_z fell back to the all-pixel median of ambiguous
        # matches, which is exactly the correlated-wrong depth.  The thread is
        # a continuum, so re-derive that Z from the adjacency graph instead:
        # Dijkstra distance along cluster-mean edges ≈ arc length, and inverse-
        # distance weighting of the two nearest reliable anchors ≈ linear
        # interpolation when they bracket the run (thread ends/curves anchor
        # the stereo, so horizontal runs are usually bracketed).  full_conf is
        # NOT raised — optim/EKF keep treating these points as low-confidence
        # measurements — and is additionally DEGRADED by anchor distance
        # (× exp(-d/INTERP_CONF_DECAY)): an interpolated depth is only as good
        # as its anchors are close, so points deep inside an ambiguous run
        # publish near-zero confidence while ones beside an anchor keep theirs.
        # Caveat: at a crossing the graph connects both strands, so an anchor
        # can sit on the other strand — still far closer than a wrong-valley
        # disparity.  Islands with no reachable anchor keep the fallback Z
        # (their confidence is left alone: it is already the lowest tier).
        n_norel = int((~has_rel_z).sum())
        if n_norel and has_rel_z.any():
            nc  = len(full_clusters)
            e_r, e_c, e_w = [], [], []
            for a, nbrs in enumerate(adjacents):
                pa = full_cluster_means[a, :2]
                for b in nbrs:
                    e_r.append(a); e_c.append(b)
                    e_w.append(float(np.linalg.norm(
                        pa - full_cluster_means[b, :2])))
            n_interp = 0
            conf_scale_min = 1.0
            if e_r:
                gmat = csr_matrix((e_w, (e_r, e_c)), shape=(nc, nc))
                gmat = gmat.maximum(gmat.T)
                anchor_idx = np.flatnonzero(has_rel_z)
                gdist = dijkstra(gmat, directed=False, indices=anchor_idx)
                for ci in np.flatnonzero(~has_rel_z):
                    d   = gdist[:, ci]
                    fin = np.flatnonzero(np.isfinite(d))
                    if not len(fin):
                        continue          # island: no reliable anchor reachable
                    near = fin[np.argsort(d[fin])[:2]]
                    w    = 1.0 / (d[near] + 1e-6)
                    full_cluster_means[ci, 2] = float(
                        np.sum(w * full_cluster_means[anchor_idx[near], 2])
                        / np.sum(w))
                    scale = float(np.exp(-float(d[near].min())
                                         / INTERP_CONF_DECAY))
                    full_conf[ci] *= scale
                    conf_scale_min = min(conf_scale_min, scale)
                    n_interp += 1
            # full_conf is the SAME array already published as
            # last_keypt_conf; re-assign anyway so the degradation above never
            # silently detaches if the publication ever becomes a copy.
            self.last_keypt_conf = full_conf
            print(f"[select] depth interpolation: {n_interp}/{n_norel} "
                  f"no-reliable-Z cluster(s) re-depthed from graph anchors "
                  f"(conf degraded by anchor distance, worst "
                  f"×{conf_scale_min:.2f})")

        if timing:
            t_loop.stop("[select] depth interpolation")
            t_loop.start()

        # ── Intersection detection (Optimized) ────────────────────────────────
        intersection_radius  = 20
        direction_var_thresh = 0.2
        ransac_residual      = 4.0
        min_inliers_per_axis = 5
        # A genuine crossing ARM passes THROUGH the crossing centre — it has
        # thread on BOTH sides.  An axis whose inliers all sit on one side of
        # the centre is a thread END / T-stub / spurious line, not a crossing
        # arm, so it is rejected.  Each side must extend at least this far (px)
        # from the centre for the arm to count.
        both_sides_min_px    = 10.0

        def fit_ransac_axes_px(nb_pts, center_pt=None):
            axes = []
            remaining = np.arange(len(nb_pts))
            if center_pt is not None:
                dists = np.linalg.norm(nb_pts - center_pt, axis=1)
                remaining = remaining[dists > ransac_residual * 1.5]
            for _ in range(2):
                if len(remaining) < 2: break
                pts = nb_pts[remaining]
                try:
                    model, inliers = ransac(pts, LineModelND,
                                            min_samples=2,
                                            residual_threshold=ransac_residual,
                                            max_trials=50)
                    inlier_mask = inliers
                except Exception:
                    break
                if inlier_mask is None or inlier_mask.sum() < min_inliers_per_axis:
                    break
                inlier_pts = pts[inlier_mask]
                ic       = inlier_pts.mean(axis=0)
                cen      = inlier_pts - ic
                ev, evec = np.linalg.eigh(cen.T @ cen)
                axes.append((evec[:, np.argmax(ev)], remaining[inlier_mask], ic))
                remaining = remaining[~inlier_mask]
            return axes

        # Intersection detection runs on the reliability-agnostic cluster
        # means (not the dense per-pixel points), so crossings are found even
        # where the thread has low-reliability stereo.  kpts_2d / kd_tree /
        # search_pts and the neighbour lookups below all index this set.
        kpts_2d = full_cluster_means[:, :2]
        kd_tree = cKDTree(kpts_2d)

        # Dense pixels (all segmentation pixels — already reliability-agnostic,
        # since reliability only filtered `clusters`, never dense_pts) are used
        # for robust RANSAC axis fitting around each detected crossing.  The
        # sparse cluster means are great for *detecting* the crossing but have
        # too few points per arm for a stable line fit.  The KD-tree over ALL
        # pixels is expensive, so build it lazily only when a crossing is
        # actually found (see below).
        dense_2d   = dense_pts[:, :2]
        dense_tree = None

        stride = 1 # TODO change if needed
        search_pts = kpts_2d[::stride]
        
        all_neighbours = kd_tree.query_ball_point(
            search_pts, intersection_radius, workers=-1)

        # Vectorised structure-tensor test over ALL search points at once (the
        # per-point Python loop was O(N) and dominated with stride=1).  The
        # computation is identical: for each search point, take its neighbours'
        # unit directions about their centroid, form the 2×2 direction-scatter
        # tensor, and flag points whose eigenvalue ratio exceeds the threshold.
        S = len(all_neighbours)
        counts = np.array([len(n) for n in all_neighbours], dtype=np.int64)
        crossing_candidates = []
        if S and counts.sum():
            owner = np.repeat(np.arange(S), counts)
            flat  = np.concatenate([np.asarray(n, dtype=np.int64)
                                    for n in all_neighbours if len(n)])
            pts   = kpts_2d[flat]                                   # (M, 2)
            cnt   = np.bincount(owner, minlength=S).astype(float)
            cnt_s = np.where(cnt > 0, cnt, 1.0)
            centroid = np.stack([np.bincount(owner, pts[:, 0], minlength=S) / cnt_s,
                                 np.bincount(owner, pts[:, 1], minlength=S) / cnt_s], axis=1)
            vecs  = pts - centroid[owner]
            norms = np.linalg.norm(vecs, axis=1)
            norms[norms < 1e-6] = 1.0
            uv    = vecs / norms[:, None]
            c00 = np.bincount(owner, uv[:, 0] * uv[:, 0], minlength=S) / cnt_s
            c11 = np.bincount(owner, uv[:, 1] * uv[:, 1], minlength=S) / cnt_s
            c01 = np.bincount(owner, uv[:, 0] * uv[:, 1], minlength=S) / cnt_s
            trace = c00 + c11
            det   = c00 * c11 - c01 * c01
            sqrt_val = np.sqrt(np.maximum(0.0, trace ** 2 - 4 * det))
            eig0 = (trace + sqrt_val) / 2.0
            eig1 = (trace - sqrt_val) / 2.0
            is_cand = (cnt >= 4) & (eig1 / (eig0 + 1e-8) > direction_var_thresh)
            # Ignore candidates whose neighbourhood centroid lands on a black
            # (non-mask) pixel: a real crossing sits on the thread, whereas a
            # centroid falling in the gap between two nearby strands is a
            # spurious detection.  centroid is (row, col) in image coords.
            cr = np.clip(np.round(centroid[:, 0]).astype(int), 0, H - 1)
            cc = np.clip(np.round(centroid[:, 1]).astype(int), 0, W - 1)
            on_mask = mask1[cr, cc] > 0
            is_cand = is_cand & on_mask
            crossing_candidates = (np.nonzero(is_cand)[0] * stride).tolist()

        merged_crossings = []
        if crossing_candidates:
            cand_pts = kpts_2d[crossing_candidates]
            if len(crossing_candidates) == 1:
                merged_crossings = [crossing_candidates[0]]
            else:
                cand_tree = cKDTree(cand_pts)
                pairs = cand_tree.query_pairs(r=intersection_radius)
                
                G = nx.Graph()
                G.add_nodes_from(range(len(cand_pts)))
                G.add_edges_from(pairs)
                
                for component in nx.connected_components(G):
                    cluster_indices = list(component)
                    cluster_pts = cand_pts[cluster_indices]
                    centroid_c = cluster_pts.mean(axis=0)
                    
                    dists = np.linalg.norm(cluster_pts - centroid_c, axis=1)
                    closest_local_idx = cluster_indices[np.argmin(dists)]
                    merged_crossings.append(crossing_candidates[closest_local_idx])

        print(f"Intersection detection: {len(merged_crossings)} crossing(s) found")

        # Build the dense-pixel KD-tree only now that we know crossings exist.
        if merged_crossings:
            dense_tree = cKDTree(dense_2d)

        intersection_segments = []
        for ki in merged_crossings:
            center = kpts_2d[ki]
            # Fit the RANSAC axes on the DENSE pixels around the crossing (many
            # points → stable line fit), not on the sparse cluster means.
            dense_nbr = dense_tree.query_ball_point(center, intersection_radius)
            nb_dense = dense_2d[dense_nbr]

            if len(nb_dense) < 4: continue
            axes = fit_ransac_axes_px(nb_dense)

            if len(axes) < 2:
                print(f"  ki={ki}: only {len(axes)} RANSAC axis, skipping.")
                continue
            print(f"  ki={ki}: 2 axes ({len(axes[0][1])} + {len(axes[1][1])} dense-px inliers)")

            axis_segs = []
            for axis_vec, inlier_idx, ic in axes:
                inlier_pts = nb_dense[inlier_idx]
                # ── require the arm to straddle the crossing centre ──────────
                # Project the inliers onto the axis relative to the crossing
                # CENTRE (not the inlier centroid): a real crossing arm has
                # points on both sides, a one-sided "arm" (thread end / stub)
                # does not, so drop it.
                proj_c = (inlier_pts - center) @ axis_vec
                if not (proj_c.min() < -both_sides_min_px and
                        proj_c.max() >  both_sides_min_px):
                    continue
                proj       = (inlier_pts - ic) @ axis_vec
                s_order    = np.argsort(proj)
                min_p, max_p = proj.min(), proj.max()
                axis_segs.append({
                    "axis_vec":       axis_vec,
                    "points":         inlier_pts[s_order],
                    "centroid":       ic,
                    "segment_center": ic + ((min_p + max_p) / 2.0) * axis_vec,
                    "min_proj":       min_p,
                    "max_proj":       max_p,
                    "keypoint_ids":   [],
                })

            # Both arms must straddle the centre for a genuine crossing.
            if len(axis_segs) < 2:
                print(f"  ki={ki}: only {len(axis_segs)} arm(s) straddle the "
                      "centre (need both sides), skipping.")
                continue

            # Assign nearby cluster-mean keypoints (not the dense pixels used
            # for the fit) to the axes; these ids index into kpts_2d.
            kpt_nbr    = kd_tree.query_ball_point(center, intersection_radius)
            nb_kpt_ids = np.array(kpt_nbr)

            if len(axis_segs) >= 2 and len(nb_kpt_ids) > 0:
                nb_kpt_pts = kpts_2d[nb_kpt_ids]
                
                # Point-to-Line Orthogonal Matrix Projection removes O(N*M) cdist dependency 
                axis_min_dists = []
                for seg in axis_segs:
                    vecs = nb_kpt_pts - seg["centroid"]
                    projs = np.dot(vecs, seg["axis_vec"])
                    proj_pts = np.outer(projs, seg["axis_vec"])
                    dists = np.linalg.norm(vecs - proj_pts, axis=1)
                    axis_min_dists.append(dists)
                    
                axis_min_dists = np.column_stack(axis_min_dists)
                assigned = np.argmin(axis_min_dists, axis=1)
                
                for ai, seg in enumerate(axis_segs):
                    ids  = nb_kpt_ids[assigned == ai]
                    proj = (kpts_2d[ids] - seg["centroid"]) @ seg["axis_vec"]
                    seg["keypoint_ids"] = ids[np.argsort(proj)].tolist()

            intersection_segments.append(axis_segs)
            
        if timing:
            t_loop.stop("[select] intersection detect")
            t_loop.start()
        
        # ── Exact line-intersection centroids ─────────────────────────────────
        intersection_centroids = []
        for crossing in intersection_segments:
            found = False
            if len(crossing) >= 2:
                p1, v1 = crossing[0]["centroid"], crossing[0]["axis_vec"]
                p2, v2 = crossing[1]["centroid"], crossing[1]["axis_vec"]
                try:
                    t = np.linalg.solve(np.column_stack((v1, -v2)), p2 - p1)
                    intersection_centroids.append(p1 + t[0] * v1)
                    found = True
                except np.linalg.LinAlgError:
                    pass
            if not found:
                centers = np.vstack([ax["segment_center"] for ax in crossing])
                intersection_centroids.append(centers.mean(axis=0))
        intersection_centroids = np.array(intersection_centroids) \
            if intersection_centroids else np.empty((0, 2))
            
        if timing:
            t_loop.stop("[select] intersection centroid")
            t_loop.start()
        
        # ── Merge nearby crossings ────────────────────────────────────────────
        merge_thresh           = 5.0
        direction_merge_thresh = 0.85
        merged_segments        = []
        used_cross             = np.zeros(len(intersection_segments), dtype=bool)
        merged_centroids       = []
        n_single_axis          = 0    # crossings collapsed to one axis by the merge

        for i in range(len(intersection_segments)):
            if used_cross[i]: continue
            group = [i]; used_cross[i] = True
            for j in range(i+1, len(intersection_segments)):
                if not used_cross[j] and np.linalg.norm(
                        intersection_centroids[i] - intersection_centroids[j]
                ) < merge_thresh:
                    group.append(j); used_cross[j] = True

            pooled = []
            for g in group: pooled.extend(intersection_segments[g])

            final_axes = []; used_ax = np.zeros(len(pooled), dtype=bool)
            for ai in range(len(pooled)):
                if used_ax[ai]: continue
                used_ax[ai] = True
                g_pts  = [pooled[ai]["points"]]
                g_kids = list(pooled[ai].get("keypoint_ids", []))
                v1     = pooled[ai]["axis_vec"]
                for aj in range(ai+1, len(pooled)):
                    if not used_ax[aj]:
                        v2 = pooled[aj]["axis_vec"]
                        if abs(np.dot(v1, v2)) > direction_merge_thresh:
                            g_pts.append(pooled[aj]["points"])
                            g_kids.extend(pooled[aj].get("keypoint_ids", []))
                            used_ax[aj] = True
                            
                combined_pts = np.unique(np.vstack(g_pts), axis=0)
                seen = set()
                dedup_kids = [k for k in g_kids if not (k in seen or seen.add(k))]
                
                ic     = combined_pts.mean(axis=0)
                cen    = combined_pts - ic
                ev, ev_ = np.linalg.eigh(cen.T @ cen)
                av     = ev_[:, np.argmax(ev)]
                proj   = (combined_pts - ic) @ av
                min_p, max_p = proj.min(), proj.max()
                
                if dedup_kids:
                    kid_proj   = (kpts_2d[dedup_kids] - ic) @ av
                    dedup_kids = [dedup_kids[k] for k in np.argsort(kid_proj)]
                    
                final_axes.append({
                    "axis_vec":       av,
                    "points":         combined_pts[np.argsort(proj)],
                    "centroid":       ic,
                    "segment_center": ic + ((min_p + max_p) / 2.0) * av,
                    "min_proj":       min_p,
                    "max_proj":       max_p,
                    "keypoint_ids":   dedup_kids,
                })
            # A crossing needs TWO arms by definition.  The per-crossing loop
            # above already rejects <2 RANSAC axes and <2 straddling arms, but
            # the direction merge here can collapse two near-parallel axes into
            # one — and that single-axis result used to be kept anyway.  It is
            # not a crossing, it is a plain strand, and downstream it does real
            # damage: warm_ordering strikes its keypoints from the match, then
            # rebuilds them along the one axis with no sibling to disambiguate
            # against (the "1 arm(s) skipped" in the crossing t-recovery log),
            # so the safety check that requires a keypoint to be closer to its
            # own arm than to any sibling can never fire.  Drop it instead.
            if len(final_axes) < 2:
                n_single_axis += 1
                continue                      # keeps merged_centroids aligned

            merged_segments.append(final_axes)
            p1, v1 = final_axes[0]["centroid"], final_axes[0]["axis_vec"]
            p2, v2 = final_axes[1]["centroid"], final_axes[1]["axis_vec"]
            try:
                t = np.linalg.solve(np.column_stack((v1, -v2)), p2 - p1)
                merged_centroids.append(p1 + t[0] * v1)
            except np.linalg.LinAlgError:
                centers = np.vstack([ax["segment_center"] for ax in final_axes])
                merged_centroids.append(centers.mean(axis=0))

        intersection_segments = merged_segments
        if n_single_axis:
            print(f"Intersection detection: dropped {n_single_axis} "
                  f"single-axis crossing(s) after the direction merge; "
                  f"{len(intersection_segments)} genuine crossing(s) remain")

        if not speedy and intersection_segments:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(mask1, cmap='gray')
            ax.scatter(kpts_2d[:, 1], kpts_2d[:, 0], c='gray', s=8, alpha=0.5)
            cmap_int  = plt.get_cmap('tab10')
            colors_ax = ['orange', 'cyan', 'magenta', 'yellow', 'lime', 'purple']
            for ii, crossing in enumerate(intersection_segments):
                color    = cmap_int(ii % 10)
                centroid = merged_centroids[ii]
                ax.add_patch(plt.Circle(
                    (centroid[1], centroid[0]), intersection_radius,
                    color=color, fill=False, lw=1.2, alpha=0.6))
                for ai, axis in enumerate(crossing):
                    iv  = axis["axis_vec"]; ic = axis["centroid"]
                    pts = axis["points"]
                    mp, xp = axis["min_proj"], axis["max_proj"]
                    ax.scatter(pts[:, 1], pts[:, 0], c=[color], s=2, alpha=0.3)
                    ax.annotate('',
                        xy=(ic[1]+xp*iv[1], ic[0]+xp*iv[0]),
                        xytext=(ic[1]+mp*iv[1], ic[0]+mp*iv[0]),
                        arrowprops=dict(arrowstyle='<->',
                                        color=colors_ax[ai % len(colors_ax)],
                                        lw=2.0))
                ax.text(centroid[1]+3, centroid[0]-3, f"cross={ii}",
                        color=color, fontsize=7)
                ax.scatter(centroid[1], centroid[0],
                           color='white', edgecolors='black',
                           marker='X', s=80, zorder=5)
            ax.set_title(f"Intersections (RANSAC on dense pixels, radius={intersection_radius}px)")
            plt.tight_layout()
            # savefig so the plot is viewable in non-speedy mode: the Agg
            # backend (set in keypt_ordering) makes plt.show() a no-op.
            plt.savefig("debug_select_intersections.png", dpi=150, bbox_inches='tight')
            plt.show()

        if not speedy:
            # Save the 2-D keypoints and the strided search points as overlay
            # plots (like the intersection plot) rather than raw arrays.
            fig_kp, ax_kp = plt.subplots(1, 1, figsize=(8, 8))
            ax_kp.imshow(mask1, cmap='gray')
            ax_kp.scatter(kpts_2d[:, 1], kpts_2d[:, 0], c='gray', s=8,
                          alpha=0.5, label=f'kpts_2d (n={len(kpts_2d)})')
            ax_kp.scatter(search_pts[:, 1], search_pts[:, 0], marker='+',
                          c='deepskyblue', s=30, lw=0.8,
                          label=f'search pts (stride={stride}, n={len(search_pts)})')
            ax_kp.legend(loc='upper right', fontsize=7)
            ax_kp.set_title("2-D keypoints and strided search points")
            plt.tight_layout()
            plt.savefig("debug_select_kpts_2d.png", dpi=150, bbox_inches='tight')
            plt.close(fig_kp)

        if timing:
            t_loop.stop("[select] intersection merge")
            t_full.stop("select full script")
        
        # slot 4 is now the full (denser) cluster means used for ordering;
        # reliable_flag marks which of them to keep after ordering (→ optim).
        return (img_3D, solid_clusters, solid_map, full_cluster_means,
                grow_paths, adjacents, intersection_segments, dense_pts,
                reliable_flag)