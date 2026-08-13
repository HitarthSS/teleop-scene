import matplotlib.pyplot as plt
import numpy as np
# import scipy.optimize
import scipy.interpolate as interp
from scipy.special import roots_legendre
import pdb 

# optimizing
import scipy.integrate


ROOTS, WEIGHTS = roots_legendre(6)


def arclength(dspline, a, b):
    t = (b - a)/2 * ROOTS + (a + b)/2
    l = (b - a)/2 * \
        np.linalg.norm(dspline(t), axis=-1).dot(WEIGHTS)
    return l 


def refit_spline(spline, keypt_u):
    keypt_u = np.asarray(keypt_u, dtype=float)
    k = spline.k
    t = spline.t

    # The warm keypoint parameters must be finite and span a real range,
    # otherwise the extracted sub-spline is degenerate.
    if not np.all(np.isfinite(keypt_u)) or keypt_u.size < 2:
        raise ValueError(
            f"refit_spline: non-finite / too few warm keypt params "
            f"(size={keypt_u.size}).")

    start = float(np.nanmin(keypt_u))
    end   = float(np.nanmax(keypt_u))
    scale = end - start
    if scale <= 1e-9:
        raise ValueError(
            f"refit_spline: warm keypts span ~zero parameter range "
            f"(start={start:.4g}, end={end:.4g}).")

    # Find the knot span that brackets start and end.
    # searchsorted gives the insertion point; we want the knot interval,
    # so clamp into [k, len(t)-k-1] to stay within the valid B-spline domain.
    i_start = np.clip(np.searchsorted(t, start, side='right') - 1, k, len(t) - k - 1)
    i_end   = np.clip(np.searchsorted(t, end,   side='left'),      k, len(t) - k - 1)

    t_new = t[i_start:i_end + 1]
    t_new = np.concatenate((
        np.full(k, t_new[0]),
        t_new,
        np.full(k, t_new[-1])
    ))
    t_new = (t_new - start) / scale
    c_new = spline.c[i_start - k:i_end]

    # A degree-k B-spline needs at least 2*(k+1) knots. If start/end land in the
    # same knot interval the sub-spline is too short to represent — signal the
    # caller to skip this frame rather than crash inside BSpline().
    if len(t_new) < 2 * (k + 1) or len(c_new) < k + 1:
        raise ValueError(
            f"refit_spline: warm segment too short for degree {k} "
            f"({len(t_new)} knots, {len(c_new)} coeffs).")

    new_spline = interp.BSpline(t_new, c_new, k)
    keypt_s = list((np.array(keypt_u) - start) / scale)

    return new_spline, keypt_s

def reparam(spline, keypt_u):
    INNER_MULT = 1 # multiplier on number control points
    OUTER_MULT = 2 # multiplier to increase number of sampled points
    
    knots, ctrl, k = spline.t, spline.c, spline.k
    
    # --- OPTIMIZATION 1: Vectorized Cumulative Integration ---
    # Create a dense grid of t values to map the curve instantly
    num_samples = max(2000, len(knots) * 50)
    t_dense = np.linspace(knots[0], knots[-1], num_samples)
    
    # Evaluate the derivative across all points in C
    dpts = spline.derivative()(t_dense)
    speeds = np.linalg.norm(dpts, axis=1)
    
    # Fast trapezoidal integration yields the arc length at every t
    s_dense = scipy.integrate.cumulative_trapezoid(speeds, t_dense, initial=0)
    total_l = s_dense[-1]

    # Degenerate (zero-length / non-finite) spline: arc-length normalization is
    # undefined and would emit NaNs that crash BSpline.design_matrix downstream.
    # Signal the caller to skip this frame rather than propagate garbage.
    if not np.isfinite(total_l) or total_l < 1e-9:
        raise ValueError(
            f"reparam: degenerate spline, total arc length={total_l} "
            "(coincident/collinear control points).")

    # Normalize lengths to [0, 1]
    s_dense_norm = s_dense / total_l
    
    # --- OPTIMIZATION 2: Fast Array Interpolation ---
    # Instantly map the given keypt_u (t-values) to their normalized arc-lengths (s-values)
    keypt_s = np.interp(keypt_u, t_dense, s_dense_norm).tolist()
    
    # Calculate m equally spaced points (0.0 to 1.0)
    m = (ctrl.shape[0] - k) * INNER_MULT * OUTER_MULT
    s_spaced_norm = np.linspace(0, 1.0, m)
    
    # Map the spaced s-values BACK to t-values using interpolation
    # This entirely replaces the `bisect` rootfinder loop!
    t_spaced = np.interp(s_spaced_norm, s_dense_norm, t_dense)
    
    # Fit spline to previously collected points
    init_pts = spline(t_spaced)
    init_s = s_spaced_norm 

    # Construct the arc-length-parameterised spline on the SAME knot vector as
    # the input (the QP init guess depends on this fixed knot/control-point
    # structure).  We used to do this with splprep(task=-1), but FITPACK's
    # least-squares-with-fixed-knots hard-fails the Schoenberg-Whitney
    # condition whenever the arc-length samples bunch up — which is exactly
    # what a near-closed (oval) loop produces — raising an opaque internal
    # error (ier=50).  Instead solve the equivalent linear least-squares
    # directly:  find control points c such that  B(init_s) @ c ≈ init_pts,
    # where B is the B-spline design matrix for `knots`.  This degrades
    # gracefully (rank-deficient → minimum-norm solution) instead of throwing,
    # and yields control points on exactly the required knots.
    n_ctrl = len(knots) - k - 1
    # design_matrix requires the abscissae to lie inside the domain; nudge the
    # closing endpoint just below knots[-1].
    x = np.clip(init_s, knots[0], np.nextafter(knots[-1], knots[0]))
    B = interp.BSpline.design_matrix(x, knots, k).toarray()
    new_c, *_ = np.linalg.lstsq(B, init_pts, rcond=None)
    if not np.all(np.isfinite(new_c)):
        raise ValueError(
            f"reparam: least-squares fit produced non-finite control points "
            f"({init_pts.shape[0]} pts, {n_ctrl} ctrl, {len(knots)} knots).")
    new_spline = interp.BSpline(knots, new_c, k)

    return new_spline, knots, keypt_s

def validate_reparam(spline):
    # Integrate curve speed to get segment lengths
    knots, ctrl, k = spline.t, spline.c, spline.k
    segment_l = []
    dspline = spline.derivative()
    samples = np.linspace(knots[0], knots[-1], 150)

    for a, b in zip(samples[:-1], samples[1:]):
        li = arclength(dspline, a, b)
        segment_l.append(li)
    
    # Visualize segments, they should have similar length
    plt.scatter(samples[:-1], segment_l)
    plt.show()


if __name__ == "__main__":
    t = np.linspace(-1, 3)
    num_ctrl = 15
    k = 3
    knots = np.linspace(0, 1, num_ctrl+k+1)
    x1 = np.sin(t)
    x2 = np.cos(t)
    x = np.stack((x1, x2))
    tck, *_ = interp.splprep(x, task=-1, t=knots, k=k)
    t = tck[0]
    c = np.array(tck[1]).T
    k = tck[2]
    tck = interp.BSpline(t, c, k)
    # validate_reparam(tck)
    new_spline = reparam(tck)
    validate_reparam(new_spline)