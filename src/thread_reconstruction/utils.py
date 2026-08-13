import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.optimize
import scipy.integrate
import scipy.interpolate as interp
from thread_reconstruction.reparam import refit_spline

import pdb

def change_coords(pts, cam2img):
    pts[:, 0], pts[:, 1] = pts[:, 1].copy(), pts[:, 0].copy()
    depths = pts[:, 2:].copy()
    pts[:, 2] = np.ones(pts.shape[0])
    pts_c = depths * (np.linalg.inv(cam2img) @ pts.copy().T).T
    return pts_c

"""
Source code here: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
"""
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

"""
Source code here: https://github.com/opencv/opencv/issues/22120
"""
def invert_map(F):
    # shape is (h, w, 2), an "xymap"
    (h, w) = F.shape[:2]
    I = np.zeros_like(F)
    I[:,:,1], I[:,:,0] = np.indices((h, w)) # identity map
    P = np.copy(I)
    for i in range(10):
        correction = I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
        P += correction // 2
    return P

def length_error(ours, gt):
    ours_der = ours.derivative()
    gt_der = gt.derivative()

    def integrand(u, dspline):
        return np.linalg.norm(dspline(u))
    
    ours_len = scipy.integrate.quad(integrand, ours.t[0], ours.t[-1], args=(ours_der))[0]
    gt_len = scipy.integrate.quad(integrand, gt.t[0], gt.t[-1], args=(gt_der))[0]
    return ours_len, gt_len, ours_len - gt_len

def curve_error(ours, gt, num_eval_pts):
    def objective(u, gt_pt):
        return np.linalg.norm(gt_pt - ours(u))
    
    # Find direction of ordering
    to_start = np.linalg.norm(gt(gt.t[0]) - ours(ours.t[0]))
    to_end = np.linalg.norm(gt(gt.t[-1]) - ours(ours.t[0]))
    aligned = True if to_start < to_end else False
    
    errors = np.zeros(num_eval_pts)
    spots = np.zeros(num_eval_pts)
    gt_pts = gt(np.linspace(gt.t[0], gt.t[-1], num_eval_pts))
    slider = ours.t[0] if aligned else ours.t[-1]
    for i, gt_pt in enumerate(gt_pts):
        bounds = [(slider, ours.t[-1]) if aligned else (ours.t[0], slider)]
        res1 = scipy.optimize.shgo(
            objective,
            bounds=bounds,
            args=(gt_pt,)
        )
        res2 = scipy.optimize.differential_evolution(
            objective,
            bounds=bounds,
            args=(gt_pt,)
        )
        if objective(res1.x, gt_pt) < objective(res2.x, gt_pt):
            best = res1.x
        else:
            best = res2.x
        # slider = best
        spots[i] = best
        errors[i] = objective(best, gt_pt)
    return errors, spots, np.mean(errors), np.max(errors)

def reprojection_error(ours, mask, P, num_eval_pts):
    segpix = np.argwhere(mask>0)
    u = np.linspace(ours.t[0], ours.t[-1], num_eval_pts)
    pts = ours(u)
    aug_pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    proj_pts = (P @ aug_pts.T).T
    proj_pts /= proj_pts[:, 2:].copy() + 1e-7
    plt.imshow(mask, cmap="gray")
    plt.scatter(proj_pts[:, 0], proj_pts[:, 1], c="r", s=1)
    plt.show()
    pixs = proj_pts[:, :2]
    pixs[:, 0], pixs[:, 1] = pixs[:, 1].copy(), pixs[:, 0].copy()
    errors = np.zeros(pts.shape[0])
    for i, pix in enumerate(pixs):
        pix = np.expand_dims(pix, 0)
        diffs = np.linalg.norm(pix - segpix, axis=1)
        errors[i] = np.min(diffs)
    return np.mean(errors), np.max(errors)

def reproject_trim(ours, mask, P, num_show_pts=50): # version two using the knots on the spline

    k = ours.k
    start_idx = k-1# start first index after padding
    min_idx = start_idx
    ours.t[k-1] 
    end_idx = ours.c.shape[0]+k-2# start at end before padding
    max_idx = end_idx
    ours.t[-(1+k-1)]
    quit = False
    while True:
        start = ours.t[start_idx]
        end = ours.t[end_idx]
        u = np.linspace(start, end, num_show_pts)
        pts = ours(u)
        aug_pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
        proj_pts = (P @ aug_pts.T).T
        proj_pts /= proj_pts[:, 2:].copy() + 1e-7
        plt.imshow(mask, cmap="gray")
        plt.scatter(proj_pts[:, 0], proj_pts[:, 1], c="r", s=1)
        plt.show()

        user = input(f"ad for end trim, zc for start trim ")
        if user == 'a':
            end_idx -= 1
            if end_idx <=min_idx: # if the end is outside the indexs of the spline's knots
                end_idx = min_idx # first index after padding
        elif user == 'd':
            end_idx += 1
            if end_idx >= max_idx:
                end_idx = max_idx
        elif user == 'z':
            start_idx -= 1
            if start_idx <= min_idx:
                start_idx = min_idx
        elif user == 'c':
            start_idx += 1
            if start_idx >=max_idx:
                start_idx = max_idx
        elif user == 'q':
            quit = True
            break

        start = ours.t[start_idx]
        end = ours.t[end_idx]
            
    spline, _ = refit_spline(ours, [start, end])
    return spline, [start, end]


def reprojection_trim(ours, mask, P, u=None, num_eval_pts=50):
    start = 0
    end = 1
    while True:
        if u is None:
            u = np.linspace(start, end, num_eval_pts)
        else:
            u = np.linspace(start, end, len(u))
        pts = ours(u)
        aug_pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
        proj_pts = (P @ aug_pts.T).T
        proj_pts /= proj_pts[:, 2:].copy() + 1e-7
        plt.imshow(mask, cmap="gray")
        plt.scatter(proj_pts[:, 0], proj_pts[:, 1], c="r", s=1)
        plt.show()

        user = input(f"ad for end trip, zc for start trim ")
        if user == 'a':
            end -= 0.01
            if end <=0:
                end = 0
        elif user == 'aa':
            end -= 0.05
            if end <=0:
                end = 0
        elif user == 'd':
            end += 0.01
            if end >=1:
                end = 1
        elif user == 'dd':
            end += 0.05
            if end >=1:
                end = 1
        elif user == 'z':
            start -= 0.01
            if start <=0:
                start = 0
        elif user == 'zz':
            start -= 0.05
            if start <=0:
                start = 0
        elif user == 'c':
            start += 0.01
            if start >=1:
                start = 1
        elif user == 'cc':
            start += 0.05
            if start >=1:
                start = 1
        elif user == 'q':
            quit = True
            break

    u = np.linspace(start, end, num_eval_pts)
    pts = ours(u)
    tck, _ = interp.splprep(pts.T, u=np.linspace(0, 1, num_eval_pts), k=ours.k, s=0)
    new_spline = interp.BSpline(tck[0], np.array(tck[1]).T, tck[2])

    return new_spline, [start, end]

def augment_keypoints(img1, segpix1, img_3D, keypoints, grow_paths, order):
    # Gather more points between keypoints to get better data for curve initialization
    init_pts = []
    size_thresh = segpix1.shape[0] // 100
    ang_thresh = np.pi/5
    interval_floor = size_thresh // 2
    keypoint_idxs = []
    for key_ord, key_id in enumerate(order[:-1]):
        # Find segmented points between keypoints
        keypoint_idxs.append(len(init_pts))
        init_pts.append(keypoints[key_id])
        curr_growth = grow_paths[key_id]
        next_id = order[key_ord+1]
        next_growth = grow_paths[next_id]
        btwn_pts = curr_growth.intersection(next_growth)

        # gather extra points if keypoint distance is large
        if len(btwn_pts) > size_thresh:
            btwn_pts = np.array(list(btwn_pts))
            btwn_depths = img_3D[btwn_pts[:, 0], btwn_pts[:, 1], 2]
            num_samples = btwn_pts.shape[0] // size_thresh
            # remove outliers
            quartiles = np.percentile(btwn_depths, [25, 75])
            iqr = quartiles[1] - quartiles[0]
            low_clip = quartiles[0]-1.5*iqr < btwn_depths
            up_clip = btwn_depths < quartiles[1]+1.5*iqr
            mask = np.logical_and(low_clip, up_clip)
            mask_idxs = np.squeeze(np.argwhere(mask))
            if mask_idxs.shape[0] < num_samples:
                continue
            filtered_pix = btwn_pts[mask_idxs]
            filtered_depths = btwn_depths[mask_idxs]
            filtered_pts = np.concatenate((filtered_pix, np.expand_dims(filtered_depths, 1)), axis=1)
            
            # project filtered points onto 2D line between keypoints
            p1 = keypoints[key_id, :2]
            p2 = keypoints[next_id, :2]
            p1p2 = p2 - p1
            p1pt = filtered_pix - np.expand_dims(p1, 0)
            p2pt = filtered_pix - np.expand_dims(p2, 0)
            proj1 = np.dot(p1pt, p1p2) / np.linalg.norm(p1p2)
            proj2 = np.dot(p2pt, -1*p1p2) / np.linalg.norm(p1p2)

            # Use angle to prune away more points
            ang1 = np.arccos(proj1 / (np.linalg.norm(p1pt, axis=1)+1e-7))
            ang2 = np.arccos(proj2 / (np.linalg.norm(p2pt, axis=1)+1e-7))
            mask1 = ang1 < ang_thresh
            mask2 = ang2 < ang_thresh
            mask = np.logical_and(mask1, mask2)
            mask_idxs = np.atleast_1d(np.squeeze(np.argwhere(mask)))
            if mask_idxs.shape[0] < num_samples:
                continue
            filtered_pix = filtered_pix[mask_idxs]
            filtered_depths = filtered_depths[mask_idxs]
            filtered_pts = filtered_pts[mask_idxs]
            proj = proj1[mask_idxs]

            # Choose evenly spaced points, based on projections
            pt2ord = np.argsort(proj)
            floor = interval_floor if interval_floor<np.max(proj) else max(np.min(proj), 0)
            intervals = np.linspace(interval_floor, np.max(proj), num_samples)
            int_idx = 0
            for pt_idx in pt2ord:
                if int_idx >= num_samples or \
                    (filtered_pix[pt_idx] == keypoints[next_id, :2]).all():
                    break
                if proj[pt_idx] >= intervals[int_idx]:
                    init_pts.append(filtered_pts[pt_idx])
                    int_idx += 1
    keypoint_idxs.append(len(init_pts))
    init_pts.append(keypoints[order[-1]])
    init_pts = np.array(init_pts)

    return init_pts, keypoint_idxs

def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )

def get_camera2markers_pose(image, P1):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)


    cam_mtx = P1[0:3, 0:3]
    dist = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
    marker_size = 0.0381
    # import pdb; pdb.set_trace()

    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    print(ids)

    if np.all(ids != None):
        for i in range(ids.squeeze(0).shape[0]):
            id = ids.squeeze(0)[i]
            c = corners[i]
            nada, R, t = cv2.solvePnP(marker_points, c, cam_mtx, dist, False, cv2.SOLVEPNP_IPPE_SQUARE)
            t = t.squeeze()
            R = R.squeeze()
            rmat = cv2.Rodrigues(R)[0]

            samples_1d = 3
            offset_1d = np.linspace(-marker_size / 2, marker_size / 2, samples_1d)
            tvec = []
            for x in range(samples_1d):
                for y in range(samples_1d):
                    tvec.append(t + rmat[:, 0]*offset_1d[x] + rmat[:, 1]*offset_1d[y])

            return tvec
