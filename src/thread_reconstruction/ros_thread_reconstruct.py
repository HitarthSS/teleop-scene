import pdb
from collections import deque
from rosbags.image import message_to_cvimage
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import os
import time
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.spatial import cKDTree
from thread_reconstruction.utils import change_coords

# Force the non-interactive Agg backend BEFORE any module imports pyplot.
# The reconstruct pipeline runs in a ROS executor thread, and the interactive
# Tk backend crashes/leaks ("main thread is not in main loop") when its figures
# are destroyed off the main thread. All debug plots savefig(), so Agg is fine
# and plt.show() becomes a harmless no-op.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sensor_msgs.msg import Image, JointState
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from thread_reconstruction.thread_reconstruct import FitEvalClass
from thread_reconstruction.warm_start import WarmStart
from thread_reconstruction.keypt_selection import Select
from thread_reconstruction.keypt_ordering import Order
from thread_reconstruction.optim import Optim, CONSTR_WIDTH_2D
from thread_reconstruction import ekf_params

from thread_reconstruction_msgs.msg import BSpline, ThreadSpecs, ThreadCall, PsmState

# Drop into pdb at every error / degenerate return in this module, so a
# silently-skipped frame stops instead of being swallowed by the caller's
# fallback path.  Export THREAD_RECON_BREAK=0 to disable every breakpoint
# in the package at once (unattended runs, live ROS sessions).
DEBUG_BREAK_ON_ERROR = os.environ.get("THREAD_RECON_BREAK", "0") != "0"


# os.environ['ROS_DOMAIN_ID'] = '100'

"""
~image reconstruction~
run Omar's docker compose
"""

"""
~thread reconstruction~
initialize:
- ros topics subscribing to img1, img2, mask1, mask2, psm1 and psm2 gripper pose, active_psm, thread_reconstruct_call
- ros topics publishing to thread, thread specs, saved pose
- thread methods initialize
- camera matrcies

when reconstruct_call is true:
- grab topics
- check for active psm change 
- grab saved warm-start
- run fit_eval
- publish thread b-spline and specs


"""

"""
~tissue reconstruction~
initialize:
- ros topics subscribe to img1, img2, thread_mask1, gripper_mask1, tissue_reconstruct_call
- ros topics publish to tissue point cloud 
- camera matrices

when reconstruct_call is true:
- grab topics
- reconstruct with saved images
- clean up floating points 
- remove points in masks
- publish point cloud

"""

"""
~manipulation~
initialize:
- ros topics subscribing to thread, thread specs, tissue point cloud, psm1 and psm2 gripper pose, 
- generate goal points, optional visualize?
- publish goal poses over ros
"""

class ROSThread():
    def __init__(self, args):
        rclpy.init()

        self.node = Node('thread_reconstruct')

        self.speedy = args.speedy
        # Warm start no longer DRIVES the ordering.  The cold graph-based
        # ordering runs every frame (it produces the most accurate shape); the
        # warm spline is used only to (a) reject ordered keypoints that sit too
        # far from the previous reconstruction — an outlier gate that removes
        # stray points from poor frames — and (b) match thread direction
        # (match_warm_order).  When False, even the outlier gate is skipped.
        self.use_warm_start = True
        # Master switch for warm-spline mask clipping.  OFF for now — the full
        # mask is passed to selection every frame regardless of the jaw/grasp
        # state below.  Flip to True to re-enable clipping.
        self.CLIP_MASK_ENABLED = False
        # ── EXPERIMENT: grasp-local mask clipping ──────────────────────────────
        # When set (pixels), the warm-spline mask clip keeps ONLY the section of
        # thread within ±this arc length (along the projected thread) of the
        # gripper's grasp point, instead of the whole needle→gripper span.  The
        # clip already only runs when the active PSM's jaw is closed AND the
        # gripper is on the thread (dist_thresh) — i.e. actually grasping;
        # otherwise the full mask is kept.  None → original span behaviour.
        self.CLIP_GRASP_WINDOW_PX = 50
        # Lateral disk radius used in grasp-local mode.  Deliberately smaller
        # than the default 80: disks also extend ALONG the thread, so a large
        # radius would silently widen the ±window (80px radius ≈ ±130px kept).
        self.CLIP_GRASP_RADIUS = 30
        # ── Segmentation blow-out guard ───────────────────────────────────────
        # Skip a frame whose raw mask area exceeds MASK_BLOWUP_RATIO x the
        # median area of the last MASK_BLOWUP_HIST accepted frames — SAM3 has
        # grabbed the instrument shaft or another non-thread object.  10x is
        # deliberately loose: a real thread cannot come close to that between
        # two frames, so this only fires on genuine segmentation failures and
        # never on the thread entering/leaving view.  (Observed blow-out: 34902
        # px raw vs a ~1600 px baseline ≈ 22x.)  Raise to be more permissive;
        # lower toward ~4x to also catch partial shaft grabs.
        self.MASK_BLOWUP_RATIO    = 10.0
        self.MASK_BLOWUP_HIST     = 5     # accepted frames in the median window
        self.MASK_BLOWUP_MIN_HIST = 5     # need a full window before rejecting
        self._mask_px_hist = deque(maxlen=self.MASK_BLOWUP_HIST)
        # ── Stall watchdog ────────────────────────────────────────────────────
        # If this many consecutive reconstruct calls end without an ACCEPTED
        # reconstruction, the pipeline is in a lock-out loop: the warm/EKF
        # spline is stale, the inflated-P χ² gate passes everything (the state
        # is UNCERTAIN, not right), the absolute caps reject almost everything
        # — yet a handful of accidental survivors keeps warm_ordering above
        # its ≥4-match bar, so the cold fallback never fires and nothing can
        # re-acquire (observed: nn px median ~160 for hundreds of frames).
        # At the threshold the warm/EKF ordering is bypassed and the frame is
        # re-acquired COLD from the data; one accepted frame then snaps the
        # EKF back (update_from_thread against a wide-open P) and refreshes
        # the warm history.  Keeps forcing cold every frame until something is
        # accepted; every publish gate still applies, so a bad cold frame
        # cannot slip through.
        self.STALL_FRAMES = 8
        self._skip_streak = 0
        # DETANGLE grasp anchor: {'psm': int, 'dir_local': (3,) unit vector in
        # the GRIPPER's local frame pointing OUT of the gripper along the kept
        # thread segment} while a gripper is holding the thread, else None.
        # Set at grasp onset (toward the ordering's beginning), rotated by the
        # gripper's current orientation each frame, held until that PSM's jaw
        # opens.  See _detangle_clip.
        self._detangle_anchor = None
        # Mode-1 segment memory: camera-frame 3-D positions of the last
        # KEPT detangle segment (see DETANGLE_HYST_*), or None before the
        # first kept selection; plus the count of consecutive frames the held
        # segment has been missing from the eligible groups.
        self._detangle_prev_kept = None
        self._detangle_hyst_miss = 0
        # ── Reprojection consistency gate ─────────────────────────────────────
        # After optim, project the reconstructed 3-D spline back into the left
        # image and require that at least MASK_REPROJ_MIN of its sampled points
        # land on the thread mask (dilated by MASK_REPROJ_TOL_PX to tolerate
        # projection/segmentation error).  A spline that deviates off the thread
        # fails this and is rejected (previous thread kept, nothing published).
        # Was 0.0, which disabled the gate entirely (`frac < 0.0` is never
        # true).  A diverged frame then published a spline with 0.01 on-mask
        # and R=11901 (control points ~500 km from the grasp); that spline
        # became prev_thread and the EKF's update target, and the pipeline
        # never recovered — every later frame matched 0/N keypoints against a
        # prediction 600 px off and died in an infeasible QP.  Healthy frames
        # measure 0.61–1.00 here, so 0.5 sits well clear of the good range and
        # far above the failure.  Raise toward 0.6 to be stricter.
        self.MASK_REPROJ_MIN    = 0.0   # min on-mask fraction to accept
        self.MASK_REPROJ_TOL_PX = 10      # mask dilation (px) before the test
        # Half-width of the published depth envelope in EKF_OUTPUT_MODE
        # ='ekf_thread', in standard deviations of the filter's own posterior
        # (see _ekf_thread_specs).  1.0 = a literal 1σ band.  Raise if
        # downstream consumers were tuned against optim's wider heuristic
        # boxes and now find the bounds too tight.
        self.EKF_SPEC_SIGMA = 1.0
        # Floor on the reliability the bound is divided by, i.e. a cap on how
        # far a low-reliability stretch may widen the published depth band:
        # 0.25 → at most 4x the raw σ.  Needed because gap degradation drives
        # reliability to ~1e-3 over a long unsupported stretch.
        self.EKF_SPEC_REL_FLOOR = 0.25
        # (endpoint reliability taper now lives in FitEvalClass /
        #  thread_reconstruct.py — tune ENDPOINT_TAPER_FRAC / ENDPOINT_MIN_FACTOR
        #  there; applied via self.FitEval.taper_endpoints below)
        # ── Depth-noise debug plot ────────────────────────────────────────────
        # When truthy, save a 3-D plot per reconstruction to debug_z_noise_<frame>.png:
        # reliable cluster means (scatter) + reconstructed thread + warm/previous
        # thread, in (row, col, depth) space.  Lets depth noise/outliers and
        # frame-to-frame drift be seen.
        #
        # OFF by default: it renders up to three matplotlib 3-D figures and
        # PNG-encodes them EVERY frame (~hundreds of ms — dominated the frame
        # budget).  Export THREAD_RECON_VIS_Z=1 to turn it back on for a debug
        # run without editing code.  (Was accidentally pinned always-on by
        # assigning a function object here, which is truthy.)
        self.VIS_Z_NOISE = False # os.environ.get("THREAD_RECON_VIS_Z", "0") != "0"
        self._vis_z_count = 0
        # Synced-message timestamp spread (refreshed in the image/mask
        # callbacks); stamped into the debug_z_noise images to expose left/right
        # sync problems that corrupt stereo depth during motion.
        self._sync_spread   = 0.0  # max-min over the 4 image/mask stamps (s)
        self._sync_lr_img   = 0.0  # |left_img  − right_img|  stamp offset (s)
        self._sync_lr_mask  = 0.0  # |left_mask − right_mask| stamp offset (s)
        self._sync_img_mask = 0.0  # newest image vs newest mask offset (s)
        # ── EKF output / coupling mode ────────────────────────────────────────
        # Tuned in ekf_params.py (EKF_OUTPUT_MODE):
        #   'kf_optim_loop' — Option A predict→fit→correct loop: KF predicts
        #       (no keypoint update), optim fits the raw keypoints with its
        #       robust boxes using the KF prediction as temporal prior, the KF
        #       is corrected from OPTIM'S thread, optim's thread is published.
        #   'ekf_thread'    — the EKF posterior spline replaces optim's fit as
        #       the validated/recorded/published output (optim still runs for
        #       the specs msg); KF updated from the raw matched keypoints.
        #   'optim'         — optim's thread published, KF updated from the
        #       raw matched keypoints.
        # The two flags below are derived from the mode (they were previously
        # independent booleans, with KF_OPTIM_LOOP overriding USE_EKF_THREAD).
        self.KF_OPTIM_LOOP  = ekf_params.EKF_OUTPUT_MODE != 'ekf_thread'
        self.USE_EKF_THREAD = ekf_params.EKF_OUTPUT_MODE == 'ekf_thread'
        # KF-prior weight in optim's QP and its data-trust scaling — see
        # ekf_params.py (KF_PRIOR_LAMBDA / PRIOR_TRUST_*) and the optim call
        # site below for the mechanism.
        self.KF_PRIOR_LAMBDA    = ekf_params.KF_PRIOR_LAMBDA
        self.PRIOR_TRUST_NN_REF = ekf_params.PRIOR_TRUST_NN_REF
        self.PRIOR_TRUST_MIN    = ekf_params.PRIOR_TRUST_MIN
        self.FitEval  = FitEvalClass(args)
        self.Select    = Select(args)
        self.Order     = Order(args)
        self.Optim     = Optim(args)
        self.Warmstart = WarmStart(args)

        self.psm1_current_T = None
        self.psm2_current_T = None
        self.H_cam_base_1 = None
        self.H_cam_base_2 = None
        
        self.prev_thread = None
        self.prev_keypts = None
        self._psm_data_ready = False

        self.img1 = None
        self.img2 = None
        self.mask1 = None
        self.mask2 = None

        self.prev_T = None
        self.thread = None
        self.reliability = None
        self.lower_constr = None
        self.upper_constr = None
        self.keypt_s = None
        self.psm = 1
        self.prev_psm = 1

        # ── Rolling frame history for warm-start selection ────────────────────
        # The last HISTORY_LEN ACCEPTED reconstructions (thread, keypt_s, tool
        # pose, quality stats).  The warm start no longer blindly seeds from the
        # immediately previous frame: a heavily occluded frame yields a
        # truncated / direction-ambiguous reconstruction that would poison the
        # next warm start.  _select_warm_source() instead picks the most recent
        # entry whose coverage matches the best in the window AND whose
        # direction agrees with the window majority — so a few occluded or
        # flipped frames are skipped over, and the thread can't flip direction
        # because one confusing frame became the warm reference.
        self.HISTORY_LEN = 12            # frames kept (5–10 sensible)
        # An entry is "occlusion-degraded" if its spline arc length is below
        # this fraction of the best arc in the window (thread visibly truncated)
        # or its thread-mask pixel count is below this fraction of the best.
        self.HIST_ARC_KEEP_FRAC  = 0.75
        self.HIST_MASK_KEEP_FRAC = 0.60
        # Hard cap on how OLD the warm source may be.  The direction/occlusion
        # consensus above is meant to skip a few bad frames, but its majority is
        # anchored on the OLDEST entry, so a genuine direction change sits at a
        # tie and pins the warm source to an ancient frame for half the window.
        # prev_thread then lags by many frames and prev_T makes |t| accumulate,
        # which blows the warm spline far off the keypoints (NN matches -> 0)
        # and keeps the fit degraded — a self-sustaining trap.  Beyond this many
        # frames of staleness, take the newest and let the fit re-acquire.
        self.WARM_SRC_MAX_AGE = 8
        self.frame_history = deque(maxlen=self.HISTORY_LEN)
        self.frame_idx = 0

        # Latest jaw angle (radians) per PSM, updated by the jaw subscribers.
        # When the jaw is open wider than JAW_OPEN_THRESH the gripper is not
        # holding the thread, so its motion must NOT be applied as a thread
        # transform.  Default to open (thread not held) until a value arrives.
        # A gripper grasping a thread closes onto the thread's THICKNESS, so a
        # holding jaw reads a few degrees, not 0.  Set this above the grasping
        # angle you actually observe in the "jaw closed/OPEN" log, or every
        # frame is zeroed and the thread never follows the tool.
        self.JAW_OPEN_THRESH = np.deg2rad(3.0)   # OPEN when wider than this
        self.psm1_jaw = None
        self.psm2_jaw = None

        self.cam_base_coord_change = np.array([
            [0., 0., -1., 0.],
            [-1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
        ])

        # Offset the gripper point along the gripper's OWN y-axis by 2.5 mm.
        # Applied as a right-multiply of the final gripper transform, so the
        # shift is in the tool's local frame (its y-axis) regardless of the
        # gripper's orientation.  Units: mm (poses are scaled to mm above).
        self.gripper_y_offset_mm = 0
        self.gripper_offset = np.eye(4)
        self.gripper_offset[1, 3] = self.gripper_y_offset_mm

        self.camera_init(args)
        self.init_cam2base(args)

        self.images_init = False
        # 1. Set up Image Subscribers.  Keep every message_filters Subscriber
        # alive on `self` — the image/mask ones are held by self.sync, but the
        # PSM subs are no longer in any synchronizer, so without a reference they
        # get garbage-collected and their subscriptions silently die (→ PSM
        # poses never arrive → reconstruct skips forever).
        left_sub = self._left_sub = Subscriber(self.node, Image, '/stereo/left/rectified_downscaled_image')
        right_sub = self._right_sub = Subscriber(self.node, Image, '/stereo/right/rectified_downscaled_image')
        left_mask_sub = self._left_mask_sub = Subscriber(self.node, Image, '/stereo/left/sam3_image')
        right_mask_sub = self._right_mask_sub = Subscriber(self.node, Image, '/stereo/right/sam3_image')

        # ── PSM pose intake runs on its OWN callback group ────────────────────
        # These were message_filters Subscribers left over from when the poses
        # were part of the synchronizer.  They shared the default (mutually
        # exclusive) callback group with reconstruct_callback, so every ~210 ms
        # reconstruction BLOCKED pose intake: messages queued in the middleware
        # and were delivered ~3 s late.  Measured externally the publisher was
        # perfectly healthy (topic delay 18-26 ms, 30 Hz) while this node's
        # buffer newest stamp sat 3.25 s behind wall clock and captured only
        # ~12.7 Hz — a pure consumer-side backlog.  A stale buffer means every
        # image stamp falls AFTER the newest pose, _psm_T_at snaps instead of
        # interpolating, consecutive frames get the same pose, and the tool
        # motion reads 0.
        #
        # Fix: plain subscriptions in a ReentrantCallbackGroup (drained by a
        # separate executor thread, so a long reconstruct cannot stall them) with
        # a shallow BEST_EFFORT/KEEP_LAST QoS so any transient backlog is DROPPED
        # rather than queued into a delay line.  Freshness beats completeness
        # here — the interpolator only needs poses bracketing the image time.
        self._fast_cbg = ReentrantCallbackGroup()
        fast_qos = QoSProfile(
            depth=20,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )

        # (subscriptions created below, once _track_stamp exists)

        # --- DEBUG TRACKING ---
        self.latest_stamps = {
            'left_img': 0.0, 'right_img': 0.0,
            'left_mask': 0.0, 'right_mask': 0.0,
            'psm1': 0.0, 'psm2': 0.0
        }

        def _track_stamp(name, msg):
            # Convert ROS header stamp to float seconds
            self.latest_stamps[name] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # ── PSM pose subscriptions (see the _fast_cbg comment above) ──────────
        # Count RAW arrivals separately from successful conversions: an
        # exception inside the handler (e.g. H_cam_base_* still None, so
        # _psm_pose_to_T evaluates None @ ...) leaves the history deque empty,
        # which is indistinguishable from "the topic is not publishing" — both
        # surface only as "pose not received yet".  Both counters are reported
        # in that warning so the two cases can never be confused again.
        self._psm_msg_count = {1: 0, 2: 0}    # messages that arrived
        self._psm_conv_fail = {1: 0, 2: 0}    # messages that failed to convert

        def _psm_pose_cb(msg, name, handler, psm_id):
            self._psm_msg_count[psm_id] += 1
            _track_stamp(name, msg)
            try:
                handler(msg)
            except Exception as e:
                self._psm_conv_fail[psm_id] += 1
                if self._psm_conv_fail[psm_id] == 1:
                    H = self.H_cam_base_1 if psm_id == 1 else self.H_cam_base_2
                    self.node.get_logger().error(
                        f"PSM{psm_id} pose callback FAILED ({type(e).__name__}: "
                        f"{e}).  Messages ARE arriving but cannot be converted, "
                        f"so the pose history stays empty.  H_cam_base_{psm_id} "
                        f"is {'None — check --psm_calibrate' if H is None else 'set'}.")

        self._psm1_pose_sub = self.node.create_subscription(
            PoseStamped, '/PSM1/measured_cp',
            lambda m: _psm_pose_cb(m, 'psm1', self.psm1_pose_callback, 1),
            fast_qos, callback_group=self._fast_cbg)
        self._psm2_pose_sub = self.node.create_subscription(
            PoseStamped, '/PSM2/measured_cp',
            lambda m: _psm_pose_cb(m, 'psm2', self.psm2_pose_callback, 2),
            fast_qos, callback_group=self._fast_cbg)

        # Register the tracker to run whenever a message arrives on that topic individually
        left_sub.registerCallback(lambda msg: _track_stamp('left_img', msg))
        right_sub.registerCallback(lambda msg: _track_stamp('right_img', msg))
        left_mask_sub.registerCallback(lambda msg: _track_stamp('left_mask', msg))
        right_mask_sub.registerCallback(lambda msg: _track_stamp('right_mask', msg))
        # psm1/psm2 stamps are tracked inside their own subscription callbacks
        # above (they are no longer message_filters Subscribers).
        # ---------------------------
        # 2. Synchronize the 4 image/mask topics on their HEADER TIMESTAMP, so
        # each reconstruction uses an image + mask from the SAME capture instant.
        # The sam3 masks carry their source-image timestamp but arrive ~3 s later
        # (inference latency), so the queue must be deep enough to keep an image
        # around until its mask catches up — otherwise the image is evicted and
        # the sync pairs a current image with an old mask (→ haywire stereo
        # during motion).  Size SYNC_QUEUE ≈ fps × (mask latency + margin).
        # PSM poses are pulled OUT into cached callbacks (different rate; a few ms
        # of pose staleness is harmless for the tool transform), so they no
        # longer force a loose slop.
        self.SYNC_SLOP  = 0.02   # left/right/mask alignment tolerance (s)
        self.SYNC_QUEUE = 500    # ≈3.5 s buffer @ 60 fps; raise if frames drop
        self.sync = ApproximateTimeSynchronizer(
            [left_sub, right_sub, left_mask_sub, right_mask_sub],
            queue_size=self.SYNC_QUEUE, slop=self.SYNC_SLOP)
        self.sync.registerCallback(self.synced_callback)

        # (PSM pose callbacks are registered on their own subscriptions above,
        # in self._fast_cbg, so they are never blocked by reconstruct_callback.)

        # latest per-stream stamps feed the sync-spread diagnostics stamped into
        # the debug_z_noise images.
        self._img_stamp_l = self._img_stamp_r = 0.0
        self._mask_stamp_l = self._mask_stamp_r = 0.0

        # ── Time-stamped PSM pose history ─────────────────────────────────────
        # The reconstructed image/mask is ~3 s old (sam3 latency), so the motion
        # model must use the tool pose FROM THAT INSTANT, not the current one —
        # otherwise the predicted tool motion doesn't match the thread motion in
        # the delayed image.  Buffer (stamp, T) per PSM and look up the pose at
        # the image timestamp in reconstruct_callback.  maxlen must span the
        # latency at the PSM publish rate (4000 ≈ 20 s @ 200 Hz).
        self.PSM_HIST_LEN = 4000
        self._psm1_hist = deque(maxlen=self.PSM_HIST_LEN)   # (stamp_s, T)
        self._psm2_hist = deque(maxlen=self.PSM_HIST_LEN)
        # Jaw history, same idea as the pose history: the reconstruction runs on
        # a ~3 s-old image (sam3 latency), so the grasp decision must use the
        # jaw angle AT THE IMAGE TIME, not the latest reading — otherwise a jaw
        # opened/closed inside that window flips the zero-motion / mask-clip
        # branches ~3 s early.
        self._psm1_jaw_hist = deque(maxlen=self.PSM_HIST_LEN)  # (stamp_s, jaw_rad)
        self._psm2_jaw_hist = deque(maxlen=self.PSM_HIST_LEN)

        # ── Stale-frame detection ────────────────────────────────────────────
        # reconstruct_callback is triggered EXTERNALLY by /thread/call, while
        # synced_callback only stores the newest synced image+mask set.  If the
        # trigger fires faster than the sync produces new sets (the sync needs a
        # sam3 mask, ~3 s latency), reconstruct re-runs on the SAME frame: the
        # keypoints are identical and frame_stamp is unchanged, so curr_T ==
        # prev_T and the tool motion reads as zero.  Worse, re-running feeds the
        # same measurement into the KF again, double-counting evidence and
        # shrinking P.  Count both streams so the ratio is visible, and skip
        # repeats.  If every trigger does bring a new frame this never fires.
        self._sync_count       = 0      # synced image+mask sets received
        self._last_sync_wall   = None   # wall clock of the last sync
        self._recon_count      = 0      # /thread/call triggers handled
        self._last_recon_stamp = None   # image stamp of the last frame processed
        self._stale_repeats    = 0      # triggers that re-used the same frame
        self.SKIP_STALE_FRAMES = True   # False = process repeats anyway

        # init thread reconstruction call
        # 2. Subscriber for the trigger to run your script
        self.call_sub = self.node.create_subscription(
            ThreadCall, 
            '/thread/call', 
            self.reconstruct_callback, 
            10
        )

        self.psm_sub = self.node.create_subscription(
            PsmState,
            '/manipulate/psm',
            self.psm_state_callback,
            10
        )

        # Jaw angle (dVRK JointState, position[0] = jaw opening in radians).
        # Latest-value subscriptions; jaw need not be time-synced with images.
        # Same fast callback group as the poses: the jaw history is looked up at
        # the image timestamp too, so it must not fall behind while a long
        # reconstruction runs.
        self.psm1_jaw_sub = self.node.create_subscription(
            JointState, '/PSM1/jaw/measured_js',
            lambda msg: self._jaw_callback(1, msg), fast_qos,
            callback_group=self._fast_cbg)
        self.psm2_jaw_sub = self.node.create_subscription(
            JointState, '/PSM2/jaw/measured_js',
            lambda msg: self._jaw_callback(2, msg), fast_qos,
            callback_group=self._fast_cbg)

        # init publishers.  Use a latching (TRANSIENT_LOCAL) QoS so a
        # subscriber that connects after a thread has been published still
        # receives the most recent spline/specs instead of waiting for the next
        # reconstruct call.
        latching_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_thread_specs = self.node.create_publisher(ThreadSpecs, '/thread/specs', latching_qos)
        self.pub_bspline = self.node.create_publisher(BSpline, '/thread/spline', latching_qos)
        # Detected self-intersections, matched onto the published thread.  One
        # ROW PER THREAD PASS, not per crossing: a crossing is where the thread
        # passes over ITSELF, so it appears at two (occasionally more) separate
        # t values, and `crossing_id` is what labels those rows as the SAME
        # physical intersection.  Float32MultiArray rather than a new .msg so
        # nothing outside this package has to be rebuilt — same encoding
        # /stereo/rectified/P1 already uses.  Row layout is in dim[1].label.
        self.pub_intersections = self.node.create_publisher(
            Float32MultiArray, '/thread/intersections', latching_qos)
        self.specs_msg = ThreadSpecs()
        self.spline_msg = BSpline()

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
        

    def init_cam2base(self, args):
        calib = args.psm_calibrate
        if Path(calib).exists():
            data = np.load(calib)

            self.H_cam_base_1 = data['PSM1'].copy()
            self.H_cam_base_1[:3, 3] *= 1000
            self.H_cam_base_2 = data['PSM2'].copy()
            self.H_cam_base_2[:3, 3] *= 1000

    def psm_state_callback(self, psm_msg):
        if psm_msg.psm_id in [1, 2]:
            self.psm = psm_msg.psm_id
            self.node.get_logger().info(f"Main PSM updated to: PSM {self.psm}")

    def _jaw_callback(self, psm_id, msg):
        if not msg.position:
            return
        jaw = float(msg.position[0])
        if psm_id == 1:
            self.psm1_jaw = jaw
            self._psm1_jaw_hist.append((self._msg_stamp(msg), jaw))
        elif psm_id == 2:
            self.psm2_jaw = jaw
            self._psm2_jaw_hist.append((self._msg_stamp(msg), jaw))

    def _jaw_at(self, stamp, psm):
        """Jaw angle (rad) at `stamp` from the timestamped history — nearest
        sample (a jaw open/close is a step, so interpolation buys nothing).
        Falls back to the latest cached value if the history is empty."""
        # Snapshot — the jaw callbacks run on a separate executor thread now
        # (see self._fast_cbg); min() would otherwise iterate a mutating deque.
        hist = list(self._psm1_jaw_hist if psm == 1 else self._psm2_jaw_hist)
        if not hist:
            return self.psm1_jaw if psm == 1 else self.psm2_jaw
        return min(hist, key=lambda sj: abs(sj[0] - stamp))[1]

    def _active_jaw_open(self, stamp=None):
        """True if the active PSM's jaw is open wider than JAW_OPEN_THRESH (so
        it is not gripping the thread).  With `stamp` (the image timestamp) the
        jaw is read from the timestamped history so the grasp decision matches
        the frame being reconstructed (~3 s old), not the current instant.  An
        unknown jaw angle is treated as open (do not apply the tool transform)
        to stay on the safe side."""
        if stamp is not None:
            jaw = self._jaw_at(stamp, self.psm)
        else:
            jaw = self.psm1_jaw if self.psm == 1 else self.psm2_jaw
        if jaw is None:
            return True
        return jaw > self.JAW_OPEN_THRESH

    @staticmethod
    def _msg_stamp(m):
        s = m.header.stamp
        return s.sec + s.nanosec * 1e-9

    def _refresh_sync_diag(self):
        """Recompute the timestamp-spread diagnostics from the latest image and
        mask stamps (shown in the debug_z_noise images).  L-R offsets corrupt
        stereo disparity during motion; the img↔mask offset shows whether the
        masks lag the images (a within-side misalignment stereo can't fix)."""
        stamps = [self._img_stamp_l, self._img_stamp_r,
                  self._mask_stamp_l, self._mask_stamp_r]
        self._sync_spread   = max(stamps) - min(stamps)
        self._sync_lr_img   = abs(self._img_stamp_l - self._img_stamp_r)
        self._sync_lr_mask  = abs(self._mask_stamp_l - self._mask_stamp_r)
        # newest image vs newest mask (how stale one stream is vs the other)
        self._sync_img_mask = abs(max(self._img_stamp_l, self._img_stamp_r)
                                  - max(self._mask_stamp_l, self._mask_stamp_r))

    def synced_callback(self, left_msg, right_msg, left_mask_msg, right_mask_msg):
        """Timestamp-synced stereo image + mask set (all four from the same
        capture instant, despite the ~3 s sam3 mask latency)."""
        self.img1  = message_to_cvimage(left_msg,       'rgb8')
        self.img2  = message_to_cvimage(right_msg,      'rgb8')
        self.mask1 = message_to_cvimage(left_mask_msg,  'mono8')
        self.mask2 = message_to_cvimage(right_mask_msg, 'mono8')
        self._img_stamp_l  = self._msg_stamp(left_msg)
        self._img_stamp_r  = self._msg_stamp(right_msg)
        self._mask_stamp_l = self._msg_stamp(left_mask_msg)
        self._mask_stamp_r = self._msg_stamp(right_mask_msg)
        self._refresh_sync_diag()
        self._sync_count     += 1
        self._last_sync_wall  = time.time()
        if not self.images_init:
            self.node.get_logger().info(
                "Received first timestamp-synced image+mask set...")
            self.images_init = True

    def _psm_pose_to_T(self, pose_msg, H_cam_base):
        pos, ori = pose_msg.pose.position, pose_msg.pose.orientation
        pose = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]
        T = self._pose_to_matrix(pose)
        T[:3, 3] *= 1000
        return H_cam_base @ (T @ self.cam_base_coord_change) @ self.gripper_offset

    def psm1_pose_callback(self, msg):
        T = self._psm_pose_to_T(msg, self.H_cam_base_1)
        self.psm1_current_T = T
        self._psm1_hist.append((self._msg_stamp(msg), T))

    def psm2_pose_callback(self, msg):
        T = self._psm_pose_to_T(msg, self.H_cam_base_2)
        self.psm2_current_T = T
        self._psm2_hist.append((self._msg_stamp(msg), T))

    @staticmethod
    def _interp_pose(s0, T0, s1, T1, stamp):
        """Interpolate a rigid pose at `stamp` between bracketing samples
        (s0,T0) and (s1,T1): linear translation + SLERP rotation."""
        a = (stamp - s0) / (s1 - s0) if s1 > s0 else 0.0
        a = min(1.0, max(0.0, a))
        Tout = np.eye(4)
        Tout[:3, 3] = (1.0 - a) * T0[:3, 3] + a * T1[:3, 3]
        try:
            rots  = R.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]]))
            Tout[:3, :3] = Slerp([0.0, 1.0], rots)([a])[0].as_matrix()
        except Exception:
            Tout[:3, :3] = (T0 if a < 0.5 else T1)[:3, :3]
        return Tout

    def _psm_T_at(self, stamp, psm):
        """Tool pose at `stamp` (the latency-delayed image's timestamp), so the
        motion model matches the thread motion in that image.  Interpolates
        between the two buffered poses that BRACKET `stamp` (poses are sparse at
        20-40 Hz, so nearest-snap can be off by up to a full gap); snaps to the
        nearest end when `stamp` is outside the buffered range.  Returns
        (T, span) where span is the bracket width (s) — the interpolation
        uncertainty — or (None, None) if no pose is buffered."""
        # Snapshot: the pose callbacks now run on a SEPARATE executor thread, so
        # iterating the live deque could raise "deque mutated during iteration".
        # list() on a deque is atomic under the GIL, so this is a safe copy.
        hist = list(self._psm1_hist if psm == 1 else self._psm2_hist)
        if not hist:
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None
        lo = hi = None
        for st, T in hist:                    # time-ordered by arrival
            if st <= stamp:
                lo = (st, T)
            else:
                hi = (st, T)
                break
        # Out-of-range snapping returns the SAME buffer end on every call, so
        # curr_T stops changing and the tool motion collapses to ~0 while the
        # thread visibly moves.  The usual cause is an image/PSM CLOCK MISMATCH
        # (e.g. sim time vs wall clock), which is otherwise silent — so shout.
        if lo is None or hi is None:
            oldest, newest = hist[0][0], hist[-1][0]
            where = "BEFORE the oldest" if lo is None else "AFTER the newest"
            st, T = (hist[0] if lo is None else lo)
            self.node.get_logger().warning(
                f"PSM{psm} pose lookup OUT OF RANGE: image stamp {stamp:.3f} is "
                f"{where} buffered pose (buffer covers {oldest:.3f}..{newest:.3f}, "
                f"{newest - oldest:.1f}s, {len(hist)} samples).  Snapping to a "
                f"buffer end {abs(st - stamp):.3f}s away — the SAME pose will be "
                "returned every frame, so tool motion reads ~0.  Check that the "
                "image and PSM timestamps share a clock (sim time vs wall clock).")
            return T, abs(st - stamp)
        (s0, T0), (s1, T1) = lo, hi
        return self._interp_pose(s0, T0, s1, T1, stamp), (s1 - s0)

    def _pose_to_matrix(self, pose):
        t = pose[:3]
        q = pose[3:]

        R_mat = R.from_quat(q).as_matrix()

        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t

        return T

    def _filter_reliable_order(self, keypoints, order, full_means, reliable_flag,
                               full_conf=None):
        """Filter an ordering down to reliable keypoints.

        Ordering is computed over the dense, reliability-agnostic
        full_cluster_means (`full_means`); `reliable_flag[i]` marks whether
        full mean i is reliable.  `keypoints[order]` is the thread-ordered
        sequence.  Each ordered point is matched back to its full mean (the
        ordered points are drawn from full_means, so this is an exact lookup)
        and dropped if that mean is not reliable.

        Returns (keypoints, order, keypt_conf).  `order` is restricted to the
        reliable points, preserving thread order, so `keypoints[order]` yields
        only reliable points for optim.  `keypt_conf` is aligned with
        `keypoints` (NOT with order), carrying each point's continuous stereo
        confidence from `full_conf`, so optim can index it as keypt_conf[order]
        exactly like keypoints[order].  None when no confidence was supplied.

        The same nearest-neighbour lookup serves both: it is done once here
        because `keypoints` may be the EKF-denoised warm output rather than
        full_means itself, so the mapping is not the identity.
        """
        if reliable_flag is None or len(order) == 0:
            return keypoints, order, None
        keypoints = np.asarray(keypoints)
        order     = np.asarray(order)
        tree      = cKDTree(np.asarray(full_means)[:, :2])
        _, idx    = tree.query(keypoints[:, :2])      # every keypoint, not just ordered
        keep      = np.asarray(reliable_flag)[idx[order]]
        kept      = order[keep].tolist()
        keypt_conf = (np.asarray(full_conf, dtype=float)[idx]
                      if full_conf is not None else None)
        msg = f"reliability filter: kept {len(kept)}/{len(order)} ordered keypoints"
        if keypt_conf is not None and len(kept):
            k = keypt_conf[np.asarray(kept)]
            msg += (f"  (stereo conf of kept: min={k.min():.2f} "
                    f"median={np.median(k):.2f} max={k.max():.2f})")
        print(msg)
        return keypoints, kept, keypt_conf

    # ── Large-gap / depth-outlier gate ────────────────────────────────────────
    # Consecutive ordered keypoints are meant to be neighbours on a smooth
    # curve.  One bad stereo depth breaks that: optim's arc-length
    # reparameterisation turns a single ~25 mm z-jump into a parameter step
    # covering most of the spline (observed dt=0.753 of the domain against a
    # next-largest 0.125), so the QP is asked to satisfy constraint boxes at
    # t=0.21 and t=0.96 with nothing in between and exits primal-infeasible.
    # In that frame the ORDER was fine (1 direction reversal in 39) — the
    # damage was purely depth, which is why the z test comes first.
    #
    # Scales are robust and data-derived (75th percentile of the observed
    # steps) with absolute floors, because the median z-step is frequently
    # exactly 0.00 — a pure multiple-of-median rule would reject everything.
    # Both tests run on the NORMALISED arc-length step that optim_init itself
    # builds — dists/sum(dists) over the points in CAMERA coordinates — so the
    # gate measures the exact quantity that breaks the QP rather than a proxy.
    # An earlier version thresholded z (mm) and pixels separately off
    # percentiles of the same steps it was trying to catch; the outliers
    # inflated those percentiles and the thresholds landed just above them.
    # ── DETANGLE route ────────────────────────────────────────────────────────
    # A deliberately partial mode.  When the thread tangles or overlaps itself,
    # the ordering through the tangle is guesswork, the optim QP goes
    # infeasible, and the whole frame is lost — including the parts that were
    # perfectly well reconstructed.  Detangle gives up on completeness instead:
    # it keeps the single most reliable contiguous run and discards the rest,
    # so a shorter but trustworthy thread is published and, more importantly,
    # becomes the next frame's warm start.  The tracker then follows the clean
    # section through the tangle rather than dying on it.
    #
    # The cost is explicit and permanent: thread outside the kept run is NOT
    # reconstructed this frame, and since the warm start is the clipped thread,
    # it will not come back on its own.  Leave off for normal operation.
    DETANGLE_ENABLED   = False
    DETANGLE_CONF_MIN  = 0.5   # keypoints below this stereo confidence break
                                # a run — inside a tangle both strands match
                                # equally well, which is what low confidence
                                # means here
    DETANGLE_MAX_DT    = 0.04   # a join spanning more than this fraction of the
                                # parameterisation breaks a run.  Tighter than
                                # GATE_MAX_DT (0.25) on purpose: detangle is
                                # meant to be suspicious, not forgiving.
    DETANGLE_MIN_LEN   = 8     # groups shorter than this (total keypoints)
                               # never win.  Keep ABOVE optim's floor: a
                               # 6-point "win" publishes nothing ("only 6
                               # keypoints, not enough") and just feeds the
                               # stall counter.
    # ── bridging: keep MORE than one run across small benign gaps ────────────
    # The gripper occluding the grasped thread splits it into two runs with a
    # short spatial hole between them.  That hole must be bridged — otherwise
    # detangle keeps only one side and the grasped section is half lost.  A
    # TANGLE also produces two nearby runs, so the discriminator is WHY the
    # run broke: an occlusion is missing data (the mask has a hole, nothing
    # was dropped), a tangle is dropped ambiguous points.  Bridge only when
    # at most _MAX_DROPPED low-confidence keypoints fell in the gap AND the
    # endpoints sit within _MAX_DT of the parameterisation.
    DETANGLE_BRIDGE_MAX_DROPPED = 2     # > this many dropped points = tangle
    DETANGLE_BRIDGE_MAX_DT      = 0.15  # > this much arc = not a small hole
                                        # (kept below GATE_MAX_DT so the
                                        # bridged join stays QP-feasible)
    # ── mode-1 segment memory ────────────────────────────────────────────────
    # The segment selected on the INITIAL frame is HELD OUTRIGHT across
    # ungrasped frames: whenever an eligible group overlaps the remembered
    # kept segment (SYMMETRIC overlap ≥ _MIN_OVERLAP: the weaker of group-
    # near-memory and memory-covered-by-group, both within _RADIUS — one-sided
    # precision let a subset fragment score 1.00 and steal the hold), that
    # group wins regardless of score.  Scores are
    # consulted only (a) before anything has been kept, or (b) after the held
    # segment has been MISSING for _LOST_FRAMES consecutive frames — then the
    # score winner is adopted as the new held segment.  Without this the
    # winner was re-elected every frame and per-frame confidence swings
    # (observed medians 0.42→0.98 across five frames) moved the crown between
    # PHYSICAL segments, feeding update_from_thread a different piece of
    # thread each time.  The memory is geometric (3-D positions), so it
    # survives re-ordering and cold re-acquisitions; it refreshes on every
    # authoritative selection — grasp anchors and closed-gripper proximity
    # included, so a released grasp hands its segment to mode 1 seamlessly —
    # but a frame that cannot find the held segment leaves it untouched.
    DETANGLE_HYST_RADIUS      = 10.0
    DETANGLE_HYST_MIN_OVERLAP = 0.3
    DETANGLE_HYST_LOST_FRAMES = 5

    # ── grasp anchor ─────────────────────────────────────────────────────────
    # A grasp within this radius of the thread (camera-frame units, same as
    # clip_mask's dist_thresh "grasping" test) BINDS the detangle anchor to
    # that PSM: from then on the WHOLE group of runs at the gripper is
    # reconstructed with its direction pinned to the noted gripper-frame
    # vector (flipped back if a re-acquire reverses it), until that PSM's
    # jaw opens.  For manipulation the grasped section is the one that
    # matters, and the grasp is the one anchor that survives a visual
    # re-shuffle that moves every conf score.
    DETANGLE_GRASP_RADIUS = 20.0
    # While ANCHORED, if the grasp sits beyond this multiple of the radius
    # from every keypoint (gripper occlusion / a lost frame), selection falls
    # back to score mode FOR THAT FRAME but the anchor is kept — the jaw is
    # still closed, so the hold resumes when tracking returns.  Release
    # happens ONLY when the anchored PSM's jaw opens.
    DETANGLE_ANCHOR_LOST_MULT = 2.0
    # (No refresh rate for the noted outward direction: it is FROZEN in the
    # gripper frame for the whole hold — clamped jaws mean the thread cannot
    # rotate relative to the gripper, and freezing means no run of messy
    # manipulation frames can corrupt it.  Gripper rotation needs no update
    # anyway: the stored vector co-rotates with the tool for free.)

    GATE_MAX_DT      = 0.25   # a single join may not span more than this
                              # fraction of the parameter domain
    GATE_DETOUR_MULT = 4.0    # out-and-back ratio marking a point as a spike
    GATE_DETOUR_FRAC = 0.02   # ...and the wasted length must be at least this
                              # fraction of the whole path, so micro-jitter on
                              # a dense, near-straight run is left alone

    @staticmethod
    def _gate_arc_steps(pts3):
        """Normalised arc-length steps, exactly as optim_init parameterises."""
        d = np.linalg.norm(np.diff(pts3, axis=0), axis=1)
        tot = float(d.sum())
        return (d / tot) if tot > 0 else d

    def _gate_large_gaps(self, keypoints, order, cam2img):
        """Drop out-and-back outliers from an ordering, then split it at any
        join that still spans too much of the parameter domain, keeping the
        longest surviving run.

        Returns the filtered `order`.  Runs shorter than the caller's
        MIN_ORDERED_KPTS are left to that guard to reject.
        """
        order = list(order)
        if len(order) < 3:
            return order

        def cam(idx):
            # change_coords mutates its argument, hence the copy.
            return change_coords(np.asarray(keypoints, dtype=float)[idx].copy(),
                                 cam2img)

        # ── A. out-and-back outliers ──────────────────────────────────────
        # A bad depth (or a stray 2-D point) makes the path detour out and
        # back, so the two legs through it hugely exceed the direct hop from
        # its predecessor to its successor.  That ratio is unit-free, which
        # matters because px and mm are not comparable — the earlier
        # per-axis thresholds were the flaw this replaces.
        pts3 = cam(order)
        d    = np.linalg.norm(np.diff(pts3, axis=0), axis=1)
        total = float(d.sum())
        keep = np.ones(len(order), dtype=bool)
        if total > 0 and len(order) >= 3:
            direct = np.linalg.norm(pts3[2:] - pts3[:-2], axis=1)
            legs   = d[:-1] + d[1:]
            excess = legs - direct
            keep[1:-1] = ~((legs > self.GATE_DETOUR_MULT * np.maximum(direct, 1e-9))
                           & (excess > self.GATE_DETOUR_FRAC * total))
        n_spike = int((~keep).sum())
        order = [o for o, k in zip(order, keep) if k]
        if len(order) < 3:
            if n_spike:
                print(f"large-gap gate: dropped {n_spike} out-and-back "
                      f"outlier(s), {len(order)} keypoints left")
            return order

        # ── B. split where one join still dominates the parameterisation ──
        steps = self._gate_arc_steps(cam(order))
        brk   = np.where(steps > self.GATE_MAX_DT)[0]      # break AFTER index i
        n_before = len(order)
        if brk.size:
            edges = [0, *(brk + 1), len(order)]
            runs  = [order[a:b] for a, b in zip(edges[:-1], edges[1:])]
            worst = float(steps.max())
            order = max(runs, key=len)
            print(f"large-gap gate: dropped {n_spike} outlier(s); split at "
                  f"{brk.size} join(s) spanning up to {worst:.2f} of the "
                  f"parameter domain (> {self.GATE_MAX_DT}) into "
                  f"{len(runs)} run(s), kept longest {len(order)}/{n_before}")
        elif n_spike:
            print(f"large-gap gate: dropped {n_spike} out-and-back outlier(s), "
                  f"{len(order)} keypoints left (max join now "
                  f"{float(steps.max()):.2f} of the domain)")
        return order

    def _detangle_clip(self, keypoints, order, keypt_conf, cam2img,
                       grasp_cands=None):
        """DETANGLE: reduce the ordering to one reliable, QP-feasible segment.

        Three-mode state machine driven by the grasp:

        MODE 1 — no gripper on the thread: keep the most reliable GROUP of
        runs, scored Σ mean_conf(run) × len(run) (groups under
        DETANGLE_MIN_LEN never win) — the original detangle behaviour that
        avoids the overlaps/tangles which make the optim QP infeasible.
        The score elects only the INITIAL segment: once something has been
        kept, the group overlapping it is HELD OUTRIGHT frame after frame
        (see DETANGLE_HYST_*), and re-election happens only after the held
        segment has been missing for DETANGLE_HYST_LOST_FRAMES consecutive
        frames — so the initial frame's choice stays the reconstruction
        target through confidence re-shuffles.

        MODE 2 — grasp ONSET: when a closed jaw sits within
        DETANGLE_GRASP_RADIUS of the thread, the anchor BINDS to that PSM
        (the ACTIVE PSM from /manipulate/psm outranks the other arm when both
        are on the thread — see the trade-off note at the onset loop; distance
        breaks ties among non-active arms only).  The WHOLE group of runs
        containing the grasp is kept — the keypoints matched while the
        gripper was away stay in the reconstruction (the earlier
        one-side-of-the-grasp truncation proved too aggressive).  The side
        toward the BEGINNING of the ordering — the t=0 end of the
        grasp-orientation-locked EKF spline (at least DETANGLE_MIN_LEN long,
        else no onset this frame) — only defines the noted outward unit
        vector (grasp → side centroid) IN THE GRIPPER'S LOCAL FRAME
        (R_gripperᵀ · dir), tying the remembered direction to the tool's
        orientation.

        MODE 3 — held: every frame while the anchored PSM's jaw stays closed,
        keep the whole group of runs containing the grasp (bridged across
        the gripper's own occlusion hole like any missing-data gap).
        Far-away messy sections beyond a tangle break are still never
        entered — GROUP membership is the clip; the grasp no longer
        truncates within the group.  The stored direction, ROTATED BY THE
        CURRENT GRIPPER ORIENTATION (R_g_now · dir_local), enforces the
        ordering's DIRECTION instead: it points from the grasp toward the
        ordering's beginning, and if this frame's group runs the other way
        (e.g. a re-acquired, direction-flipped ordering) the kept order is
        FLIPPED back, so the published direction stays continuous through
        the hold.  Because the direction co-rotates with the tool, it stays
        valid through the occlusions and re-shuffles a manipulation causes.
        The local direction is FROZEN for the whole hold: clamped jaws mean
        the thread cannot rotate relative to the gripper, so the onset
        direction is physically constant in the gripper frame and no run of
        messy frames can corrupt or flip it.  The anchor survives frames
        where the grasp is momentarily far from every keypoint (occlusion;
        score mode for that frame only) and releases ONLY when the anchored
        PSM's jaw opens.

        Run breaking and bridging (shared by all modes): a run is broken by a
        keypoint below DETANGLE_CONF_MIN (overlap ambiguity IS the low
        confidence) or a join over DETANGLE_MAX_DT of the normalised arc
        parameterisation; consecutive runs re-join into a group when at most
        DETANGLE_BRIDGE_MAX_DROPPED points were dropped in the gap AND the
        endpoints sit within DETANGLE_BRIDGE_MAX_DT (missing data, e.g. the
        gripper hole — a tangle drops several ambiguous points and is never
        bridged).

        `grasp_cands` is {psm: (4,4) camera-frame gripper pose} for every PSM
        whose jaw is CLOSED this frame — either arm may hold the thread.  The
        translation is the grasp point; the rotation carries the remembered
        direction through manipulations.

        Returns the clipped order, or the input unchanged when nothing usable
        is found (better a normal frame that may fail than a guaranteed stub).
        """
        grasp_cands = {int(k): np.asarray(v, dtype=float).reshape(4, 4)
                       for k, v in dict(grasp_cands or {}).items()}

        # ── anchor lifecycle: release ONLY on the anchored jaw opening ────
        # Checked before any early return so a release can never be missed on
        # a degenerate frame.
        anchor = self._detangle_anchor
        if anchor is not None and anchor['psm'] not in grasp_cands:
            print(f"detangle: anchor RELEASED (PSM{anchor['psm']} jaw opened)")
            anchor = self._detangle_anchor = None

        order = list(order)
        if len(order) < self.DETANGLE_MIN_LEN:
            return order

        conf = None
        if keypt_conf is not None:
            c = np.asarray(keypt_conf, dtype=float)
            idx = np.asarray(order, dtype=int)
            if idx.size and idx.max() < c.size and idx.min() >= 0:
                conf = c[idx]                       # aligned with `order`
        if conf is None:
            conf = np.ones(len(order))              # no confidence → geometry only

        # Camera-frame geometry, the same parameterisation optim builds
        # (change_coords mutates, hence the copy).
        pts3 = change_coords(
            np.asarray(keypoints, dtype=float)[np.asarray(order, dtype=int)].copy(),
            cam2img)
        d = np.linalg.norm(np.diff(pts3, axis=0), axis=1)
        total = float(d.sum())
        if total <= 0:
            return order
        steps = d / total
        break_after = set(np.where(steps > self.DETANGLE_MAX_DT)[0].tolist())

        # Walk the ordering, cutting at low-confidence points (which are
        # themselves dropped) and at over-long joins (which only break).
        runs, cur = [], []
        for i in range(len(order)):
            if conf[i] < self.DETANGLE_CONF_MIN:
                if cur:
                    runs.append(cur)
                cur = []
                continue
            cur.append(i)
            if i in break_after:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        if not runs:
            print(f"detangle: nothing survived the confidence cut from "
                  f"{len(order)} keypoints; keeping the ordering unchanged")
            return order

        # ── bridge benign gaps into groups ────────────────────────────────
        # `dropped` falls straight out of the run indices: positions between
        # run i's tail and run i+1's head are exactly the low-conf points cut
        # between them (a long-join break leaves zero).  The gap span is the
        # direct endpoint distance — what optim will see once the gap points
        # are gone.
        groups, n_bridge = [[runs[0]]], 0
        for prev, nxt in zip(runs[:-1], runs[1:]):
            dropped = nxt[0] - prev[-1] - 1
            gap_dt  = float(np.linalg.norm(pts3[nxt[0]] - pts3[prev[-1]])) / total
            if (dropped <= self.DETANGLE_BRIDGE_MAX_DROPPED
                    and gap_dt <= self.DETANGLE_BRIDGE_MAX_DT):
                groups[-1].append(nxt)
                n_bridge += 1
            else:
                groups.append([nxt])

        def g_len(g):
            return sum(len(r) for r in g)

        def g_score(g):
            return sum(float(np.mean(conf[r])) * len(r) for r in g)

        eligible = [g for g in groups if g_len(g) >= self.DETANGLE_MIN_LEN]
        if not eligible:
            print(f"detangle: no group of >= {self.DETANGLE_MIN_LEN} keypoints "
                  f"survived ({len(runs)} run(s) in {len(groups)} group(s) "
                  f"from {len(order)} keypoints); keeping the full ordering "
                  f"unchanged")
            return order

        # ── grasp-anchored selection (modes 2 and 3) ──────────────────────
        def _portion_dir(portion, gpos):
            """Unit vector grasp → portion centroid; None if degenerate."""
            v = pts3[portion].mean(axis=0) - gpos
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-9 else None

        def _host_portions(gpos):
            """The group of runs containing the grasp: its FULL ascending
            position list plus the split at the grasp keypoint into the two
            directions the thread leaves the gripper (the split only feeds
            the direction bookkeeping now — the whole group is what gets
            kept).  Returns (grasp→nearest-keypoint dist, idxs, up, down),
            all ascending position lists into `order`."""
            gdist = np.linalg.norm(pts3 - gpos.reshape(1, 3), axis=1)
            i_g   = int(np.argmin(gdist))
            kept_pos  = [i for g in groups for r in g for i in r]
            near_kept = min(kept_pos, key=lambda i: abs(i - i_g))
            host = next(g for g in groups if any(near_kept in r for r in g))
            idxs = sorted(i for r in host for i in r)
            up   = [i for i in idxs if i >= i_g]
            down = [i for i in idxs if i <= i_g]
            return float(gdist[i_g]), idxs, up, down

        kept_idx, desc, tag = None, "", ""
        mem_ok = True      # may this selection refresh the segment memory?
        if anchor is not None:
            # MODE 3 — held: keep the WHOLE group of runs containing the
            # grasp — the keypoints matched while the gripper was away stay
            # in the reconstruction (the earlier one-side-of-the-grasp
            # truncation proved too aggressive).  The stored GRIPPER-FRAME
            # direction (rotated into the camera by the tool's CURRENT
            # orientation) now enforces DIRECTION instead of picking a side:
            # by the mode-2 convention it points from the grasp toward the
            # ordering's BEGINNING, so if this frame's group has its
            # beginning on the opposite side, the kept order is FLIPPED to
            # keep the published direction continuous through the hold.
            # dir_local itself stays FROZEN: with the jaws clamped the
            # thread cannot rotate relative to the gripper, so the onset
            # direction is physically constant in the gripper frame and no
            # run of messy manipulation frames can corrupt it.
            T_g  = grasp_cands[anchor['psm']]
            gpos = T_g[:3, 3]
            R_g  = T_g[:3, :3]
            gmin, idxs, up, down = _host_portions(gpos)
            if gmin <= (self.DETANGLE_ANCHOR_LOST_MULT
                        * self.DETANGLE_GRASP_RADIUS):
                if len(idxs) >= self.DETANGLE_MIN_LEN:
                    dir_cam = R_g @ anchor['dir_local']
                    # Beginning-ward probe: the below-grasp side when it has
                    # substance, else the first third of the group (grasp at
                    # the very start leaves `down` degenerate).
                    probe = (down if len(down) >= 2
                             else idxs[:max(2, len(idxs) // 3)])
                    beg   = _portion_dir(probe, gpos)
                    kept_idx = idxs
                    flip_note = ""
                    if beg is not None and float(beg @ dir_cam) < 0.0:
                        kept_idx  = idxs[::-1]
                        flip_note = ", FLIPPED to stored direction"
                    desc = f"grasp-host group ({len(idxs)} pts{flip_note})"
                    tag  = f", ANCHORED PSM{anchor['psm']}"
                else:
                    tag = ", anchored but host group too short → score mode"
            else:
                tag = (f", anchored but grasp {gmin:.1f} from thread "
                       "(occluded?) → score mode this frame")
        elif grasp_cands:
            # MODE 2 — onset: bind to the nearest on-thread grasp.  The kept
            # side is ALWAYS the one toward the BEGINNING of the ordering;
            # its outward direction is stored in the gripper's local frame.
            # Bind within the GENEROUS ceiling, not the bare radius: the
            # tracked gripper tip sits a near-constant ~14mm off the
            # reconstructed thread (hand-eye offset — the same reason the
            # EKF orientation lock dropped its distance gate), so the tight
            # radius made the anchor go idle exactly while holding.
            bind_ceil = (self.DETANGLE_ANCHOR_LOST_MULT
                         * self.DETANGLE_GRASP_RADIUS)
            # The ACTIVE PSM (/manipulate/psm) outranks proximity: during a
            # trade-off both closed grippers sit on the thread, and a
            # nearest-wins bind could flip between arms across frames.  The
            # topic names the manipulating arm; distance only breaks ties
            # among arms the topic does not name.
            act = getattr(self, 'psm', None)
            best_onset = None
            for psm in sorted(grasp_cands):
                T_g = grasp_cands[psm]
                gmin, idxs, up, down = _host_portions(T_g[:3, 3])
                if gmin > bind_ceil:
                    if not self.speedy:
                        print(f"detangle: PSM{psm} jaw closed but grasp is "
                              f"{gmin:.1f} from the nearest keypoint "
                              f"(> {bind_ceil:.0f}); no anchor")
                    continue
                if len(down) < self.DETANGLE_MIN_LEN:
                    if not self.speedy:
                        print(f"detangle: PSM{psm} grasp on thread but the "
                              f"beginning side has only {len(down)} keypoints "
                              f"(< {self.DETANGLE_MIN_LEN}); no onset")
                    continue
                key = (0 if psm == act else 1, gmin)
                if best_onset is None or key < best_onset[0]:
                    best_onset = (key, psm, T_g, down, idxs)
            if best_onset is not None:
                _, psm, T_g, portion, idxs = best_onset
                dv = _portion_dir(portion, T_g[:3, 3])
                if dv is not None:
                    anchor = self._detangle_anchor = {
                        'psm': psm, 'dir_local': T_g[:3, :3].T @ dv}
                    # The WHOLE host group is kept; the beginning side only
                    # defines the noted direction (used by mode 3's flip and
                    # _orient_order_to_anchor).
                    kept_idx = idxs
                    desc = (f"grasp-host group ({len(idxs)} pts; direction "
                            f"noted toward the beginning)")
                    tag  = f", anchor SET on PSM{psm}"

        if kept_idx is None:
            # MODE 1 — no anchored selection.  A CLOSED jaw still steers it:
            # reconstruction should happen AT the gripper, so if any eligible
            # group sits near a closed gripper in 3-D (within the same
            # generous ceiling the anchor uses), the NEAREST group wins over
            # the global score.  Pure score only when every closed gripper is
            # far from every group (e.g. holding a needle elsewhere).
            near_ceil = (self.DETANGLE_ANCHOR_LOST_MULT
                         * self.DETANGLE_GRASP_RADIUS)
            act   = getattr(self, 'psm', None)
            gnear = None
            for psm, T_g in grasp_cands.items():
                gd = np.linalg.norm(pts3 - T_g[:3, 3].reshape(1, 3), axis=1)
                for g in eligible:
                    dmin = min(float(gd[i]) for r in g for i in r)
                    key  = (0 if psm == act else 1, dmin)
                    if dmin <= near_ceil and (gnear is None
                                              or key < gnear[0]):
                        gnear = (key, g)
            if gnear is not None:
                best = gnear[1]
                desc = (f"group nearest the closed gripper ({gnear[0][1]:.1f} "
                        f"away), score {g_score(best):.1f}")
            else:
                best = max(eligible, key=g_score)
                desc = (f"group of {len(best)} run(s), "
                        f"score {g_score(best):.1f}")
                # ── hold the remembered segment OUTRIGHT ──────────────────
                # See DETANGLE_HYST_*: the group overlapping the held kept
                # segment wins regardless of score — the initial frame's
                # choice stays the reconstruction target until a gripper
                # redefines it or the segment is gone for _LOST_FRAMES.
                if self._detangle_prev_kept is not None:
                    kd_prev = cKDTree(self._detangle_prev_kept)
                    inc, inc_ov = None, 0.0
                    for g in eligible:
                        gi = [i for r in g for i in r]
                        dq, _ = kd_prev.query(
                            pts3[gi], k=1,
                            distance_upper_bound=self.DETANGLE_HYST_RADIUS)
                        prec = float(np.isfinite(dq).mean())
                        # SYMMETRIC overlap.  Precision alone (fraction of the
                        # GROUP near the memory) scores a strict SUBSET of the
                        # held segment 1.00 by construction — observed: a
                        # tangle break split the thread and an 11-point
                        # fragment (all inside the memory) outranked the true
                        # continuation, collapsing the reconstruction onto
                        # itself frame after frame (the memory then refreshes
                        # to the fragment).  Also require the group to COVER
                        # the memory — fraction of the REMEMBERED points
                        # within the radius of this group — and elect on the
                        # weaker of the two.
                        dr, _ = cKDTree(pts3[gi]).query(
                            self._detangle_prev_kept, k=1,
                            distance_upper_bound=self.DETANGLE_HYST_RADIUS)
                        ov = min(prec, float(np.isfinite(dr).mean()))
                        if ov > inc_ov:
                            inc, inc_ov = g, ov
                    if (inc is not None
                            and inc_ov >= self.DETANGLE_HYST_MIN_OVERLAP):
                        if inc is not best:
                            desc = (f"HELD segment (overlap {inc_ov:.2f}, "
                                    f"score {g_score(inc):.1f} vs best "
                                    f"{g_score(best):.1f})")
                            best = inc
                    else:
                        # Held segment not found this frame: publish the
                        # score winner but DO NOT adopt it as the new held
                        # segment until the loss persists.
                        self._detangle_hyst_miss += 1
                        if (self._detangle_hyst_miss
                                < self.DETANGLE_HYST_LOST_FRAMES):
                            mem_ok = False
                            desc += (f" [held segment missing "
                                     f"{self._detangle_hyst_miss}/"
                                     f"{self.DETANGLE_HYST_LOST_FRAMES}, "
                                     f"memory kept]")
                        else:
                            desc += " [held segment lost — adopting this one]"
            kept_idx = [i for r in best for i in r]

        kept = [order[i] for i in kept_idx]
        # Segment memory: geometric positions of the kept segment.  Refreshed
        # by every authoritative selection (anchored/onset/gripper-proximity/
        # held/adopted) so a released grasp hands its segment to mode 1;
        # frames that could not find the held segment leave it untouched
        # (mem_ok False), and unchanged-fallback frames never reach here.
        if mem_ok:
            self._detangle_prev_kept = pts3[np.asarray(kept_idx,
                                                       dtype=int)].copy()
            self._detangle_hyst_miss = 0
        print(f"detangle: kept {len(kept)}/{len(order)} keypoints — {desc} "
              f"({len(runs)} run(s) → {len(groups)} group(s), "
              f"{n_bridge} gap(s) bridged){tag}")
        return kept

    def _orient_order_to_anchor(self, keypoints, order, grasp_cands):
        """Keep the HELD thread direction through a cold re-acquisition.

        A cold keypt_ordering starts from scratch, so its t-direction is
        arbitrary — the classic failure is the cold fallback coming back
        FLIPPED, which then pins the warm source and fights the EKF's
        direction lock.  While a detangle anchor is held, the gripper itself
        remembers the direction: the noted outward vector (R_g · dir_local)
        points from the grasp toward the ordering's BEGINNING (the mode-2
        onset convention, frozen for the whole hold).  If a fresh order has
        it pointing toward the END instead, the order is flipped before
        anything downstream (detangle, optim, the EKF correction) sees it.

        No-op without an anchor, without the anchored PSM's pose, or when the
        grasp is far from every ordered keypoint (nothing to judge against).
        """
        anchor = self._detangle_anchor
        if anchor is None or order is None or len(order) < 2:
            return order
        T_g = grasp_cands.get(anchor['psm'])
        if T_g is None:
            return order
        T_g  = np.asarray(T_g, dtype=float)
        pts3 = change_coords(
            np.asarray(keypoints, dtype=float)[np.asarray(order, int)].copy(),
            self.cam2img1)
        gpos = T_g[:3, 3]
        gd   = np.linalg.norm(pts3 - gpos.reshape(1, 3), axis=1)
        if float(gd.min()) > (self.DETANGLE_ANCHOR_LOST_MULT
                              * self.DETANGLE_GRASP_RADIUS):
            return order
        i_g     = int(np.argmin(gd))
        dir_cam = T_g[:3, :3] @ np.asarray(anchor['dir_local'], dtype=float)
        # Which side of the grasp does the noted outward direction select?
        # Compare side centroids (same measure the anchor was noted from); a
        # side the grasp keypoint terminates has nothing to compare.
        s_beg = (float((pts3[:i_g + 1].mean(axis=0) - gpos) @ dir_cam)
                 if i_g >= 1 else -np.inf)
        s_end = (float((pts3[i_g:].mean(axis=0) - gpos) @ dir_cam)
                 if i_g <= len(pts3) - 2 else -np.inf)
        if s_end > s_beg:
            print("ANCHOR DIRECTION: cold ordering came back with the held "
                  "outward side toward the END; flipping the order to keep "
                  "the grasp-noted thread direction.")
            return list(order)[::-1]
        return order

    def _project_to_row_col_depth(self, thread, P, n=200):
        """Sample a 3-D spline and project to (row, col, depth) to match the
        cluster-mean space: row/col are the pinhole image coords, depth is the
        camera-frame Z.  Returns (rows, cols, depths) each (n,)."""
        us    = np.linspace(0.0, 1.0, n)
        p3    = np.asarray(thread(us))
        aug   = np.concatenate([p3, np.ones((len(p3), 1))], axis=1)
        proj  = (P @ aug.T).T
        proj  = proj / (proj[:, 2:3] + 1e-7)
        rows  = proj[:, 1]
        cols  = proj[:, 0]
        depth = p3[:, 2]
        return rows, cols, depth

    def _measure_lag(self, means, threads, tip_prev, tip_curr, P):
        """Median signed offset of each thread from the observed keypoint cloud,
        measured IN THE IMAGE PLANE (row, col px) and projected onto the tool's
        motion direction.

        Sign convention: POSITIVE = the keypoints are ahead of the thread along
        the direction the tool moved, i.e. the thread LAGS.  Comparing the
        warped (predicted) thread against the final optim thread separates the
        two possible causes: if warm lags but optim does not, only the predictor
        under-shoots (the fit still corrects it); if optim lags too, the output
        itself is trailing the data and the prior is over-weighted.

        Returns {name: lag_px}, plus 'motion_px' = the tool's image-plane motion
        for scale (a lag ≈ motion_px means a full one-frame lag).
        """
        m = np.asarray(means, float)
        if len(m) < 3 or tip_prev is None or tip_curr is None:
            return {}
        # Tool motion in the image plane, from the projected gripper tip.
        rp, cp, _ = self._project_pts_row_col_depth(tip_prev, P)
        rc, cc, _ = self._project_pts_row_col_depth(tip_curr, P)
        mv = np.array([rc[0] - rp[0], cc[0] - cp[0]])
        nmv = float(np.linalg.norm(mv))
        if nmv < 1e-6:
            return {}
        u = mv / nmv
        out = {'motion_px': nmv}
        for name, th in threads.items():
            if th is None:
                continue
            try:
                r, c, _ = self._project_to_row_col_depth(th, P, n=200)
                p2 = np.column_stack([r, c])
                # nearest thread sample per keypoint, offset keypoint - thread
                d   = np.linalg.norm(m[:, None, :2] - p2[None, :, :], axis=2)
                off = m[:, :2] - p2[d.argmin(axis=1)]
                out[name] = float(np.median(off @ u))
            except Exception:
                continue
        return out

    @staticmethod
    def _project_pts_row_col_depth(pts3, P):
        """Project discrete 3-D camera-frame points to the plot's
        (row, col, depth) space — same convention as
        _project_to_row_col_depth, but for an explicit point set (used for the
        gripper origin and its rotation-axis endpoints)."""
        p3   = np.atleast_2d(np.asarray(pts3, dtype=float))
        aug  = np.concatenate([p3, np.ones((len(p3), 1))], axis=1)
        proj = (P @ aug.T).T
        proj = proj / (proj[:, 2:3] + 1e-7)
        return proj[:, 1], proj[:, 0], p3[:, 2]

    @staticmethod
    def _constr_box_faces(rc, cc, zlo, zhi, w):
        """6 quad faces of the axis-aligned constraint cuboid centred at image
        (row=rc, col=cc) with lateral half-width w and depth span [zlo, zhi].
        Axes match the plot: x=row, y=col, z=depth."""
        x0, x1 = rc - w, rc + w
        y0, y1 = cc - w, cc + w
        z0, z1 = zlo, zhi
        v = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
             [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
        return [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
                [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
                [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]]

    def _draw_gripper_overlay(self, ax, warm_thread, curr_T, prev_T,
                              optim_thread, P):
        """Draw the transformed (motion-warped) previous thread, the gripper
        pose that produced the warp (x/y/z triad + tip), and the tool
        translation onto a 3-D axis.  Returns the 'tool: |t|=.. rot=..' string
        for the title.  Shared by both debug_z_noise figures so the gripper /
        motion context appears on each."""
        # transformed (warped) previous thread — dashed orange
        if warm_thread is not None:
            r, c, d = self._project_to_row_col_depth(warm_thread, P)
            ax.plot(r, c, d, c='tab:orange', lw=1.5, ls='--',
                    label='transformed (warped) thread')

        tool_txt = ''
        if curr_T is not None:
            tip = np.asarray(curr_T)[:3, 3]
            axis_len = 15.0                     # mm, gripper triad arm length
            ends = np.array([tip + np.asarray(curr_T)[:3, i] * axis_len
                             for i in range(3)])
            r0, c0, d0 = self._project_pts_row_col_depth(tip,  P)
            re, ce, de = self._project_pts_row_col_depth(ends, P)
            for i, (col, lab) in enumerate(zip(
                    ['red', 'lime', 'deepskyblue'],
                    ['gripper x', 'gripper y', 'gripper z'])):
                ax.plot([r0[0], re[i]], [c0[0], ce[i]], [d0[0], de[i]],
                        c=col, lw=2.5, label=lab)
            ax.scatter(r0, c0, d0, c='k', s=70, marker='*',
                       depthshade=False, label='gripper tip')

            # The translation actually applied this frame: prev tip -> curr tip
            if prev_T is not None:
                pt = np.asarray(prev_T)[:3, 3]
                rp, cp, dp = self._project_pts_row_col_depth(pt, P)
                ax.plot([rp[0], r0[0]], [cp[0], c0[0]], [dp[0], d0[0]],
                        c='k', lw=1.5, ls='--', label='tool translation')
                dt   = float(np.linalg.norm(tip - pt))
                Rrel = np.asarray(curr_T)[:3, :3] @ np.asarray(prev_T)[:3, :3].T
                ang  = float(np.degrees(np.arccos(
                    np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0))))
                tool_txt = f'\ntool: |t|={dt:.2f}mm  rot={ang:.2f}deg'
                # 3-D grasp->thread distance drives the warp weight
                # w=exp(-d/deform_radius).  A grasp that looks right in the
                # 2-D projection can still be far in DEPTH, which silently
                # zeroes the warp — so report it next to the motion.
                if optim_thread is not None:
                    try:
                        p3 = np.asarray(optim_thread(np.linspace(0, 1, 100)))
                        d_grasp = float(np.linalg.norm(p3 - tip, axis=1).min())
                        tool_txt += f'  grasp->thread={d_grasp:.1f}mm'
                    except Exception:
                        pass
        return tool_txt

    def _vis_z_noise(self, full_means, reliable_flag, lower_constr, upper_constr,
                     reliability, ekf_warm, optim_thread, reseeded, P,
                     max_boxes=40, warm_thread=None, curr_T=None, prev_T=None,
                     masks=None, dbg1=None):
        """Save a 3-D plot per reconstruction: the cluster means (scatter,
        reliable vs unreliable) + the optim constraint boxes (±CONSTR_WIDTH_2D
        px laterally, [lower, upper] depth span, coloured by reliability) + the
        three reference curves — EKF warm-start (this frame's EKF posterior),
        optim warm thread (previous reconstruction fed to optim), and optim
        thread (this frame's QP fit) — in (row, col, depth) space.  x-axis =
        image row [0, 480], y-axis = image col [0, 640], z-axis = depth
        [100, 160].  The title flags frames where the EKF was re-seeded from the
        raw warm spline.  Gated by self.VIS_Z_NOISE."""
        try:
            fig = plt.figure(figsize=(9, 7))
            ax  = fig.add_subplot(projection='3d')

            # ── reliability-scaled constraint boxes ────────────────────────────
            # lower_constr/upper_constr are (N, 3) = (X, Y, depth-bound) sampled
            # along the thread; their [:,2] are the reliability-scaled depth
            # bounds.  Box centre (row, col) comes from projecting the thread at
            # the matching parameters; the depth span is [lower, upper]; colour
            # encodes reliability (as optim derives it from the box height).
            lc = np.asarray(lower_constr) if lower_constr is not None else None
            uc = np.asarray(upper_constr) if upper_constr is not None else None
            rel = np.asarray(reliability).ravel() if reliability is not None else None
            if (optim_thread is not None and lc is not None and uc is not None
                    and len(lc) and rel is not None):
                N = len(lc)
                rows, cols, _ = self._project_to_row_col_depth(optim_thread, P, n=N)
                stride = max(1, N // max_boxes)
                idxs   = range(0, N, stride)
                norm   = Normalize(vmin=0.0, vmax=1.0)
                cmap   = cm.viridis
                faces, facecolors = [], []
                for i in idxs:
                    faces.extend(self._constr_box_faces(
                        rows[i], cols[i], lc[i, 2], uc[i, 2], CONSTR_WIDTH_2D))
                    facecolors.extend([cmap(norm(rel[i]))] * 6)
                pc = Poly3DCollection(faces, facecolors=facecolors,
                                      edgecolors='k', linewidths=0.2, alpha=0.10)
                ax.add_collection3d(pc)
                sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
                fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1, label='reliability')

            # cluster means (row, col, depth): reliable = blue dots, unreliable
            # = red triangles.  These are the raw stereo points behind the
            # boxes — their z scatter is the depth noise the boxes bound.
            if full_means is not None and len(full_means):
                fm  = np.asarray(full_means)
                rf  = (np.asarray(reliable_flag, dtype=bool)
                       if reliable_flag is not None
                       else np.ones(len(fm), dtype=bool))
                rm, um = fm[rf], fm[~rf]
                if len(rm):
                    ax.scatter(rm[:, 0], rm[:, 1], rm[:, 2], c='tab:blue', s=10,
                               depthshade=False, label=f'reliable means ({len(rm)})')
                if len(um):
                    ax.scatter(um[:, 0], um[:, 1], um[:, 2], c='tab:red', s=10,
                               marker='^', depthshade=False,
                               label=f'unreliable means ({len(um)})')

            # optim thread (this frame's QP fit) — solid green
            if optim_thread is not None:
                r, c, d = self._project_to_row_col_depth(optim_thread, P)
                ax.plot(r, c, d, c='tab:green', lw=2.0, label='optim thread')

            # # optim warm thread (previous reconstruction fed to optim) — dashed orange
            # if optim_warm is not None:
            #     r, c, d = self._project_to_row_col_depth(optim_warm, P)
            #     ax.plot(r, c, d, c='tab:orange', lw=1.5, ls='--',
            #             label='optim warm thread')

            # EKF warm-start (this frame's EKF posterior spline) — dotted purple
            if ekf_warm is not None:
                r, c, d = self._project_to_row_col_depth(ekf_warm, P)
                ax.plot(r, c, d, c='tab:purple', lw=2.0, ls=':',
                        label='EKF warm start')

            # transformed (warped) thread + the gripper pose that produced the
            # warp + tool translation; returns the 'tool: |t|.. rot..' readout.
            tool_txt = self._draw_gripper_overlay(
                ax, warm_thread, curr_T, prev_T, optim_thread, P)

            ax.set_xlabel('image y (row)')
            ax.set_ylabel('image x (col)')
            ax.set_zlabel('z (depth)')
            ax.set_xlim(0, 480)
            ax.set_ylim(0, 640)
            ax.set_zlim(100, 160)
            # Spin the camera about the vertical (z) axis by 3° per frame.  The
            # 3-D view orbits the centre of the axis box, which with these fixed
            # limits is exactly (x=240, y=320) — so the scene rotates about that
            # point.  Wrap at 360° so the azimuth stays bounded.
            ax.view_init(elev=25, azim=(60 + 3 * self._vis_z_count) % 360)
            reseed_tag = '  [EKF RESEEDED]' if reseeded else ''
            sync_str = (f"L-R img={self._sync_lr_img*1e3:.0f}ms  "
                        f"L-R mask={self._sync_lr_mask*1e3:.0f}ms  "
                        f"img-mask={self._sync_img_mask*1e3:.0f}ms  "
                        f"spread={self._sync_spread*1e3:.0f}ms")
            ax.set_title(f'reliable means + threads  '
                         f'(frame {self._vis_z_count}){reseed_tag}\n{sync_str}'
                         f'{tool_txt}',
                         color=('tab:red' if reseeded else 'black'))
            ax.legend(fontsize=8)
            plt.tight_layout()
            # reseeded frames also get a distinct filename suffix so they're
            # easy to find/scrub to among the sequence.
            suffix = "_reseed" if reseeded else ""
            plt.savefig(f"debug_z_noise_{self._vis_z_count:04d}{suffix}.png",
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
            if reseeded:
                self.node.get_logger().info(
                    f"VIS_Z_NOISE: frame {self._vis_z_count} EKF was re-seeded.")

            # ── Raw-keypoints-only view → debug_z_noise.png (overwritten every
            # frame, so it's a stable "live" file to watch) ────────────────────
            fig2 = plt.figure(figsize=(9, 7))
            ax2  = fig2.add_subplot(projection='3d')
            if full_means is not None and len(full_means):
                fm = np.asarray(full_means)
                rf = (np.asarray(reliable_flag, dtype=bool)
                      if reliable_flag is not None
                      else np.ones(len(fm), dtype=bool))
                rm, um = fm[rf], fm[~rf]
                if len(rm):
                    ax2.scatter(rm[:, 0], rm[:, 1], rm[:, 2], c='tab:blue', s=12,
                                depthshade=False, label=f'reliable ({len(rm)})')
                if len(um):
                    ax2.scatter(um[:, 0], um[:, 1], um[:, 2], c='tab:red', s=12,
                                marker='^', depthshade=False,
                                label=f'unreliable ({len(um)})')
            # Same gripper x/y/z triad, tip, tool translation, transformed warped
            # thread and 'tool: |t|.. rot..' readout as the main figure, so the
            # live-overwritten raw view carries the motion context too.
            tool_txt2 = self._draw_gripper_overlay(
                ax2, warm_thread, curr_T, prev_T, optim_thread, P)
            ax2.set_xlabel('image y (row)')
            ax2.set_ylabel('image x (col)')
            ax2.set_zlabel('z (depth)')
            ax2.set_xlim(0, 480)
            ax2.set_ylim(0, 640)
            ax2.set_zlim(100, 160)
            ax2.view_init(elev=25, azim=(60+3 * self._vis_z_count) % 360)
            ax2.set_title(f'raw keypoints  (frame {self._vis_z_count})\n'
                          f'{sync_str}{tool_txt2}')
            ax2.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig("debug_z_noise.png", dpi=150, bbox_inches='tight')
            plt.close(fig2)

            # ── Masks used this frame → debug_z_noise_<n>_masks.png ────────────
            # The exact mask images the reconstruction consumed (left pre-clip,
            # left as-fed-to-selection, right), so a bad frame can be traced back
            # to the segmentation rather than the fit.
            # if masks:
            #     items = [(k, v) for k, v in masks.items() if v is not None]
            #     if items:
            #         figm, axm = plt.subplots(1, len(items),
            #                                  figsize=(5.5 * len(items), 5),
            #                                  squeeze=False)
            #         for a, (name, m) in zip(axm[0], items):
            #             m = np.asarray(m)
            #             if m.ndim == 3:
            #                 m = m[..., 0]
            #             a.imshow(m, cmap='gray')
            #             a.set_title(f'{name}  ({m.shape[0]}x{m.shape[1]}, '
            #                         f'{int((m > 0).sum())} px)', fontsize=9)
            #             a.axis('off')
            #         figm.suptitle(f'masks  (frame {self._vis_z_count})\n{sync_str}',
            #                       fontsize=10)
            #         plt.tight_layout()
            #         plt.savefig(f"debug_z_noise_{self._vis_z_count:04d}_masks.png",
            #                     dpi=150, bbox_inches='tight')
            #         plt.close(figm)

            # ── Matched keypoints (DEBUG 1 style) → ..._matched.png ────────────
            # Same view keypt_ordering's DEBUG 1 draws: warm spline projection,
            # matched keypoints coloured by their t-parameter, distance-rejected
            # and intersection-excluded points, and the keypoint->spline links.
            if dbg1:
                kp        = np.asarray(dbg1['keypoints'])
                pp        = np.asarray(dbg1['proj_pts'])
                mm        = np.asarray(dbg1['matched_mask'], dtype=bool)
                mt        = np.asarray(dbg1['matched_t'])
                bwi       = np.asarray(dbg1['best_warm_idx'])
                excl_ids  = [i for i in dbg1['crossing_kpt_ids'] if i < len(mm)]
                figk, axk = plt.subplots(figsize=(11, 8))
                if dbg1.get('mask') is not None:
                    dm = np.asarray(dbg1['mask'])
                    if dm.ndim == 3:
                        dm = dm[..., 0]
                    axk.imshow(dm, cmap='gray')

                axk.plot(pp[:, 1], pp[:, 0], c='red', lw=1.2, alpha=0.5,
                         label='warm spline (2D)')
                axk.scatter(pp[0, 1],  pp[0, 0],  c='red', s=80, marker='^',
                            zorder=6, label='t=0')
                axk.scatter(pp[-1, 1], pp[-1, 0], c='red', s=80, marker='v',
                            zorder=6, label='t=1')

                dist_fail = np.where(~mm)[0]
                matched   = np.where(mm)[0]
                if len(dist_fail):
                    axk.scatter(kp[dist_fail, 1], kp[dist_fail, 0], c='gray',
                                s=20, marker='D', alpha=0.6, zorder=3,
                                label=f'dist-rejected ({len(dist_fail)})')
                if excl_ids:
                    axk.scatter(kp[excl_ids, 1], kp[excl_ids, 0], c='red', s=40,
                                marker='x', linewidths=2, zorder=5,
                                label=f'excl. intersection ({len(excl_ids)})')
                if len(matched):
                    sck = axk.scatter(kp[matched, 1], kp[matched, 0],
                                      c=mt[matched], cmap='plasma', vmin=0, vmax=1,
                                      s=35, zorder=4, edgecolors='white',
                                      linewidths=0.5,
                                      label=f'matched ({len(matched)})')
                    figk.colorbar(sck, ax=axk, fraction=0.03, pad=0.02,
                                  label='t-param')
                    for ki in matched:
                        wi = int(bwi[ki])
                        axk.plot([kp[ki, 1], pp[wi, 1]], [kp[ki, 0], pp[wi, 0]],
                                 c='pink', lw=0.7, alpha=0.6)

                # EKF reseed/divergence-recovery status + the values that
                # trigger it (gate pass-rate, low-gate streak, motion — each
                # vs its threshold), captured at the recovery check this frame.
                rec = getattr(getattr(self.Order, 'ekf', None),
                              '_last_recovery_info', None)
                ori = getattr(getattr(self.Order, 'ekf', None),
                              '_last_orient_info', None)
                if rec is not None:
                    fired = rec['fired']
                    flipped   = bool(ori['flipped'])   if ori is not None else False
                    corrected = bool(ori.get('corrected')) if ori is not None else False
                    # `corrected` (the grasp lock reversed the state) must show
                    # even when the endpoint detector didn't fire — a lock-only
                    # reversal used to display as 'ok', hiding a spurious fire.
                    if corrected and flipped:
                        ostat = 'FLIPPED→corrected (grasp lock)'
                    elif corrected:
                        ostat = 'LOCK-REVERSED (grasp lock fired)'
                    elif flipped:
                        ostat = 'FLIPPED (ends crossed)'
                    else:
                        ostat = 'ok'
                    ori_txt = (
                        f"  ||  orient: {ostat}"
                        + (f" d_same={ori['d_same']:.1f} d_flip={ori['d_flip']:.1f}"
                           if ori is not None else ""))
                    rec_txt = (
                        f"EKF reseed: "
                        f"{('FIRED ×%.1f' % rec['factor']) if fired else 'idle'}  |  "
                        f"gate={rec['gate_frac']:.2f}(<{rec['frac_thresh']:.2f})  "
                        f"streak={rec['streak']}/{rec['frames_thresh']}  "
                        f"motion={rec['motion']:.2f}(<{rec['motion_max']:.2f})"
                        + ori_txt)
                    rec_color = 'tab:red' if (fired or flipped or corrected) else 'black'
                else:
                    rec_txt, rec_color = '', 'black'
                axk.set_title(f'matched keypoints / NN t-assignment  '
                              f'(frame {self._vis_z_count})\n{sync_str}{tool_txt}\n'
                              f'{rec_txt}',
                              fontsize=10, color=rec_color)
                axk.legend(fontsize=7, loc='upper right')
                plt.tight_layout()
                plt.savefig(f"debug_z_noise_{self._vis_z_count:04d}_matched.png",
                            dpi=150, bbox_inches='tight')
                plt.close(figk)
        except Exception as e:
            self.node.get_logger().warning(f"VIS_Z_NOISE plot failed: {e}")
        finally:
            self._vis_z_count += 1

    def _get_ekf_bspline(self):
        """EKF posterior spline as a cubic BSpline with .t/.c/.k (msg-ready),
        or None if the filter isn't initialised / conversion fails.  get_spline()
        returns a CubicSpline (PPoly — wrong .c layout, no .t/.k), so convert the
        control points with make_interp_spline (not-a-knot → same curve)."""
        if not getattr(self.Order, '_ekf_initialized', False):
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None
        try:
            from scipy.interpolate import make_interp_spline
            ekf = self.Order.ekf
            return make_interp_spline(ekf.t_ctrl, ekf.x.reshape(ekf.M, 3), k=3)
        except Exception as e:
            self.node.get_logger().warning(f"EKF spline conversion failed ({e}).")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None

    def _ekf_thread_specs(self, thread, keypoints, order, keypt_conf,
                          n_samples=200):
        """thread_specs built straight from the filter — no optim, no QP.

        Lets EKF_OUTPUT_MODE='ekf_thread' publish without running the solver at
        all; optim's specs are the only thing that still forced it.

        reliability  per-keypoint STEREO confidence (keypt_selection's
                     last_keypt_conf, indexed by `order`) interpolated over the
                     published samples — the same thing optim's
                     RELIABILITY_MODE='measurement' publishes.  Deliberately
                     NOT derived from the filter covariance: that measures
                     where the FILTER is confident, not where the SENSOR was,
                     which is exactly the flaw optim documents for
                     RELIABILITY_MODE='geometry'.  It would read most confident
                     on the epipolar-parallel stretches where the filter has
                     been coasting — the worst possible place for it.
        lower/upper  the curve ± EKF_SPEC_SIGMA · σ_z, σ_z from
                     SplineEKF.sample_pos_sigma (posterior covariance pushed
                     through the cardinal basis).  A real 1σ depth envelope
                     rather than optim's max(|z−local trend|·1.5, line_std)
                     heuristic, which is contaminated by neighbouring keypoints
                     through its shared window.
        keypt_s      each ordered keypoint's t, by nearest point on the curve.

        Returns None — caller keeps optim's specs — if the filter or the
        inputs aren't usable.
        """
        ekf = getattr(self.Order, 'ekf', None)
        if ekf is None or order is None or len(order) < 2:
            return None
        us  = np.linspace(0.0, 1.0, n_samples)
        pts = np.asarray(thread(us), dtype=float)               # (n, 3) camera
        sig = ekf.sample_pos_sigma(us)
        if pts.shape != (n_samples, 3) or sig is None or len(sig) != n_samples:
            return None
        # Raw posterior depth σ.  Scaled by reliability once `rel` is final —
        # see below; the filter's own covariance is nearly flat along the
        # thread (P0 is isotropic and the smoothness prior couples the control
        # points), so on its own it gives an almost uniform envelope.
        sigma_z = sig[:, 2]

        # keypt_s: nearest point on the published curve for each ordered
        # keypoint (change_coords mutates, hence the copy).
        kp3 = change_coords(
            np.asarray(keypoints, dtype=float)[np.asarray(order, int)].copy(),
            self.cam2img1)
        d       = np.linalg.norm(kp3[:, None, :] - pts[None, :, :], axis=2)
        keypt_s = us[np.argmin(d, axis=1)]

        kc = np.ones(len(order))
        if keypt_conf is not None:
            c   = np.asarray(keypt_conf, dtype=float)
            idx = np.asarray(order, int)
            if idx.size and idx.max() < c.size and idx.min() >= 0:
                kc = np.clip(c[idx], 0.0, 1.0)
        # argmin gives no ordering or uniqueness guarantee — sort and dedupe
        # before interpolating, else np.interp silently returns garbage.
        o = np.argsort(keypt_s)
        s_u, iu = np.unique(keypt_s[o], return_index=True)
        rel = (np.interp(us, s_u, kc[o][iu]) if s_u.size >= 2
               else np.full(n_samples, float(kc.mean())))
        # DATA-GAP degradation — the same pass optim applies to its own
        # published reliability.  Without it a stretch carrying NO keypoints
        # (occlusion, dropped clusters, the gripper's own hole) simply
        # interpolates between its bracketing keypoints and reads just as
        # confident as they do, which is the single biggest reason the
        # published reliability looks uniform along the thread.  Scales each
        # sample by exp(-max(0, d - deadband) / GAP_RELIAB_DECAY_S), d being
        # its arc-length distance to the nearest supporting keypoint.
        rel = np.asarray(self.Optim._degrade_gap_reliability(
            rel, us, np.asarray(s_u, dtype=float), speedy=self.speedy),
            dtype=float)

        # ── Bounds = σ scaled by reliability ──────────────────────────────────
        # half = EKF_SPEC_SIGMA · σ_z / max(rel, EKF_SPEC_REL_FLOOR).  Same
        # convention taper_endpoints used at the tips (lower reliability ⇒
        # wider bound), applied along the WHOLE thread instead of only the
        # outer 30%.  Without it the envelope is near-uniform: σ_z varies only
        # ~15% end-to-middle, so the covariance alone cannot express "this
        # stretch carries no data" — but `rel` can, since gap degradation has
        # just written that information into it.
        # The floor matters: after gap degradation rel reaches ~1e-3 over a
        # long unsupported stretch, and an unfloored divide would publish a
        # metre-wide band there.  0.25 caps the widening at 4x, close to the
        # 3.3x the endpoint taper used to apply at the tips.
        half = (self.EKF_SPEC_SIGMA * sigma_z
                / np.maximum(rel, self.EKF_SPEC_REL_FLOOR))

        if not self.speedy:
            self.node.get_logger().info(
                f"EKF thread specs (no QP): depth σ min={sigma_z.min():.2f} "
                f"median={np.median(sigma_z):.2f} max={sigma_z.max():.2f}  "
                f"reliability min={rel.min():.2f} median={np.median(rel):.2f}  "
                f"bound half-width min={half.min():.2f} median="
                f"{np.median(half):.2f} max={half.max():.2f} "
                f"(±{self.EKF_SPEC_SIGMA:.1f}σ / max(rel, "
                f"{self.EKF_SPEC_REL_FLOOR:.2f}))")
        return {"reliability": rel,
                "lower_constr": np.column_stack(
                    [pts[:, 0], pts[:, 1], pts[:, 2] - half]),
                "upper_constr": np.column_stack(
                    [pts[:, 0], pts[:, 1], pts[:, 2] + half]),
                "keypt_s": list(keypt_s)}

    # Radius (image px) within which a sampled thread point counts as passing
    # THROUGH a crossing.  Measured on left_thread_segment.png the merged
    # two-strand blobs span 26x47 and 35x24 px, so 25 covers a pass without
    # reaching the next crossing along.
    INTERSECTION_MATCH_PX = 25.0

    def _intersection_thread_points(self, thread, intersection_segments, P,
                                    n=200):
        """Match each detected crossing onto the published thread.

        A crossing is where the thread passes over ITSELF, so it corresponds to
        TWO separate places along t — this returns one row per PASS, tagged
        with a `crossing_id` shared by all passes of the same physical
        intersection.  That id is the whole point: it says "these two points on
        the thread are the same crossing", which the 2-D detection alone cannot
        express (it only knows there is a blob at some pixel).

        Method: take the crossing's centre in image coords (the two RANSAC arm
        axes' intersection, falling back to the mean of their segment centres —
        the same derivation _detangle_clip uses), project the thread to 2-D,
        and find each CONTIGUOUS RUN of samples within INTERSECTION_MATCH_PX of
        that centre.  One run per pass; its closest sample is the match.  Runs
        rather than a global argmin is what separates the two passes instead of
        collapsing them onto whichever is marginally nearer.

        Returns (rows, n_crossings) where each row is
            [crossing_id, t, x, y, z, u_centre, v_centre]
        with x,y,z the camera-frame position of that pass and u,v (col,row) the
        crossing centre in the image.  Empty when there is nothing to match.
        """
        if thread is None or not intersection_segments:
            return np.zeros((0, 7), dtype=float), 0
        rows_img, cols_img, _ = self._project_to_row_col_depth(thread, P, n=n)
        us = np.linspace(0.0, 1.0, n)
        p3 = np.asarray(thread(us), dtype=float)
        out = []
        for cid, crossing in enumerate(intersection_segments):
            # ── crossing centre, in (row, col) as the detection works in ──
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
                c = np.mean([np.asarray(a['segment_center'], dtype=float)
                             for a in crossing], axis=0)
            d = np.hypot(rows_img - c[0], cols_img - c[1])
            near = d <= self.INTERSECTION_MATCH_PX
            if not near.any():
                continue
            # contiguous runs of `near` → one per pass through the crossing
            edges = np.flatnonzero(np.diff(near.astype(int)))
            starts = ([0] if near[0] else []) + list(edges[near[edges + 1]] + 1)
            for st in starts:
                en = st
                while en + 1 < n and near[en + 1]:
                    en += 1
                i = st + int(np.argmin(d[st:en + 1]))
                out.append([float(cid), float(us[i]),
                            p3[i, 0], p3[i, 1], p3[i, 2],
                            float(c[1]), float(c[0])])       # u=col, v=row
        return (np.asarray(out, dtype=float) if out
                else np.zeros((0, 7), dtype=float)), len(intersection_segments)

    def _publish_intersections(self, thread, intersection_segments, P):
        """Publish /thread/intersections (see _intersection_thread_points)."""
        try:
            rows, n_cross = self._intersection_thread_points(
                thread, intersection_segments, P)
        except Exception as e:
            self.node.get_logger().warning(f"intersection matching failed ({e}).")
            return
        msg = Float32MultiArray()
        msg.layout = MultiArrayLayout(dim=[
            MultiArrayDimension(label='pass', size=int(rows.shape[0]),
                                stride=int(rows.size)),
            MultiArrayDimension(
                label='crossing_id,t,x,y,z,u,v', size=7, stride=7)])
        msg.data = [float(v) for v in rows.ravel()]
        self.pub_intersections.publish(msg)
        if not self.speedy:
            per = {}
            for r in rows:
                per.setdefault(int(r[0]), []).append(round(float(r[1]), 3))
            detail = "  ".join(f"#{k}:t={v}" for k, v in sorted(per.items()))
            self.node.get_logger().info(
                f"/thread/intersections: {len(rows)} thread pass(es) over "
                f"{n_cross} crossing(s){'  ' + detail if detail else ''}")

    def _mask_reprojection_frac(self, thread, mask, P, n=200):
        """Fraction of `n` points sampled along the 3-D spline that reproject
        onto the 2-D `mask` (dilated by MASK_REPROJ_TOL_PX).  Points that fall
        off the mask OR outside the image count against the fraction, so a
        spline that deviates from the segmented thread scores low."""
        us    = np.linspace(0.0, 1.0, n)
        pts3d = np.asarray(thread(us))
        aug   = np.concatenate([pts3d, np.ones((len(pts3d), 1))], axis=1)
        proj  = (P @ aug.T).T
        proj  = proj / (proj[:, 2:3] + 1e-7)
        proj  = proj[:, [1, 0, 2]]                 # → (row, col, _)
        H, W  = mask.shape
        r = np.round(proj[:, 0]).astype(int)
        c = np.round(proj[:, 1]).astype(int)
        inb = (r >= 0) & (r < H) & (c >= 0) & (c < W)

        m = (mask > 0).astype(np.uint8)
        if self.MASK_REPROJ_TOL_PX > 0:
            k = 2 * self.MASK_REPROJ_TOL_PX + 1
            m = cv2.dilate(m, np.ones((k, k), np.uint8))
        on = np.zeros(n, dtype=bool)
        on[inb] = m[r[inb], c[inb]] > 0
        return float(on.mean())

    # ══════════════════════════════════════════════════════════════════════
    #  Frame history — occlusion-robust warm-start source selection
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _thread_ends_and_arc(thread, n=100):
        """(end0, end1, arc_length) of a BSpline over its valid t-span."""
        lo = float(thread.t[thread.k])
        hi = float(thread.t[-(thread.k + 1)])
        p  = np.asarray(thread(np.linspace(lo, hi, n)), dtype=float)
        arc = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        return p[0], p[-1], arc

    @staticmethod
    def _same_direction(entry_a, entry_b):
        """True if two history entries run the same way along the thread,
        judged by which pairing of their endpoints is spatially closer."""
        d_same = (np.linalg.norm(entry_a["end0"] - entry_b["end0"])
                  + np.linalg.norm(entry_a["end1"] - entry_b["end1"]))
        d_flip = (np.linalg.norm(entry_a["end0"] - entry_b["end1"])
                  + np.linalg.norm(entry_a["end1"] - entry_b["end0"]))
        return d_same <= d_flip

    def _record_frame(self, thread, keypt_s, curr_T, mask_px, reproj_frac):
        """Append an ACCEPTED reconstruction to the rolling frame history."""
        end0, end1, arc = self._thread_ends_and_arc(thread)
        self.frame_idx += 1
        self._skip_streak = 0          # an accepted frame ends any stall
        self.frame_history.append({
            "idx":     self.frame_idx,
            "thread":  thread,
            "keypt_s": keypt_s,
            "T":       curr_T.copy(),
            "psm":     self.psm,
            "mask_px": int(mask_px),
            "arc":     arc,
            "end0":    end0,
            "end1":    end1,
            "reproj":  float(reproj_frac),
        })

    def _select_warm_source(self):
        """
        Pick the history entry the warm start should seed from.

        Preference order: the NEWEST entry, unless it is occlusion-degraded
        (arc length / mask coverage well below the best in the window) or its
        thread direction disagrees with the window majority — both signatures
        of a reconstruction made from a heavily occluded, confusing mask.  In
        that case fall back to the most recent entry that has good coverage
        AND runs in the majority direction, so the warm start (and therefore
        the published thread's orientation) stays anchored to the consensus of
        the last few frames instead of one bad frame.

        Returns the chosen entry dict, or None when no history exists.
        """
        hist = list(self.frame_history)
        if not hist:
            return None
        newest = hist[-1]
        if len(hist) < 3:
            return newest        # not enough context to out-vote the newest

        best_arc  = max(e["arc"]     for e in hist)
        best_mask = max(e["mask_px"] for e in hist)

        def degraded(e):
            # Under DETANGLE the thread is clipped ON PURPOSE, so a short arc
            # is the intended output rather than evidence of occlusion.  Left
            # in, the arc test would mark every detangled frame degraded, the
            # warm source would pin to a pre-detangle frame and go stale, and
            # the route would defeat itself.  Mask coverage still applies — it
            # measures the segmentation, which detangle does not touch.
            arc_bad = (False if self.DETANGLE_ENABLED
                       else e["arc"] < self.HIST_ARC_KEEP_FRAC * best_arc)
            return arc_bad or e["mask_px"] < self.HIST_MASK_KEEP_FRAC * best_mask

        good = [e for e in hist if not degraded(e)]
        if not good:
            return newest        # everything shrank alike → trust the newest

        # Majority direction among the good entries, anchored on the oldest
        # good entry: each votes same/flipped relative to it.
        anchor = good[0]
        same   = [e for e in good if self._same_direction(anchor, e)]
        flip   = [e for e in good if e not in same]
        majority = same if len(same) >= len(flip) else flip

        newest_ok = (not degraded(newest)) and any(
            e is newest for e in majority)
        if newest_ok:
            return newest

        # Most recent good, majority-direction entry.
        pick = majority[-1]
        why = []
        if degraded(newest):
            why.append(f"occluded (arc {newest['arc']:.0f}/{best_arc:.0f}, "
                       f"mask {newest['mask_px']}/{best_mask})")
        if not any(e is newest for e in majority):
            why.append("direction disagrees with window majority")

        # Staleness cap: refusing the newest frame for several frames running
        # pins prev_thread/prev_T to an old frame, so |t| accumulates and the
        # warm spline drifts far from the keypoints — which degrades the fit and
        # keeps the newest frame looking wrong.  Break the loop.
        age = newest["idx"] - pick["idx"]
        if age > self.WARM_SRC_MAX_AGE:
            self.node.get_logger().warning(
                f"Warm-start source: would seed from frame {pick['idx']} but it "
                f"is {age} frames older than the newest ({newest['idx']}) — over "
                f"the WARM_SRC_MAX_AGE={self.WARM_SRC_MAX_AGE} cap "
                f"({'; '.join(why)}).  The consensus has been rejecting new "
                "frames long enough that the warm start is stale (|t| "
                "accumulates and the warm spline drifts off the keypoints); "
                "using the NEWEST frame and letting the fit re-acquire.")
            return newest

        self.node.get_logger().warning(
            f"Warm-start source: frame {newest['idx']} skipped "
            f"({'; '.join(why)}); seeding from frame {pick['idx']} "
            f"(age {age}, {len(self.frame_history)} in history).")
        return pick

    def reconstruct_callback(self, call):
        if not self.images_init:
            self.node.get_logger().info("image topics not ready before reconstruct call...")
            return
            
        self.node.get_logger().info("Received reconstruct call...")
        psm = self.psm
        prev_spm = self.prev_psm
        if self.psm not in (1, 2):
            print(f"no psm defined, psm: {self.psm}")
            return
        # Use the tool pose FROM THE IMAGE'S TIMESTAMP (~3 s ago), not the
        # current pose — the motion model must match the thread motion in the
        # delayed image (see _psm1_hist).  Falls back to the latest pose if the
        # buffer can't reach that far back.
        frame_stamp = self._img_stamp_l

        # ── Is this actually a NEW frame? ─────────────────────────────────────
        # Compare the stored image stamp against the last one processed.  Also
        # report how stale the stored set is (wall-clock age since the sync last
        # fired) and the trigger:sync ratio — if triggers >> syncs, /thread/call
        # is outrunning the mask pipeline and most calls are re-work.
        self._recon_count += 1
        sync_age = (time.time() - self._last_sync_wall
                    if self._last_sync_wall is not None else float('nan'))
        is_repeat = (self._last_recon_stamp is not None
                     and frame_stamp == self._last_recon_stamp)
        if is_repeat:
            self._stale_repeats += 1
            # One line: /thread/call outruns the mask rate routinely, so the
            # full explanation (identical keypoints + pose → zero motion) would
            # bury every other log.  Specs kept, prose dropped.
            self.node.get_logger().warning(
                f"STALE FRAME: stamp={frame_stamp:.3f} unchanged, no new sync "
                f"in {sync_age:.2f}s  syncs={self._sync_count} "
                f"triggers={self._recon_count} repeats={self._stale_repeats}  "
                f"({'skipped' if self.SKIP_STALE_FRAMES else 'processing'})")
            if self.SKIP_STALE_FRAMES:
                return
        else:
            self.node.get_logger().info(
                f"new frame: stamp={frame_stamp:.3f} (sync {sync_age:.2f}s ago) "
                f"syncs={self._sync_count} triggers={self._recon_count} "
                f"repeats={self._stale_repeats}")
        # Tag every EKF log line this frame with a processed-frame number so a
        # logged flip / recovery / gate can be located in the log stream.  Own
        # counter (per processed NEW frame — stale repeats skipped above don't
        # advance it) because _vis_z_count only advances inside _vis_z_noise:
        # with the debug PNGs disabled it stays 0 and every line said
        # "[frame 0]".  With the PNGs enabled both counters advance once per
        # processed frame, so tags line up with debug_z_noise_<frame>_*.png.
        self._ekf_frame_count = getattr(self, '_ekf_frame_count', -1) + 1
        self.Order.ekf.frame = self._ekf_frame_count
        # Keep the PREVIOUS processed frame's stamp so the other arm's motion can
        # be measured over the same image-time interval as the active arm's.
        self._last_recon_stamp_prev = self._last_recon_stamp
        self._last_recon_stamp = frame_stamp
        # Stall watchdog: assume this frame will be skipped; _record_frame
        # (called only for ACCEPTED reconstructions) resets the streak.
        self._skip_streak += 1

        curr_T, pose_span = self._psm_T_at(frame_stamp, self.psm)
        if curr_T is None:
            n_msg  = getattr(self, '_psm_msg_count', {}).get(self.psm, 0)
            n_fail = getattr(self, '_psm_conv_fail', {}).get(self.psm, 0)
            if n_msg == 0:
                why = (f"NO messages have arrived on /PSM{self.psm}/measured_cp "
                       "— check the topic is publishing and the QoS matches")
            elif n_fail:
                why = (f"{n_msg} messages arrived but {n_fail} failed to convert "
                       "— see the 'pose callback FAILED' error above")
            else:
                why = f"{n_msg} messages arrived but the history is empty"
            self.node.get_logger().warning(
                f"PSM{self.psm} pose unavailable ({why}); skipping reconstruct.")
            return
        self.node.get_logger().info(
            f"PSM{self.psm} pose interpolated at image time "
            f"(bracket span {pose_span*1e3:.0f}ms).")
        img1=self.img1
        img2=self.img2
        mask1=self.mask1
        mask2=self.mask2

        if np.sum(mask1) < 10 or np.sum(mask2) < 10:
            self.node.get_logger().warning("Masks are empty...")
            return
        
        # ── Warm-start source: best recent frame, not blindly the last one ────
        # _select_warm_source() walks the rolling history of the last accepted
        # reconstructions and skips entries that were made from a heavily
        # occluded mask (truncated arc / low coverage) or whose direction
        # disagrees with the window majority — so occlusion can neither poison
        # the warm start nor flip the thread's direction.
        src = self._select_warm_source()
        if src is not None:
            prev_thread = src["thread"]
            prev_keypts = src["keypt_s"]
            src_T       = src["T"]
            src_psm     = src["psm"]
        else:
            prev_thread = self.prev_thread   # cold start: no history yet
            prev_keypts = self.prev_keypts
            src_T       = self.prev_T
            src_psm     = self.prev_psm

        # ── Is the ACTIVE arm the one that is actually moving? ────────────────
        # The motion model only ever uses self.psm (from /manipulate/psm).  If
        # the thread is being dragged by the OTHER arm, the tracked pose barely
        # changes while the keypoints move, and no amount of warp tuning can
        # explain the motion.  Report both arms' displacement over the same
        # image-time interval so that case is immediately visible.
        if src_T is not None:
            d_active = float(np.linalg.norm(curr_T[:3, 3] - src_T[:3, 3]))
            other    = 2 if self.psm == 1 else 1
            _prev_stamp = getattr(self, '_last_recon_stamp_prev', None)
            oth_now, _  = self._psm_T_at(frame_stamp, other)
            oth_prev, _ = (self._psm_T_at(_prev_stamp, other)
                           if _prev_stamp is not None else (None, None))
            if oth_now is not None and oth_prev is not None:
                d_other = float(np.linalg.norm(oth_now[:3, 3] - oth_prev[:3, 3]))
                flag = ("  <-- OTHER ARM IS MOVING, ACTIVE ONE IS NOT"
                        if d_other > max(3.0 * d_active, 1.0) else "")
                self.node.get_logger().info(
                    f"arm motion over this frame: active PSM{self.psm}="
                    f"{d_active:.3f}mm  other PSM{other}={d_other:.3f}mm{flag}")
            else:
                self.node.get_logger().info(
                    f"arm motion over this frame: active PSM{self.psm}="
                    f"{d_active:.3f}mm (other arm unavailable)")

        # Detect a switch of the active PSM (1↔2) relative to the SELECTED warm
        # source.  curr_T then comes from a DIFFERENT tool than the source's T,
        # so their relative transform is a large pose discontinuity that the
        # EKF/warm-start would otherwise read as a thread translation and drag
        # the reconstruction across the scene.  The thread itself is continuous
        # across the switch, so treat this frame as zero tool motion:
        # prev_T = curr_T makes the relative transform identity (motion≈0), so
        # no spurious drag is applied.
        # Also suppress tool motion when the active gripper's jaw is open: an
        # open jaw is not holding the thread, so the tool's motion must not be
        # applied as a thread transform.
        psm_changed = (self.psm != src_psm)
        jaw_open    = self._active_jaw_open(stamp=frame_stamp)
        # Report the MEASURED jaw angle, not just the verdict: a gripper holding
        # a thread closes onto the thread's thickness, so a grasping jaw sits at
        # a few degrees rather than 0.  If that exceeds JAW_OPEN_THRESH the tool
        # motion is zeroed EVERY frame and the warped thread just repeats the
        # previous reconstruction (looks like a one-frame lag), so the actual
        # angle vs the threshold has to be visible.
        jaw_meas = self._jaw_at(frame_stamp, self.psm)
        jaw_str  = ('unknown' if jaw_meas is None
                    else f'{np.rad2deg(jaw_meas):.2f} deg')
        if psm_changed or jaw_open:
            reason = ("active PSM changed "
                      f"{src_psm} -> {self.psm}" if psm_changed
                      else (f"PSM {self.psm} jaw reads OPEN: {jaw_str} > "
                            f"{np.rad2deg(self.JAW_OPEN_THRESH):.1f} deg thresh"))
            self.node.get_logger().warning(
                f"{reason}; treating this frame as ZERO tool motion "
                "(not applying the PSM transform to the thread) — the warped "
                "thread will just repeat the previous reconstruction.")
            prev_T = curr_T
        else:
            self.node.get_logger().info(
                f"PSM {self.psm} jaw closed ({jaw_str} <= "
                f"{np.rad2deg(self.JAW_OPEN_THRESH):.1f} deg); applying tool "
                "motion to the thread.")
            prev_T = src_T

        # ── Per-stage wall-clock timing (always on; one summary line/frame) ──
        _stage_t  = {}
        _t0_frame = time.perf_counter()
        _t0       = _t0_frame
        def _lap(name):
            nonlocal _t0
            now = time.perf_counter()
            _stage_t[name] = now - _t0
            _t0 = now

        # Execute your processing pipeline strictly in-memory
        warm_thread, warm_keypts = self.Warmstart.refresh_warm_start(prev_thread=prev_thread,
                                          prev_keypts=prev_keypts,
                                          curr_T=curr_T,
                                          prev_T=prev_T,
        )
        _lap("warm")

        # Warm-start spline clips the mask to the neighbourhood of the previous
        # (motion-compensated) reconstruction, keeping only the thread mask that
        # is relevant to the ACTIVE PSM's grasp (the needle→gripper span around
        # curr_T) and dropping background clutter before selection.
        #
        # That clip is anchored on the gripper position, so it is only
        # meaningful when the active gripper is actually holding the thread.
        # The active PSM comes from /manipulate/psm (self.psm → curr_T) and its
        # jaw from /PSMx/jaw/measured_js: clip only when that jaw is CLOSED.  An
        # open jaw (gripper not grasping) or unknown jaw → keep the full mask.
        jaw_closed  = not self._active_jaw_open(stamp=frame_stamp)
        clip_thread = (warm_thread
                       if (self.CLIP_MASK_ENABLED and self.use_warm_start and jaw_closed)
                       else None)
        if self.CLIP_MASK_ENABLED and self.use_warm_start and not jaw_closed:
            self.node.get_logger().info(
                f"PSM {self.psm} jaw open (> {np.rad2deg(self.JAW_OPEN_THRESH):.1f} deg) "
                "or unknown; not clipping mask (gripper not grasping the "
                "thread), keeping the full mask this frame.")
        # Grasp candidates from BOTH arms (either PSM may hold the thread):
        # camera-frame gripper poses for every CLOSED jaw at the frame stamp.
        # Used by the cold-order direction keeper below and by the DETANGLE
        # clip; the active arm reuses the already-interpolated curr_T.
        grasp_cands = {}
        for p in (1, 2):
            jaw_p = self._jaw_at(frame_stamp, p)
            if jaw_p is None or jaw_p > self.JAW_OPEN_THRESH:
                continue
            T_p = (curr_T if p == self.psm
                   else self._psm_T_at(frame_stamp, p)[0])
            if T_p is not None:
                grasp_cands[p] = T_p     # full pose: translation is the grasp,
                                         # rotation carries the anchored
                                         # direction

        # Keep the un-clipped left mask for the debug mask panel, so the clip's
        # effect (and any segmentation dropout) is visible side by side.
        mask1_preclip = np.asarray(mask1).copy() if self.VIS_Z_NOISE else None
        img1, img2, mask1 = self.prep_images(curr_T, img1, img2, mask1, mask2, clip_thread, dist_thresh=20, speedy=self.speedy)
        _lap("prep")

        # ── Segmentation blow-out guard ───────────────────────────────────────
        # When SAM3 grabs the instrument shaft (or any non-thread object) the
        # mask area jumps by an order of magnitude in a single frame.  Nothing
        # downstream survives that: stereo triangulates garbage (observed Z up
        # to 1e10), the fat mask merges clusters, the merged clusters fake
        # intersections, and the resulting order zig-zags until the optim QP is
        # primal-infeasible.  The thread cannot physically grow 10x between two
        # frames, so treat it as a segmentation failure and skip.
        #
        # Compared against the MEDIAN of the last ACCEPTED frames (not the mean,
        # and not including rejects) so one blow-out that slips through can't
        # raise the bar and let the next one past.  Counted on the raw mask
        # BEFORE keypt_selection's range filter — that filter drops the wild
        # pixels, so its own output understates the blow-out badly (frame 1057:
        # 34902 raw vs 5015 after).
        mask_px = int(np.count_nonzero(np.asarray(mask1) > 0))
        if len(self._mask_px_hist) >= self.MASK_BLOWUP_MIN_HIST:
            ref = float(np.median(self._mask_px_hist))
            if ref > 0 and mask_px > self.MASK_BLOWUP_RATIO * ref:
                self.node.get_logger().warning(
                    f"SEGMENTATION BLOW-OUT: mask={mask_px}px is "
                    f"{mask_px / ref:.1f}x the median of the last "
                    f"{len(self._mask_px_hist)} accepted frames ({ref:.0f}px); "
                    f"skipping (likely instrument shaft in the mask).")
                return
        self._mask_px_hist.append(mask_px)

        (img_3D, __, cluster_map, keypoints,
         grow_paths, adjacents, intersection_segments, dense_pts, reliable_flag) = \
            self.Select.keypt_selection(img1=img1, img2=img2, mask1=mask1, Q=self.Q, speedy=self.speedy)
        _lap("select")

        # `keypoints` is the DENSE, reliability-agnostic full_cluster_means used
        # for ordering; reliable_flag[i] marks whether full mean i is reliable.
        # After ordering we filter the ordered sequence down to reliable points
        # before optimisation (see _filter_reliable_order).
        full_means = keypoints

        # Warm + EKF ordering drives the shape again: the EKF-predicted spline
        # from the previous frame is matched to this frame's keypoints (graph-
        # consistent NN + Mahalanobis gate) and then updated with the matches.
        # When it can't produce a usable order (stale/scribbled warm spline or
        # too few matches) it returns None and we fall back to cold ordering.
        # Stall watchdog trip: bypass the warm/EKF path entirely so the cold
        # ordering below re-acquires from this frame's data alone.  The EKF is
        # NOT predicted/updated this frame; if the cold frame is accepted,
        # update_from_thread (which runs on both paths) snaps it back.
        force_cold = self._skip_streak > self.STALL_FRAMES
        if force_cold:
            self.node.get_logger().warning(
                f"STALL WATCHDOG: {self._skip_streak - 1} consecutive "
                "reconstruct calls without an accepted frame — bypassing the "
                "warm/EKF ordering and re-acquiring COLD from this frame's "
                "data.")
            warm_keypoints, order, new_warm_keypts = None, None, None
        else:
            warm_keypoints, order, new_warm_keypts = \
                self.Order.run_warm_ordering_with_ekf(
                    mask1, keypoints,
                    P1=self.P1,
                    warm_thread=warm_thread,
                    prev_keypts=warm_keypts,
                    curr_T=curr_T,
                    prev_T=prev_T,
                    speedy=self.speedy,
                    # KF↔optim loop: the filter is corrected from optim's
                    # thread after optim, not from the raw matched keypoints.
                    update_ekf=not self.KF_OPTIM_LOOP,
                    adjacents=adjacents,
                    intersection_segments=intersection_segments,
                    dense_pts=dense_pts,
                    # Stereo confidence aligned with `keypoints` (full means):
                    # the EKF update inflates R for ambiguous-stereo keypoints
                    # (see SplineEKF.update conf / EKF_CONF_R_FLOOR).
                    keypt_conf=getattr(self.Select, 'last_keypt_conf', None),
                )
        _lap("order")

        warm_path = warm_keypoints is not None
        # Per-keypoint warm-match quality, aligned with `keypoints` and indexed
        # by `order` exactly like keypt_conf (the reliability/gap/detangle
        # filters below only restrict `order`, so alignment survives them).
        # Optim widens low-quality keypoints' constraint boxes with it.  Warm
        # path only — cold ordering has no warm match to score.
        match_q = None
        if warm_path:
            keypoints = warm_keypoints
            match_q = getattr(self.Order, '_match_q_ordered', None)
            if match_q is not None and len(match_q) != len(keypoints):
                match_q = None
        else:
            # Cold ordering fallback — most accurate shape when warm is unusable.
            # last_keypt_conf is indexed by full_cluster_mean, and on this
            # branch `keypoints` IS still full_means (warm ordering returned
            # None), so the two align directly — no nearest-neighbour lookup.
            __, keypoints, __, order = self.Order.keypt_ordering(
                img1, img_3D, cluster_map, keypoints, grow_paths,
                adjacents, intersection_segments=intersection_segments,
                speedy=self.speedy,
                keypt_conf=getattr(self.Select, 'last_keypt_conf', None))
            # A held grasp remembers the thread direction — carry it through
            # the cold re-acquisition (flip the fresh order if it came back
            # reversed relative to the gripper-noted outward direction).
            # DISABLED: grasp-anchored direction locks are off while direction
            # handling is isolated.  NOTE this was the only thing keeping a
            # COLD re-acquisition's arbitrary t-direction aligned during a
            # hold — with it off, a cold frame may come back reversed.
            # order = self._orient_order_to_anchor(keypoints, order, grasp_cands)

        keypoints, order, keypt_conf = self._filter_reliable_order(
            keypoints, order, full_means, reliable_flag,
            full_conf=getattr(self.Select, 'last_keypt_conf', None))
        # Drop stray large-gap joins that would make the optim QP infeasible.
        order = self._gate_large_gaps(keypoints, order, self.cam2img1)
        # DETANGLE route (see _detangle_clip): most reliable group when free,
        # grasp-anchored "segment out of the gripper" while a PSM holds the
        # thread.  Off unless DETANGLE_ENABLED.  Candidates come from BOTH
        # arms — either may hold; the anchor binds to one PSM and releases
        # only when that jaw opens.  Positions are camera-frame (same space
        # as change_coords / the published spline); the active arm reuses the
        # already-interpolated curr_T, the other arm is read from its own
        # pose history at the frame stamp.
        if self.DETANGLE_ENABLED:
            order = self._detangle_clip(
                keypoints, order, keypt_conf, self.cam2img1,
                grasp_cands=grasp_cands)
        _lap("gate")

        # Degenerate frame guard: optim needs a handful of ordered points (its
        # init indexes endpoints and fits ~16 control points).  An empty or
        # near-empty order here means this frame's selection/ordering collapsed
        # — skip it and keep the previous thread rather than crash in optim.
        MIN_ORDERED_KPTS = 4
        if order is None or len(order) < MIN_ORDERED_KPTS:
            self.node.get_logger().warning(
                f"Only {0 if order is None else len(order)} ordered keypoints "
                f"after filtering (< {MIN_ORDERED_KPTS}); skipping frame, "
                "keeping previous thread.")
            return

        # ── Quality-gate the temporal prior ───────────────────────────────────
        # The prior feeds each frame's output back into the next frame's
        # objective.  Once a wobble enters the published spline, the Tikhonov
        # term DEFENDS it against the bending smoother, so waviness compounds
        # frame over frame and never recovers ("spaghetti").  If the warm spline
        # is kinked (min bend radius too small) or scrambled, drop the prior for
        # this frame: optim rebuilds from current data only, which is the
        # recovery path.  Thresholds are the calibrated warm-spline ones from
        # keypt_ordering (loops/self-intersections still pass).
        prior_ok = False
        if warm_thread is not None:
            # No ref → scramble comes back NaN ("not measured"); gate on the
            # bend radius only.  A NaN must count as OK, not as a failure.
            p_radius, p_scramble = self.Order._spline_quality(warm_thread)
            scramble_bad = (np.isfinite(p_scramble)
                            and p_scramble > self.Order._WARM_SCRAMBLE_MAX)
            prior_ok = (p_radius >= self.Order._WARM_MIN_BEND_RADIUS
                        and not scramble_bad)
            if not prior_ok:
                self.node.get_logger().warning(
                    f"Temporal prior dropped: warm spline degraded "
                    f"(R={p_radius:.2f}, scramble={p_scramble:.2f}; "
                    f"limits R>={self.Order._WARM_MIN_BEND_RADIUS}, "
                    f"scramble<={self.Order._WARM_SCRAMBLE_MAX}) — "
                    "reconstructing this frame from data only.")

        # ── EKF-thread mode: no QP at all ─────────────────────────────────────
        # The filter already holds the answer (it updated from the matched
        # keypoints in run_warm_ordering_with_ekf) and _ekf_thread_specs derives
        # the published reliability/bounds from its own posterior covariance, so
        # optim contributes nothing here but latency.  Worse, it used to run
        # anyway and its "QP solver returned non-finite solution (infeasible?),
        # skipping frame" path returns BEFORE the output swap further down —
        # so a solver this mode does not even consume was killing whole frames.
        # Decide here, and leave optim uncalled.
        ekf_only = False
        if self.USE_EKF_THREAD and not self.KF_OPTIM_LOOP:
            ekf_bspline = self._get_ekf_bspline()
            ekf_specs = (self._ekf_thread_specs(
                             ekf_bspline, keypoints, order, keypt_conf)
                         if ekf_bspline is not None else None)
            if ekf_specs is not None:
                thread_dict       = {'thread': ekf_bspline}
                thread_specs_dict = ekf_specs
                ekf_only = True
            else:
                # Filter not ready (frame 0) or degenerate ordering — fall
                # through to optim for this frame rather than publish nothing.
                self.node.get_logger().info(
                    "EKF thread unavailable this frame; falling back to optim.")

        if ekf_only:
            pass
        elif self.KF_OPTIM_LOOP:
            # Option A: fit the raw keypoints with optim's boxes (robust), using
            # the KF's PREDICTED spline as the temporal shape prior.  Always the
            # cold optim() path so the prior is the KF, not the warm thread.
            ekf_prior = (self.Order.ekf.get_spline()
                         if getattr(self.Order, '_ekf_initialized', False)
                         else None)
            # ── MOTION-ADAPTIVE temporal prior ────────────────────────────
            # The optim temporal prior is translation-invariant, so it only
            # damps SHAPE change (bending) — which is exactly the deformation a
            # real tool motion induces near the tool.  At a constant λ the thread
            # under-tracks that deformation while moving.  Relax λ with the (held)
            # tool motion so the reconstruction FOLLOWS the keypoints while the
            # tool moves it, then restore full λ at rest for denoising.  The held
            # motion decays over ~5 frames after the tool stops, which keeps λ
            # relaxed long enough for the keypoints that jump AFTER the motion to
            # be followed.  Same floor formula as the KF's own shape priors.
            if ekf_prior is not None:
                motion = float(getattr(self.Order.ekf, '_last_motion', 0.0))
                floor  = float(getattr(self.Order.ekf, 'motion_prior_floor',
                                       ekf_params.MOTION_PRIOR_FLOOR))
                lam_eff = self.KF_PRIOR_LAMBDA * (floor + (1.0 - floor) * (1.0 - motion))

                # ── DATA-TRUST scaling: recover from a bad motion model ───────
                # The prior IS the motion-model-warped prediction, so when the
                # warp is wrong (bad grasp offset, unmodelled rotation, stale
                # warm source) the prior actively drags the fit off the data.
                # nn_median measures that disagreement directly, this frame.
                # Healthy tracking sits at ~2-3 px; scale the prior down by
                # REF/nn_median so a prediction that has drifted loses authority
                # to the keypoints and the fit re-acquires on its own.
                nn_med = float(getattr(self.Order, '_last_nn_median', float('nan')))
                trust  = 1.0
                if np.isfinite(nn_med) and nn_med > self.PRIOR_TRUST_NN_REF:
                    trust = max(self.PRIOR_TRUST_MIN,
                                self.PRIOR_TRUST_NN_REF / max(nn_med, 1e-6))
                    lam_eff *= trust
                self.node.get_logger().info(
                    f"optim temporal prior: lambda={lam_eff:.3f} "
                    f"(base {self.KF_PRIOR_LAMBDA}, motion={motion:.2f}, "
                    f"floor {floor}, nn_median={nn_med:.1f}px -> data-trust "
                    f"x{trust:.2f})")
            else:
                lam_eff = 0.0
            thread_dict, thread_specs_dict = self.Optim.optim(
                mask1, keypoints, order,
                self.cam2img1, self.P1, self.P2, speedy=self.speedy,
                x_prior_thread=ekf_prior,
                lambda_temporal=lam_eff,
                keypt_conf=keypt_conf,
                keypt_quality=match_q)
        elif warm_path:
            thread_dict, thread_specs_dict = self.Optim.optim_warm_start(
                mask1, keypoints, order, self.cam2img1, self.P1,
                warm_thread=warm_thread,
                warm_keypts=new_warm_keypts,
                speedy=self.speedy,
                keypt_conf=keypt_conf,
                **({} if prior_ok else {'lambda_temporal': 0.0}),
            )
        else:
            # Cold-ordering fallback still gets the temporal prior: warm_thread is
            # the previous reconstruction already motion-compensated to this frame
            # (same warp as SplineEKF.predict()), so it is passed as x_prior_thread
            # as-is (trans left at None to avoid double-counting the motion).  If
            # there is no warm thread it is None → prior is a no-op.
            thread_dict, thread_specs_dict = self.Optim.optim(
                mask1, keypoints, order,
                self.cam2img1, self.P1, self.P2, speedy=self.speedy,
                x_prior_thread=warm_thread if prior_ok else None,
                keypt_conf=keypt_conf)
        _lap("optim")

        # optim / optim_warm_start return (None, None) on a degenerate frame.
        # Skip publishing and keep the previous thread for the next round.
        if thread_dict is None:
            self.node.get_logger().warning(
                "Optim failed on this frame (degenerate ordering); skipping, "
                "keeping previous thread.")
            return

        # Direction reference = the EKF spline, i.e. the SAME curve
        # keypt_ordering t-sorted this frame's keypoints against
        # (filtered_warm_thread) and the same one update_from_thread's guard
        # pairs endpoints against.  Passing the raw warped previous thread here
        # made optim the only party voting on a DIFFERENT curve, which is why
        # its flips and the filter's guard cancelled each other frame after
        # frame.  Falls back to the raw warm thread before the filter exists.
        # _get_ekf_bspline, NOT ekf.get_spline(): match_warm_order samples
        # warm_thread.t, and get_spline() returns a CubicSpline (PPoly — no
        # .t/.k).  Falls back to the raw warm thread if either is unavailable.
        dir_ref = self._get_ekf_bspline() if getattr(
            self.Order, '_ekf_initialized', False) else None
        if dir_ref is None:
            dir_ref = warm_thread
        thread_dict, thread_specs_dict = self.Optim.match_warm_order(img1,
            thread_dict, thread_specs_dict,
            warm_thread=dir_ref, P=self.P1, interactive=False)

        thread = thread_dict.get("thread")
        optim_thread = thread          # optim's QP fit (published under Option A)

        # ── Option A: correct the KF from optim's robust thread ───────────────
        # The predict already ran (in run_warm_ordering_with_ekf); now fuse
        # optim's box-robust fit as the measurement, so the KF denoises optim
        # over time without ever seeing the raw keypoints.  optim_thread stays
        # the published output.
        if self.KF_OPTIM_LOOP and getattr(self.Order, '_ekf_initialized', False):
            try:
                self.Order.ekf.update_from_thread(
                    optim_thread,
                    # Published per-sample reliability (gap- and interpolation-
                    # degraded in optim/keypt_selection): stretches that carried
                    # no fresh data this frame barely correct the KF — it
                    # coasts on its prediction there (SplineEKF.update conf).
                    reliability=thread_specs_dict.get("reliability")
                    if isinstance(thread_specs_dict, dict) else None)
                # Direction lock (KF↔optim path): anchor the t-direction to
                # the ACTIVE PSM from /manipulate/psm — the single authority
                # for which arm is manipulating.
                #
                # TRIED AND REVERTED (2026-07-27): anchoring to whichever
                # gripper was NEAREST the thread instead.  During a bimanual
                # trade-off BOTH grippers are next to the thread, so the
                # nearest pick oscillated between the arms frame to frame,
                # and every oscillation re-anchored the lock (tool_id change)
                # at a different physical point — the thread kept flipping
                # exactly during handoffs.  The /manipulate/psm topic switches
                # ONCE when the hold transfers, so the lock re-anchors once,
                # deterministically.  If the operator drives the thread with
                # an arm the topic does not name, the topic — not a proximity
                # heuristic — is the thing to fix.
                # DISABLED: grasp-anchored direction locks are off while
                # direction handling is isolated.
                # ekf = self.Order.ekf
                # if ekf.x is not None and curr_T is not None:
                #     ekf.lock_orientation_to_grasp(
                #         curr_T[:3, 3], curr_T[:3, :3], tool_id=self.psm)
            except Exception as e:
                self.node.get_logger().warning(f"KF update-from-thread failed ({e}).")

        # ── EXPERIMENT: use the EKF spline as the output thread ───────────────
        # Replace optim's fit with the EKF posterior HERE (before the quality
        # log, reprojection gate, vis, history and publish all consume `thread`)
        # so the EKF spline is the validated, recorded, and published result.
        # Disabled under KF_OPTIM_LOOP (Option A publishes optim's thread).
        # Falls back to optim's thread if the EKF isn't ready.
        used_ekf_thread = False
        # Already decided (and optim skipped entirely) above — see `ekf_only`.
        if ekf_only:
            used_ekf_thread = True
            self.node.get_logger().info("Output = EKF thread (no QP ran).")

        # Observability: log the output spline's quality each frame so waviness
        # onset is visible (R shrinking / scramble rising over frames).  Not a
        # reject — a genuinely looping thread can sit near the limits.  The warm
        # spline is the trusted reference for the scramble measure; on the first
        # frame (no warm) scramble is NaN and only the radius is judged.
        o_radius, o_scramble = self.Order._spline_quality(thread, ref=warm_thread)
        o_scramble_bad = (np.isfinite(o_scramble)
                          and o_scramble > self.Order._WARM_SCRAMBLE_MAX)
        quality_ok = (o_radius >= self.Order._WARM_MIN_BEND_RADIUS
                      and not o_scramble_bad)
        # rclpy caches severity per source call site, so .info/.warning must be
        # two distinct calls — a shared log_fn(...) line raises
        # "Logger severity cannot be changed between calls."
        if quality_ok:
            self.node.get_logger().info(
                f"Output spline quality: R={o_radius:.2f} "
                f"scramble={o_scramble:.2f}")
        else:
            self.node.get_logger().warning(
                f"Output spline quality: R={o_radius:.2f} "
                f"scramble={o_scramble:.2f}  "
                "(DEGRADED — next frame will drop the temporal prior)")

        reliability = thread_specs_dict.get("reliability")
        lower_constr = thread_specs_dict.get("lower_constr")
        upper_constr = thread_specs_dict.get("upper_constr")
        # Down-weight reliability AND widen the depth bounds near the ends
        # (least certain region) — implemented in FitEvalClass
        # (thread_reconstruct.py); done before the vis/publish consume them.
        reliability, lower_constr, upper_constr = self.FitEval.taper_endpoints(
            reliability, lower_constr, upper_constr)
        keypt_s = thread_specs_dict.get("keypt_s")

        # ── Reject reconstructions that don't reproject onto the mask ─────────
        reproj_frac = self._mask_reprojection_frac(thread, self.mask1, self.P1)
        if reproj_frac < self.MASK_REPROJ_MIN:
            self.node.get_logger().warning(
                f"Reconstruction rejected: only {reproj_frac:.2f} of the spline "
                f"reprojects onto the mask (< {self.MASK_REPROJ_MIN}); "
                "keeping previous thread.")
            return
        self.node.get_logger().info(
            f"Reprojection check passed ({reproj_frac:.2f} on mask).")

        # ── Lag measurement ───────────────────────────────────────────────────
        # Is the thread trailing the keypoints, and which stage is responsible?
        # Positive = keypoints are ahead of the thread along the tool's motion.
        try:
            # 'prev(unwarped)' is the REQUIRED displacement: how far this frame's
            # keypoints sit ahead of the previous reconstruction before any warp.
            # 'warm(predicted)' is the RESIDUAL after warping.  Reading the two
            # together identifies the failure mode directly:
            #   required ≈ +D, residual ≈ 0     -> warp is correct
            #   required ≈ +D, residual ≈ -D    -> warp applied ~2x (double-count:
            #                                      the keypoints had already moved)
            #   required ≈ +D, residual ≈ +D    -> warp applied ~nothing
            _lag = self._measure_lag(
                full_means,
                {'prev(unwarped)':  prev_thread,
                 'warm(predicted)': warm_thread,
                 'optim(output)':   optim_thread},
                prev_T[:3, 3], curr_T[:3, 3], self.P1)
            if _lag:
                _mp  = _lag.pop('motion_px', 0.0)
                _req = _lag.get('prev(unwarped)')
                _res = _lag.get('warm(predicted)')
                _verdict = ""
                if _req is not None and _res is not None and abs(_req) > 2.0:
                    _ratio = 1.0 - (_res / _req)   # fraction of required applied
                    if _ratio > 1.5:
                        _verdict = (f"  <-- WARP OVERSHOOTS: applied "
                                    f"{_ratio:.2f}x the required displacement "
                                    "(motion double-counted; the keypoints had "
                                    "already moved)")
                    elif _ratio < 0.5:
                        _verdict = (f"  <-- WARP UNDER-APPLIES: only "
                                    f"{_ratio:.2f}x the required displacement")
                    else:
                        _verdict = f"  (warp applied {_ratio:.2f}x of required)"
                self.node.get_logger().info(
                    "lag along tool motion: "
                    + "  ".join(f"{k}={v:+.2f}px" for k, v in _lag.items())
                    + f"  (tool moved {_mp:.2f}px this frame)" + _verdict)
        except Exception as e:
            self.node.get_logger().warning(f"lag measurement failed ({e}).")

        if self.VIS_Z_NOISE:
            # Three reference curves: the EKF warm-start (this frame's EKF
            # posterior spline), the optim warm thread (previous reconstruction
            # fed into optim), and the optim thread (this frame's QP fit).
            ekf_warm = None
            if getattr(self.Order, '_ekf_initialized', False):
                try:
                    ekf_warm = self.Order.ekf.get_spline()
                except Exception:
                    ekf_warm = None
            self._vis_z_noise(full_means, reliable_flag,
                              lower_constr, upper_constr, reliability,
                              ekf_warm, optim_thread,
                              self.Order._ekf_reseeded, self.P1,
                              # transformed (motion-warped) previous thread plus
                              # the gripper pose that produced that warp
                              warm_thread=warm_thread,
                              curr_T=curr_T, prev_T=prev_T,
                              masks={'left mask (pre-clip)': mask1_preclip,
                                     'left mask (used)':     mask1,
                                     'right mask':           mask2},
                              dbg1=getattr(self.Order, '_dbg1', None))
        _lap("finalize")

        total = time.perf_counter() - _t0_frame
        self.node.get_logger().info(
            "[timing] " +
            "  ".join(f"{k}={v*1e3:.0f}ms" for k, v in _stage_t.items()) +
            f"  TOTAL={total*1e3:.0f}ms")

        # publish
        current_time = self.node.get_clock().now().to_msg()
        self.specs_msg.header.stamp = current_time       # <--- Assign the timestamp
        self.specs_msg.header.frame_id = "camera_frame"
        self.specs_msg.reliability = np.asarray(reliability, dtype=np.float64).flatten().tolist()
        self.specs_msg.lower_constr = np.asarray(lower_constr, dtype=np.float64).flatten().tolist()
        self.specs_msg.upper_constr = np.asarray(upper_constr, dtype=np.float64).flatten().tolist()
        self.specs_msg.keypt_s = np.asarray(keypt_s, dtype=np.float64).flatten().tolist()

        self.pub_thread_specs.publish(self.specs_msg)
        self.node.get_logger().info("Thread specs published!")

        # `thread` is already the EKF spline when USE_EKF_THREAD is on (swapped
        # right after optim, above), so it publishes as-is.
        self.spline_msg.header.stamp = current_time      # <--- Assign the matching timestamp
        self.spline_msg.header.frame_id = "camera_frame"
        self.spline_msg.knots = np.asarray(thread.t, dtype=np.float64).flatten().tolist()
        self.spline_msg.coeffs = np.asarray(thread.c, dtype=np.float64).flatten().tolist()
        self.spline_msg.degree = int(thread.k)
        
        self.pub_bspline.publish(self.spline_msg)
        self.node.get_logger().info("Thread published!")

        # Self-intersections matched onto the thread just published, so a
        # consumer can pair the two t values that are the same crossing.
        self._publish_intersections(thread, intersection_segments, self.P1)

        # save info for next round: the rolling history is the warm-start
        # source pool (see _select_warm_source); the self.prev_* mirrors are
        # kept as a cold-start fallback and for external readers.
        self._record_frame(thread, keypt_s, curr_T,
                           mask_px=np.count_nonzero(self.mask1 > 0),
                           reproj_frac=reproj_frac)
        self.prev_T = curr_T
        self.prev_psm = self.psm
        self.prev_keypts = keypt_s
        self.prev_thread = thread


    def prep_images(self, curr_T, img1, img2, mask1, mask2, warm_thread, dist_thresh, speedy):
        # kernel = np.ones((3, 3), np.uint8) # TODO put this in Omar's pipeline instead
        # mask1 = cv2.erode(mask1, kernel, iterations=1)
        # mask2 = cv2.erode(mask2, kernel, iterations=1)

        if warm_thread is not None:
            if self.CLIP_GRASP_WINDOW_PX is not None:
                # EXPERIMENT: clip to just the grasped section (±window px of
                # arc along the thread around the gripper).  Not-grasped frames
                # never reach here (jaw gate at the call site) or return
                # unclipped inside clip_mask (gripper not on the thread).
                mask1, mask2 = self.clip_mask(
                    mask1, mask2, self.P1, self.P2, curr_T, warm_thread,
                    dist_thresh=dist_thresh,
                    clip_radius=self.CLIP_GRASP_RADIUS,
                    speedy=speedy,
                    grasp_window_px=self.CLIP_GRASP_WINDOW_PX)
            else:
                mask1, mask2 = self.clip_mask(mask1, mask2, self.P1, self.P2, curr_T, warm_thread, dist_thresh=dist_thresh, speedy=speedy)

        # Apply segmentation masks and convert to uint32 in one broadcast
        # (avoids the 3-channel np.stack + np.where temporaries)
        img1 = img1 * (mask1 > 0).astype(np.uint32)[..., None]
        img2 = img2 * (mask2 > 0).astype(np.uint32)[..., None]
        return img1,img2,mask1

    def clip_mask(self, mask1, mask2, P1, P2, curr_T, trans_thread, dist_thresh=20, clip_radius=80, speedy=False,
                  grasp_window_px=None):
        return self.FitEval.clip_mask(mask1, mask2, P1, P2, curr_T, trans_thread, dist_thresh=dist_thresh, clip_radius=clip_radius, speedy=speedy,
                                      grasp_window_px=grasp_window_px)
    
    def debug_sync_status(self):
        if self.images_init:
            return # Stop spamming once we successfully sync
            
        print("\n--- Synchronizer Diagnostic ---")
        for name, stamp in self.latest_stamps.items():
            if stamp == 0.0:
                print(f"[MISSING] {name}: No messages received yet.")
            else:
                print(f"[OK]      {name}: Last stamp at {stamp:.3f}")

        # Only the 4 image/mask topics go through the ApproximateTimeSynchronizer;
        # PSM poses are handled by separate cached callbacks + the timestamp buffer
        # (self._psm{1,2}_hist) and are NOT part of the sync, so they must NOT be
        # counted toward the slop check — a 0.5 s PSM spread does not make the sync
        # reject anything.  Also, comparing the LATEST stamp of an image vs the
        # latest mask is meaningless here: masks carry their ~3 s-old SOURCE-image
        # stamp, so that gap is sam3 latency, not a slop violation.  The only pairs
        # that must fall within slop are left↔right image and left↔right mask.
        li, ri = self.latest_stamps['left_img'],  self.latest_stamps['right_img']
        lm, rm = self.latest_stamps['left_mask'], self.latest_stamps['right_mask']
        ok = True
        if li and ri:
            d = abs(li - ri)
            flag = "" if d <= self.SYNC_SLOP else "  <-- EXCEEDS SLOP"
            print(f"\n=> L-R image spread: {d*1e3:.1f} ms (slop {self.SYNC_SLOP*1e3:.0f} ms){flag}")
            ok &= d <= self.SYNC_SLOP
        if lm and rm:
            d = abs(lm - rm)
            flag = "" if d <= self.SYNC_SLOP else "  <-- EXCEEDS SLOP"
            print(f"=> L-R mask  spread: {d*1e3:.1f} ms (slop {self.SYNC_SLOP*1e3:.0f} ms){flag}")
            ok &= d <= self.SYNC_SLOP
        if li and lm:
            # Informational only: this is the sam3 inference latency, handled by the
            # deep queue matching each mask to its (older) source image by stamp.
            print(f"=> img->mask latency (newest img vs newest mask): "
                  f"{(li - lm)*1e3:.0f} ms  [sam3 lag, not a slop check]")
        if ok:
            print("=> L-R pairs within slop. Sync should fire once a mask's source "
                  "image is still in the queue (needs queue_size >= fps x latency).")
        else:
            print("=> A stereo pair exceeds slop. Raise SYNC_SLOP or fix L/R publish "
                  "timing (this is independent of the PSM poses).")
        print("-------------------------------")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--speedy',           action="store_true")
    parser.add_argument('--time',             action="store_true",
        help="print per-stage Timer breakdowns (selection/ordering/optim) "
             "even in --speedy mode, to find where frame time goes")
    parser.add_argument('--calib',            default=None)
    parser.add_argument('--psm_calibrate',
        default=os.path.dirname(__file__) + "/../../../RaftStereo/assets/psm_calibration_servo.npz")
    args = parser.parse_args()

    reconstr = ROSThread(args)
    try:
        reconstr.node.get_logger().info("Starting continuous stereo processing. Waiting for messages...")
        while rclpy.ok() and not reconstr.images_init:
            rclpy.spin_once(reconstr.node, timeout_sec=0.1)
            reconstr.debug_sync_status()
        
        if reconstr.images_init:
            print("\n✅ Stereo pair captured and synchronized! Entering normal spin loop...")
            # MultiThreadedExecutor, NOT rclpy.spin(): reconstruct_callback runs
            # ~200 ms+ and, on a single-threaded executor, blocked PSM pose/jaw
            # intake for that whole time.  The middleware queued the poses and
            # delivered them seconds late — the publisher measured healthy from
            # outside (30 Hz, 18-26 ms delay) while this node's pose buffer sat
            # ~3 s behind wall clock, so every image stamp fell after the newest
            # buffered pose and the tool motion read 0.  The pose/jaw
            # subscriptions live in self._fast_cbg (Reentrant), so with >1
            # thread they are drained concurrently with reconstruction.
            executor = MultiThreadedExecutor(num_threads=4)
            executor.add_node(reconstr.node)
            try:
                executor.spin()
            finally:
                executor.shutdown()

        print("Stereo pair captured!")


    except KeyboardInterrupt:
        reconstr.node.get_logger().info("Keyboard interrupt detected. Shutting down processor...")
    finally:
        # Clean up resources safely
        reconstr.node.destroy_node()
        if rclpy.ok():
            
            rclpy.shutdown()


