import pickle
import numpy as np
import pdb 
import matplotlib.pyplot as plt
import copy

# ros messages to read psm current state
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from functools import partial
import threading
import time
# transform messages
from scipy.spatial.transform import Rotation as Rot
import scipy.interpolate as interp


# get prev trial path
from pathlib import Path
# change prev trial name
import re

# shared EKF/warp tuning constants (single source of truth)
from thread_reconstruction import ekf_params


class WarmStart(Node):
    def __init__(self, args):
        if not rclpy.ok():
            rclpy.init()
        super().__init__('warm_start')

        # settings
        self.jaw_close_thresh = 5
        # e-folding distance (mm) over which the tool's rigid motion decays
        # along the thread. Points within ~deform_radius of the grasped tool
        # tip follow the tool; far points stay anchored.  All three warps
        # (this one, SplineEKF.predict, optim's temporal-prior compensation)
        # share the ONE constant in ekf_params.py, so they always agree.
        self.deform_radius = ekf_params.DEFORM_RADIUS

        # file paths
        self.thread_file_path = None

        # object initalization
        self.prev_thread = None
        self.prev_keypts = None
        self.trans_thread = None

        # ros flags
        self.psm1_fake_pose_received = False
        self.psm2_fake_pose_received = False
        self.psm1_jaw_received = False
        self.psm2_jaw_received = False

        # ros topics
        self.psm1_current_cam_T = np.eye(4)
        self.psm2_current_cam_T = np.eye(4)
        self.psm1_current_fake_T = np.eye(4)
        self.psm2_current_fake_T = np.eye(4)

        self.fake_H_cam_base_1 = None
        self.fake_H_cam_base_2 = None
        self.init_cam2base(args)
            # [0., -1., 0., 0.], 
            # [0., 0., 1., 0.], 
            # [-1., 0., 0., 0.], 
            # [0., 0., 0., 1.], 

        self.cam_base_coord_change = np.array([
            [0., 0., -1., 0.], 
            [-1., 0., 0., 0.], 
            [0., 1., 0., 0.], 
            [0., 0., 0., 1.], 
        ])

        self.curr_T = None
        self.prev_T = None
        self.trans = None

    def init_cam2base(self, args):
        calib = args.psm_calibrate
        if Path(calib).exists():
            data = np.load(calib)

            self.fake_H_cam_base_1 = data['PSM1'].copy()
            self.fake_H_cam_base_1[:3, 3] *= 1000
            self.fake_H_cam_base_2 = data['PSM2'].copy()
            self.fake_H_cam_base_2[:3, 3] *= 1000
        # pdb.set_trace()

    def refresh_warm_start(self, prev_thread, prev_keypts, curr_T, prev_T):

        if prev_thread is not None and prev_T is not None and len(prev_thread.t) > 8:
            trans = np.identity(4)
            rot = curr_T[:3, :3] @ prev_T[:3, :3].T @ np.eye(3)
            translate = (curr_T[:3, 3] - prev_T[:3, 3])
            trans[:3, :3] = rot
            trans[:3, 3] = translate
            ctrl = np.asarray(prev_thread.c)

            # Distance-weighted rigid warp about the grasp (prev_T translation):
            # BOTH the translation and the rotation decay with distance from the
            # grasp, w = exp(-d/deform_radius) — only the near-tool portion of
            # the thread follows the tool; farther parts lag behind (slack/
            # anchored thread does not get dragged whole).  At w=1 (at the
            # grasp) this is exactly the full rigid transform
            # curr_T @ inv(prev_T); at w=0 (far end) the thread does not move.
            pivot   = prev_T[:3, 3]
            delta   = ctrl - pivot
            rot_dev = (rot @ delta.T).T - delta              # (R - I)(p - pivot)
            dist    = np.linalg.norm(delta, axis=1)
            w       = np.exp(-dist / max(self.deform_radius, 1e-6))[:, None]
            new_ctrl = ctrl + w * (translate + rot_dev)

            # ── Warp diagnostic ───────────────────────────────────────────────
            # Now that the TRANSLATION is distance-weighted too, a grasp that
            # sits far from the thread in 3-D (a depth/calibration error is
            # invisible in the 2-D projection!) drives w -> 0 everywhere and the
            # warp silently becomes a no-op.  Print the grasp->thread distance
            # and the resulting weights so that case is obvious: if w_max is
            # small, the pivot is not on the thread, not "the thread didn't
            # move".
            disp = np.linalg.norm(new_ctrl - ctrl, axis=1)
            print(f"warm warp: |t|={np.linalg.norm(translate):.2f}mm  "
                  f"grasp->ctrl dist min/mean/max="
                  f"{dist.min():.1f}/{dist.mean():.1f}/{dist.max():.1f}mm  "
                  f"(deform_radius={self.deform_radius:.0f})  "
                  f"w min/max={w.min():.3f}/{w.max():.3f}  "
                  f"ctrl displacement min/max={disp.min():.2f}/{disp.max():.2f}mm")
            if w.max() < 0.2 and np.linalg.norm(translate) > 1e-6:
                print("warm warp: WARNING — grasp is far from every control "
                      f"point (nearest {dist.min():.1f}mm >> deform_radius "
                      f"{self.deform_radius:.0f}mm), so the tool motion barely "
                      "moves the thread.  Check the gripper DEPTH (z) "
                      "calibration or raise deform_radius.")

            transformed_thread = interp.BSpline(prev_thread.t, new_ctrl, prev_thread.k)

            self.prev_thread = prev_thread
            self.prev_keypts = prev_keypts
            self.trans_thread = transformed_thread
            self.curr_T = curr_T 
            self.prev_T = prev_T
            self.trans = trans

        else:
            transformed_thread = None
            self.prev_thread = None
            self.prev_keypts = None
            self.trans_thread = None
            self.curr_T = None 
            self.prev_T = None
            self.trans = None

        return self.trans_thread, self.prev_keypts
    
    ##############################################
    ### Functions not needed in the ros method ###
    ##############################################
    '''
    def shut_down_node(self):
        self._executor.shutdown()
        self.destroy_node()

    def init_ros(self, args):
        
        self.init_ros_topics()

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()
        self._wait_for_ros_data()

    def _thread_grasp_index(self, thread, grasp_H):
        
        # find the index on the thread that is the closest to the grasp point
        
        goal_pos = grasp_H[:3, 3]
        thread_points = thread(np.linspace(0, 1, 200))
        closest = np.inf
        thread_idx = None
        for idx, point in enumerate(thread_points):
            dis = np.linalg.norm(point - goal_pos)
            if dis <= closest:
                thread_idx = idx
                closest = dis
    
        return thread_idx, closest
    
    
    def _decrement_trial(self, path):
        path = str(path)
        match = re.search(r'trial_(\d+)', path)
        if not match:
            raise ValueError("No trial number found")

        n = int(match.group(1))
        new_n = n - 1

        # Replace ALL occurrences of trial_n with trial_{n-1}
        new_path = re.sub(r'trial_\d+', f"trial_{new_n}", path)

        return new_path

    def _pose_to_matrix(self, pose):
        t = pose[:3]
        q = pose[3:]

        R_mat = Rot.from_quat(q).as_matrix()

        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t

        return T

    def init_prev_pose(self, thread_file_path, use_goal_file=True):
        # locate previous trial
        folder = Path(thread_file_path).parent / "grasp"
        folder.mkdir(parents=True, exist_ok=True)

        path = str(thread_file_path)
        match = re.search(r'trial_(\d+)', path)
        if not match:
            raise ValueError("No trial number found")

        trial_num = int(match.group(1))
        if use_goal_file:
            pose_file = folder / f"trial_{trial_num}_goal_H_cam_matrices.pkl"
            prev_pose_file = self._decrement_trial(pose_file)
            if Path(prev_pose_file).exists():
                print("using previous goal file for prev_T")
                with open(prev_pose_file, 'rb') as f: # if saved as a pickle file
                    data = pickle.load(f)
                    psm_T = data.get('goal_H_cam_matrices')[0] # the first one is the selected one
            else:
                print("Previous goal file does not exist")
                psm_T = None
                psm_jaw = None
                psm = None
                return psm_T, psm_jaw, psm

            psm1_jaw = self.psm1_current_jaw
            psm2_jaw = self.psm2_current_jaw
            
            psm1_dis = np.inf
            psm2_dis = np.inf
            if psm1_jaw / np.pi * 180 < self.jaw_close_thresh:
                __, psm1_dis = self._thread_grasp_index(self.prev_thread, self.psm1_current_fake_T)
            if psm2_jaw / np.pi * 180 < self.jaw_close_thresh:
                __, psm2_dis = self._thread_grasp_index(self.prev_thread, self.psm2_current_fake_T)

            grasp_psm_idx = np.argmin([psm1_dis, psm2_dis])
            # pick the psm that is currently closer to the thread with jaw closed
            print(f"~~~PSM {grasp_psm_idx + 1} is grasping thread~~~")
            psm = grasp_psm_idx + 1     

            return psm_T, psm_jaw, psm
        
        else:
            pose_file = folder / f"trial_{trial_num}_pose.pkl" # use prev grasp H saved
            prev_pose_file = self._decrement_trial(pose_file)
            if Path(prev_pose_file).exists():
                print("using previous pose file for prev_T")
                with open(prev_pose_file, 'rb') as f: # if saved as a pickle file
                    data = pickle.load(f)
                    psm_T = data.get('PSM_T')
                    psm_jaw = data.get('PSM_jaw')
                    psm = data.get('active_PSM')
                psm_T[:3, 3] *= 1000
            else:
                print("Previous pose file does not exist")
                psm_T = None
                psm_jaw = None
                psm = None

            return psm_T, psm_jaw, psm

    def _add_prev_thread(self, thread_file_path):
        thread_specs_path = thread_file_path.replace("thread.pkl", "thread_specs.pkl")

        prev_thread_path = self._decrement_trial(thread_file_path)
        prev_thread_specs_path = self._decrement_trial(thread_specs_path)
        print("previous thread file: ", prev_thread_path)
        print("previous thread specs file: ", prev_thread_specs_path)

        if Path(prev_thread_path).exists():
            with open(prev_thread_path, 'rb') as f:
                data = pickle.load(f)
                warm_start_thread = data.get('thread')

            with open(prev_thread_specs_path, 'rb') as f:
                data = pickle.load(f)
                warm_start_keypts = data.get('keypt_s')
        else:
            warm_start_thread, warm_start_keypts = None, None

        # self.warm_start_thread = warm_start_thread
        # self.warm_start_keypts = warm_start_keypts
        return warm_start_thread, warm_start_keypts

    def update_prev_to_curr(self, thread_file_path:str, use_goal_file=True):
        prev_T, prev_jaw, psm = self.init_prev_pose(thread_file_path, use_goal_file) # returns none for all if no file found
        if prev_T is not None:
            curr_T = self.psm1_current_fake_T if psm == 1 else self.psm2_current_fake_T 
            # curr_T[:3, 3] += curr_T[:3, 1] *5 # positive towards the tip 5mm offset from the end effector pose

            trans = np.identity(4)
            rot = curr_T[:3, :3] @ prev_T[:3, :3].T @ np.eye(3)
            translate = (curr_T[:3, 3] - prev_T[:3, 3])
            trans[:3, :3] = rot
            trans[:3, 3] = translate

            # self.curr_T = curr_T
            # self.prev_T = prev_T
            # self.trans = trans
            print("Prev and curr pose updated")
        else:
            print("Previous pose file does not exist, prev and curr pose not updated")
            # self.curr_T = None
            # self.prev_T = None
            # self.trans = None
            curr_T = None
            prev_T = None
            trans = None
        print(f"transformation matrix \n{trans}")
        return curr_T, prev_T, trans

    def pose_callback(self, msg: PoseStamped, arm_id):
        pos = msg.pose.position
        ori = msg.pose.orientation
        pose = [pos.x*1000, pos.y*1000, pos.z*1000, ori.x, ori.y, ori.z, ori.w]

        # self.get_logger().info(
        #     f"Pose received: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})\n \
        #     quat received: ({ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}), {ori.w:.3f})\n"
        # )
        T = self.pose_to_matrix(pose)
        
        # store it if needed
        if arm_id==2:
            self.psm2_current_pose = pose
            self.psm2_current_T = T
            self.psm2_current_cam_T = self.H_cam_base_2 @ T
            self.psm2_pose_received = True
            

        elif arm_id==1:
            self.psm1_current_pose = pose
            self.psm1_current_T = T
            self.psm1_current_cam_T = self.H_cam_base_1 @ T
            self.psm1_pose_received = True

                def init_ros_topics(self):
        # self.psm2_pose_sub = self.create_subscription(
        #     PoseStamped,
        #     # '/PSM2/measured_cp',
        #     '/particle_filter/PSM2/position_cartesian_current',
        #     partial(self.pose_callback, arm_id=2),
        #     10
        # )
        self.psm2_pose_fake_sub = self.create_subscription( # fake pose for the dvrk's published poses. 
            PoseStamped,
            '/PSM2/measured_cp',
            # '/particle_filter/PSM2/position_cartesian_current',
            partial(self._fake_pose_callback, arm_id=2),
            10
        )

        # camera to robot base transform from the particle filter, the axis rotation 
        self.psm2_jaw_sub = self.create_subscription(
            JointState,
            '/PSM2/jaw/measured_js',
            partial(self._jaw_callback, arm_id=2),
            10
        )

        # self.psm1_pose_sub = self.create_subscription(
        #     PoseStamped,
        #     # '/PSM1/measured_cp',
        #     '/particle_filter/PSM1/position_cartesian_current',
        #     partial(self.pose_callback, arm_id=1),
        #     10
        # )

        self.psm1_pose_fake_sub = self.create_subscription( # fake pose for the dvrk's published poses.
            PoseStamped,
            '/PSM1/measured_cp',
            # '/particle_filter/PSM1/position_cartesian_current',
            partial(self._fake_pose_callback, arm_id=1),
            10
        )

        self.psm1_jaw_sub = self.create_subscription(
            JointState,
            '/PSM1/jaw/measured_js',
            partial(self._jaw_callback, arm_id=1),
            10
        )
        self._grasp_data_ready = False

    def _wait_for_ros_data(self):
        """Block until all reconstruct topics and ros_dvrk pose_cam_base topics have been received."""
        print("Waiting for reconstruct ROS topics and ros_dvrk sync msgs...")
        while True:
            # sync_msg = self.ros_dvrk.getSyncMsg()
            # self.pose_cam_base1 = sync_msg['pose_cam_base1'] if sync_msg else None
            # self.pose_cam_base2 = sync_msg['pose_cam_base2'] if sync_msg else None

            missing = []
            if self.psm2_fake_pose_received is False: missing.append('psm2_pose')
            if self.psm2_jaw_received is False: missing.append('psm2_jaw')
            if self.psm1_fake_pose_received is False: missing.append('psm1_pose')
            if self.psm1_jaw_received is False: missing.append('psm1_jaw')
            # if self.pose_cam_base1 is None: missing.append('pose_cam_base1')
            # if self.pose_cam_base2 is None: missing.append('pose_cam_base2')

            if not missing:
                break
            print(f"  waiting for: {', '.join(missing)}")
            time.sleep(0.5)
        print("All reconstruct ROS topics and pose_cam_base msgs received.")
    
    def _jaw_callback(self, msg: JointState, arm_id):
        angle = msg.position[0]
        velocity = msg.velocity
        effort = msg.effort

        # self.get_logger().info(
        #     f"Jaw angle received (degrees): ({angle:.3f})\n"
        # )

        if arm_id==2:
            self.psm2_current_jaw = angle
            self.psm2_jaw_received = True

        elif arm_id==1:
            self.psm1_current_jaw = angle
            self.psm1_jaw_received = True


    def _fake_pose_callback(self, msg: PoseStamped, arm_id):
        pos = msg.pose.position
        ori = msg.pose.orientation
        pose = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]

        # self.get_logger().info(
        #     f"Pose received: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})\n \
        #     quat received: ({ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}), {ori.w:.3f})\n"
        # )
        T = self._pose_to_matrix(pose)
        # if self.pose_cam_base1 is not None and self.pose_cam_base2 is not None:
        #     H_cam_base_1 = self.pose_to_matrix(self.pose_cam_base1)
        #     H_cam_base_2 = self.pose_to_matrix(self.pose_cam_base2)
            
        # else:
        H_cam_base_1, H_cam_base_2 = self.fake_H_cam_base_1, self.fake_H_cam_base_2
        # store it if needed
        if arm_id==2:
            psm2_current_T = T
            psm2_current_T[:3, 3] *= 1000
            self.psm2_current_fake_T = H_cam_base_2 @ (T @ self.cam_base_coord_change)
            # self.psm2_current_fake_T[:3, 3] += np.array([-123.70179159,   36.6573593,  121.17085326])
            self.psm2_fake_pose_received = True
            # psm2_curr_pub = copy.copy(self.psm2_current_fake_T)
            # psm2_curr_pub[:3, 3] *= 0.001
            # self.psm2_fake_T_pub.publish(self._matrix_to_msg(psm2_curr_pub))

        elif arm_id==1:
            psm1_current_T = T
            psm1_current_T[:3, 3] *= 1000
            self.psm1_current_fake_T = H_cam_base_1 @ (T @ self.cam_base_coord_change)
            # self.psm1_current_fake_T[:3, 3] += np.array([49.02675771, -31.27579621, -38.9100777])
            self.psm1_fake_pose_received = True

            # psm1_curr_pub = copy.copy(self.psm2_current_fake_T)
            # psm1_curr_pub[:3, 3] *= 0.001
            # self.psm1_fake_T_pub.publish(self._matrix_to_msg(psm1_curr_pub))

    '''