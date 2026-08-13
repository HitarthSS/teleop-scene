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


# get prev trial path
from pathlib import Path
# change prev trial name
import re

# warm start splien
import scipy.interpolate as interp
import os
import pdb

# Drop into pdb at every error / degenerate return in this module, so a
# silently-skipped frame stops instead of being swallowed by the caller's
# fallback path.  Export THREAD_RECON_BREAK=0 to disable every breakpoint
# in the package at once (unattended runs, live ROS sessions).
DEBUG_BREAK_ON_ERROR = os.environ.get("THREAD_RECON_BREAK", "1") != "0"



class WarmStart(Node):
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        super().__init__('warm_start')
        self.jaw_closed = 5
        self.prev_thread = None
        self.prev_thread_keypts = None
        self.transformed_thread = None

        self.psm1_fake_pose_received = False
        self.psm2_fake_pose_received = False
        self.psm1_jaw_received = False
        self.psm2_jaw_received = False
        self.psm1_current_cam_T = np.eye(4)
        self.psm2_current_cam_T = np.eye(4)
        self.psm1_current_fake_T = np.eye(4)
        self.psm2_current_fake_T = np.eye(4)

        self.fake_H_cam_base_1 = np.array([
            [-0.59235094, -0.68685556, -0.42112919,  0.04430956],
            [-0.64149465,  0.71832064, -0.26925838, -0.02940214],
            [ 0.4874474 ,  0.11065667, -0.86611208, -0.03066634],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
            ])
        
        self.fake_H_cam_base_2 = np.array([
            [-0.80336077,  0.34479305, -0.48551955, -0.12610999],
            [ 0.37840734,  0.92512489,  0.03085153,  0.03258377],
            [ 0.45980361, -0.15893926, -0.87368127,  0.12214784],
            [ 0.        ,  0.        ,  0.        ,  1.        ]        
            ])
        
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

        self.init_ros()

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()
        self._wait_for_ros_data()

    def init_ros(self):
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
            partial(self.fake_pose_callback, arm_id=2),
            10
        )

        # camera to robot base transform from the particle filter, the axis rotation 
        self.psm2_jaw_sub = self.create_subscription(
            JointState,
            '/PSM2/jaw/measured_js',
            partial(self.jaw_callback, arm_id=2),
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
            partial(self.fake_pose_callback, arm_id=1),
            10
        )

        self.psm1_jaw_sub = self.create_subscription(
            JointState,
            '/PSM1/jaw/measured_js',
            partial(self.jaw_callback, arm_id=1),
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
    
    def jaw_callback(self, msg: JointState, arm_id):
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

    # def pose_callback(self, msg: PoseStamped, arm_id):
    #     pos = msg.pose.position
    #     ori = msg.pose.orientation
    #     pose = [pos.x*1000, pos.y*1000, pos.z*1000, ori.x, ori.y, ori.z, ori.w]

    #     # self.get_logger().info(
    #     #     f"Pose received: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})\n \
    #     #     quat received: ({ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}), {ori.w:.3f})\n"
    #     # )
    #     T = self.pose_to_matrix(pose)
        
    #     # store it if needed
    #     if arm_id==2:
    #         self.psm2_current_pose = pose
    #         self.psm2_current_T = T
    #         self.psm2_current_cam_T = self.H_cam_base_2 @ T
    #         self.psm2_pose_received = True
            

    #     elif arm_id==1:
    #         self.psm1_current_pose = pose
    #         self.psm1_current_T = T
    #         self.psm1_current_cam_T = self.H_cam_base_1 @ T
    #         self.psm1_pose_received = True

    def fake_pose_callback(self, msg: PoseStamped, arm_id):
        pos = msg.pose.position
        ori = msg.pose.orientation
        pose = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]

        # self.get_logger().info(
        #     f"Pose received: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})\n \
        #     quat received: ({ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}), {ori.w:.3f})\n"
        # )
        T = self.pose_to_matrix(pose)
        # if self.pose_cam_base1 is not None and self.pose_cam_base2 is not None:
        #     H_cam_base_1 = self.pose_to_matrix(self.pose_cam_base1)
        #     H_cam_base_2 = self.pose_to_matrix(self.pose_cam_base2)
            
        # else:
        H_cam_base_1, H_cam_base_2 = self.fake_H_cam_base_1, self.fake_H_cam_base_2
        # store it if needed
        if arm_id==2:
            psm2_current_pose = pose
            psm2_current_T = T
            psm2_current_T[:3, 3] *= 1000
            self.psm2_current_fake_T = H_cam_base_2 @ (T @ self.cam_base_coord_change)
            self.psm2_current_fake_T[:3, 3] += np.array([-123.70179159,   36.65738593,  121.17085326])
            self.psm2_fake_pose_received = True
            # psm2_curr_pub = copy.copy(self.psm2_current_fake_T)
            # psm2_curr_pub[:3, 3] *= 0.001
            # self.psm2_fake_T_pub.publish(self._matrix_to_msg(psm2_curr_pub))

        elif arm_id==1:
            psm1_current_pose = pose
            psm1_current_T = T
            psm1_current_T[:3, 3] *= 1000
            self.psm1_current_fake_T = H_cam_base_1 @ (T @ self.cam_base_coord_change)
            self.psm1_current_fake_T[:3, 3] += np.array([49.02675771, -31.27579621, -38.9100777])
            self.psm1_fake_pose_received = True

            # psm1_curr_pub = copy.copy(self.psm2_current_fake_T)
            # psm1_curr_pub[:3, 3] *= 0.001
            # self.psm1_fake_T_pub.publish(self._matrix_to_msg(psm1_curr_pub))

    def thread_grasp_index(self, thread, grasp_H):
        '''
        find the index on the thread that is the closest to the grasp point
        '''
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
    
    def add_prev_thread(self, thread_file_path:str):
        thread_specs_path = thread_file_path.replace("spline.pkl", "spline_specs.pkl")

        prev_thread_path = self.decrement_trial(thread_file_path)
        prev_thread_specs_path = self.decrement_trial(thread_specs_path)
        print("previous thread file: ", prev_thread_path)
        print("previous thread specs file: ", prev_thread_specs_path)

        thread_path = Path(prev_thread_path)
        if Path(prev_thread_path).exists():
            with open(prev_thread_path, 'rb') as f:
                data = pickle.load(f)
                warm_start_spline = data.get('spline')

            with open(prev_thread_specs_path, 'rb') as f:
                data = pickle.load(f)
                warm_start_keypts = data.get('keypt_s')
        else:
            warm_start_spline, warm_start_keypts = None, None

        return warm_start_spline, warm_start_keypts

        self.add_prev_to_curr(thread_file_path)
        warm_keypoints = spline_npy(warm_start_keypts)
        # rotate around T_prev's origin, then translate
        warm_keypoints = (self.trans[:3, :3] @ (warm_keypoints - self.T_prev[:3, 3]).T).T + self.T_prev[:3, 3] + self.trans[:3, 3]
        
        
        return warm_keypoints, warm_start_keypts

    def add_prev_to_curr(self, thread_file_path:str):
        folder = Path(thread_file_path).parent / "grasp"
        folder.mkdir(parents=True, exist_ok=True)

        path = str(thread_file_path)
        match = re.search(r'trial_(\d+)', path)
        if not match:
            raise ValueError("No trial number found")

        trial_num = int(match.group(1))
        pose_file = folder / f"trial_{trial_num}_goal_H_cam_matrices.pkl" # use prev grasp H saved
        prev_pose_file = self.decrement_trial(pose_file)

        if Path(prev_pose_file).exists():
            with open(prev_pose_file, 'rb') as f: # if saved as a pickle file
                data = pickle.load(f)
                T_prev = data.get('goal_H_cam_matrices')[0] # the first one is the selected one
            psm1_jaw = self.psm1_current_jaw
            psm2_jaw = self.psm2_current_jaw
            
            psm1_dis = np.inf
            psm2_dis = np.inf
            if psm1_jaw / np.pi * 180 < self.jaw_closed:
                __, psm1_dis = self.thread_grasp_index(self.prev_thread, self.psm1_current_fake_T)
            if psm2_jaw / np.pi * 180 < self.jaw_closed:
                __, psm2_dis = self.thread_grasp_index(self.prev_thread, self.psm2_current_fake_T)

            grasp_psm_idx = np.argmin([psm1_dis, psm2_dis])
            # pick the psm that is currently closer to the thread with jaw closed
            T_curr = self.psm1_current_fake_T if grasp_psm_idx == 0 else self.psm2_current_fake_T 
            T_curr[:3, 3] += T_curr[:3, 1] *5 # positive towards the tip 5mm offset from the end effector pose
            print(f"~~~PSM {grasp_psm_idx + 1} is grasping thread~~~")
            self.psm = grasp_psm_idx + 1
            # psm1_pose[:3, 3] *= 1000
            # psm2_pose[:3, 3] *= 1000
            trans = np.identity(4)
            rot = T_curr[:3, :3] @ T_prev[:3, :3].T @ np.eye(3)
            translate = (T_curr[:3, 3] - T_prev[:3, 3])
            trans[:3, :3] = rot
            trans[:3, 3] = translate

            self.T_curr = T_curr
            self.T_prev = T_prev
            self.trans = trans
        else:
            print("Previous pose file does not exist")
            psm1_pose = self.psm1_current_fake_T
            psm1_jaw = self.psm1_current_jaw
            psm2_pose = self.psm2_current_fake_T
            psm2_jaw = self.psm2_current_jaw

            self.T_curr = None
            self.T_prev = None
            self.trans = None
    
    def decrement_trial(self, path):
        path = str(path)
        match = re.search(r'trial_(\d+)', path)
        if not match:
            raise ValueError("No trial number found")

        n = int(match.group(1))
        new_n = n - 1

        # Replace ALL occurrences of trial_n with trial_{n-1}
        new_path = re.sub(r'trial_\d+', f"trial_{new_n}", path)

        return new_path
    
    def pose_to_matrix(self, pose):
        t = pose[:3]
        q = pose[3:]

        R_mat = Rot.from_quat(q).as_matrix()

        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t

        return T

    def warm_start_keypoints(self, mask, keypoints, spline_file, P1, speedy=False, max_dist = 30, thresh=7, max_neigh=2, T_curr_radius=25, n_samples=200): #max dist can be better tuned
        self.prev_thread, self.prev_thread_keypts = self.add_prev_thread(spline_file)

        if self.prev_thread is None: # return nothing if there is not previous thread found
            new_keypoints, order, new_warm_start_keypts = None, None, None
            return new_keypoints, order, new_warm_start_keypts
        
        # ctrl = np.asarray(self.prev_thread.c)
        # new_ctrl = (self.trans[:3, :3] @ (ctrl - self.T_prev[:3, 3]).T).T + self.T_prev[:3, 3] + self.trans[:3, 3]
        # self.transformed_thread = interp.BSpline(self.prev_thread.t, new_ctrl, self.prev_thread.k)
        warm_keypoints = self.transformed_thread(self.prev_thread_keypts)

        # mask out warm keypoints far from T_curr's grasp position (in 3D camera frame)
        far_from_grasp = np.linalg.norm(self.transformed_thread - self.T_curr[:3, 3], axis=1) > T_curr_radius

        '''
        match warm start spline keypoints to the closest keypoints on current image, this will also order the keypoints since keypt_s is already ordered
        output keypoints, order
        '''
        new_keypoints = []

        # warm_start_idx = np.linspace(warm_start_keypts[0], warm_start_keypts[-1], 20)
        # warm_keypoints = warm_start_spline(warm_start_idx)

        # project to image coordinates
        # concatenate for 2d points, don't when we're using the points from the 3d spline
        aug_pts = np.concatenate((warm_keypoints, np.ones((warm_keypoints.shape[0], 1))), axis=1)
        proj_pts = (P1 @ aug_pts.T).T
        proj_pts /= proj_pts[:, 2:].copy() + 1e-7
        proj_pts[:, 2] = warm_keypoints[:, 2]   # restore original z
        old_keypoints = copy.copy(keypoints)
        warm_start_ids = []
        new_warm_start_keypts = []
        # keypoints are stored (y, x, z); reorder to (x, y, z) to match proj_pts (u, v, z)
        keypoints = np.asarray(keypoints)
        proj_pts = np.asarray(proj_pts)
        used = np.zeros(len(keypoints), dtype=bool)
        for i, pt in enumerate(proj_pts):
            if far_from_grasp[i]:
                continue  # skip warm keypoints far from T_curr
            dists = np.linalg.norm(keypoints[:, [1, 0, 2]] - pt[:3], axis=1)
            dists[used] = np.inf  # don't reuse an image keypoint
            idx = np.argmin(dists)
            print("distance between matched point", dists[idx])
            if dists[idx] > max_dist:
                continue
            print(f"matched warm keypoint idx {i} to keypoint {idx}")
            warm_start_ids.append(i)
            new_keypoints.append(copy.copy(keypoints[idx]))
            new_warm_start_keypts.append(copy.copy(self.prev_thread_keypts[i]))
            used[idx] = True
        pdb.set_trace()

        new_keypoints = np.asarray(new_keypoints) #[:, [1, 0, 2]] # swap x, y back
        n = len(new_keypoints)
        order = np.arange(n)

        if n < 4:
            print("\nnot enough warm start keypoints found\n")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None, None

        if not speedy:
            pass
            a = np.array([new_keypoints[:, 1], new_keypoints[:, 0]]).T
            b = np.array([proj_pts[warm_start_ids][:, 0], proj_pts[warm_start_ids][:, 1]]).T
            ab_pairs = np.c_[a, b]
            ab_args = ab_pairs.reshape(-1, 2, 2).swapaxes(1, 2).reshape(-1, 2)

            plt.imshow(mask, cmap="gray")
            plt.plot(*ab_args, c='pink', lw=1)
            # plt.plot(*a.T, 'bo')

            plt.scatter(proj_pts[:, 0], proj_pts[:, 1], c="r", s=3, label="warm start") # warm start spline
            plt.scatter(old_keypoints[:, 1], old_keypoints[:, 0], c='g', s=3, label="original keypoints") # original keypoints
            plt.scatter(new_keypoints[:, 1], new_keypoints[:, 0], c='b', s=5, label="matched keypoints") # keypoints matching warm start spline
            plt.legend()
            # plt.title(f"warm start frame: {}")
            plt.show()
            # plt.show(block=False)
            # plt.pause(0.5) # comment for profiling
            # plt.close()

        # pdb.set_trace()
        print(f"warm start matched keypoints {new_keypoints}\n")
        # pdb.set_trace()
        self._executor.shutdown()
        self.destroy_node()
        return new_keypoints, order, new_warm_start_keypts