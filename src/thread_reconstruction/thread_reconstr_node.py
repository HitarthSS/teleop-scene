#! /home/autosurg/anaconda3/envs/thread_ros/bin/python

import rospy
import message_filters
import cv_bridge
from tf2_ros import Buffer, TransformListener
from tf.transformations import *
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from __thread_reconstruction.srv import \
    Reconstruct, ReconstructResponse, \
    TraceThread, TraceThreadResponse, \
    RecordPSMPath, Grasp

import os
import cv2
import numpy as np

from src.thread_reconstruction.segmenter import SAMSegmenter
from src.thread_reconstruction.keypt_selection import keypt_selection
from src.thread_reconstruction.keypt_ordering import keypt_ordering
from src.thread_reconstruction.optim import optim
from src.thread_reconstruction.utils import *
import pdb

# Drop into pdb at every error / degenerate return in this module, so a
# silently-skipped frame stops instead of being swallowed by the caller's
# fallback path.  Export THREAD_RECON_BREAK=0 to disable every breakpoint
# in the package at once (unattended runs, live ROS sessions).
DEBUG_BREAK_ON_ERROR = os.environ.get("THREAD_RECON_BREAK", "1") != "0"


CAPTURE = 0
GUIDE = 1
GRASP = 2

SIMPLE = 0
ROBUST = 1

MM_TO_M = 1/1000
PSM = 2

class GraspError(Exception):
    def __init__(self, message="Grasp error has occurred"):
        super().__init__(message)

class ThreadReconstrNode:
    def __init__(self, grasp_service, record_service):
        # Save service proxies
        self.grasp_service_handle = grasp_service
        self.record_service = record_service
        
        # Set up camera params
        calib = os.path.dirname(__file__) + "/../../dvrk_camera_calibration_before_rectify.yaml"#"/../../camera_calibration_sarah.yaml"
        cv_file = cv2.FileStorage(calib, cv2.FILE_STORAGE_READ)
        K1 = cv_file.getNode("K1").mat()
        D1 = cv_file.getNode("D1").mat()
        K2 = cv_file.getNode("K2").mat()
        D2 = cv_file.getNode("D2").mat()
        R = cv_file.getNode("R").mat()
        T = cv_file.getNode("T").mat()
        ImageSize = cv_file.getNode("ImageSize").mat()
        img_size = (int(ImageSize[0][1]), int(ImageSize[0][0]))
        new_size = (int(ImageSize[0][1]/2), int(ImageSize[0][0]/2))

        R1, R2, self.P1, self.P2, self.Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=new_size)
        
        print(self.P1)
        print(self.P2)
        print(self.Q)

        self.cam2img = self.P1[:,:-1]
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(K1, D1, R1, self.P1, new_size, cv2.CV_16SC2)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(K2, D2, R2, self.P2, new_size, cv2.CV_16SC2)

        # Segmenter
        self.segmenter = SAMSegmenter("cpu", "vit_h")

        # Set up transform listers
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf)

        # Set up pubs
        self.reconstr_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)
        self.grasp_pub = rospy.Publisher("/thread_reconstr/grasp_pose", PoseStamped, queue_size=10)

        # Set up rectified stereo img listener
        self.left = None
        self.right = None
        self.stamp = None
        self.bridge = cv_bridge.CvBridge()
        self.rectified_left_sub = message_filters.Subscriber("/stereo/left/image", Image, queue_size=1)
        self.rectified_right_sub = message_filters.Subscriber("/stereo/right/image", Image, queue_size=1)
        self.sync_stereo = message_filters.ApproximateTimeSynchronizer([self.rectified_left_sub, self.rectified_right_sub], queue_size=1, slop=0.2)
        self.sync_stereo.registerCallback(self.get_stereo_pair)

        # Store reconstructions
        self.spline, self.reliability = None, None
    
    def get_stereo_pair(self, left, right):
        try:
            self.left = self.bridge.imgmsg_to_cv2(left, desired_encoding="rgb8")
            self.right = self.bridge.imgmsg_to_cv2(right, desired_encoding="rgb8")
            self.stamp = left.header.stamp
        except Exception as e:
            rospy.logwarn("thread_reconstr_node: Image pair can't be saved: %s" % (e,))
    
    def reconstruct(self, request):
        if self.left is None or self.right is None or self.stamp is None:
            rospy.logwarn("thread_reconstr_node: No images received yet")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None
        
        img1 = cv2.remap(self.left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        img2 = cv2.remap(self.right, self.map2x, self.map2y, cv2.INTER_LINEAR)

        # # Gamma correction
        # gamma = 0.9
        # img1 = np.uint8((img1/255) ** gamma * 255)
        # img2 = np.uint8((img2/255) ** gamma * 255)

        # # Contrast adjustment
        # alpha = 2
        # beta = -255/2
        # img1[..., 2] = cv2.convertScaleAbs(img1[..., 2], alpha=alpha, beta=beta)
        # img2[..., 2] = cv2.convertScaleAbs(img2[..., 2], alpha=alpha, beta=beta)
        mask1 = self.segmenter.segmentation(img1)
        mask2 = self.segmenter.segmentation(img2)

        stack_mask1 = np.stack((mask1, mask1, mask1), axis=-1)
        img1 = np.where(stack_mask1>0, img1, 0)
        stack_mask2 = np.stack((mask2, mask2, mask2), axis=-1)
        img2 = np.where(stack_mask2>0, img2, 0)
        
        img1 = np.float32(img1)
        img2 = np.float32(img2)
        
        # Perform reconstruction
        img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents = keypt_selection(img1, img2, mask1, mask2, self.Q)
        img_3D, keypoints, grow_paths, order = keypt_ordering(img1, img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents)
        self.spline, self.reliability = optim(img1, mask1, mask2, img_3D, keypoints, grow_paths, order, self.cam2img, self.P1, self.P2)

        self.publish_spline()

        return ReconstructResponse()

    def publish_spline(self):
        # Publish spline to rviz
        params = np.linspace(0, 1, 50)
        spline_pts = self.spline(params) * MM_TO_M
        pt_reliabs = self.reliability(params)
        marker = Marker()
        marker.header.frame_id = "dvrk_cam"
        marker.header.stamp = rospy.Time()
        marker.ns = "thread_reconstr"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.001 # Default 1 mm thickness
        marker.points = []
        marker.colors = []
        for pt, rel in zip(spline_pts, pt_reliabs):
            point = Point()
            point.x, point.y, point.z = pt[0], pt[1], pt[2]
            marker.points.append(point)
            color = ColorRGBA()
            # red-yellow-green color scale
            color.a = 1.0
            color.r = 1.0 - np.clip(2*rel - 1.0, 0.0, 1.0)
            color.g = np.clip(2*rel, 0.0, 1.0)
            color.b = 0.0
            marker.colors.append(color)
        marker.lifetime = rospy.Duration(0)
        marker.frame_locked = True
        self.reconstr_pub.publish(marker)

        # Add markers to denote curve sections
        params = np.linspace(0, 1, 11)
        spline_pts = self.spline(params) * MM_TO_M
        marker = Marker()
        marker.header.frame_id = "dvrk_cam"
        marker.header.stamp = rospy.Time()
        marker.ns = "thread_reconstr"
        marker.id = 1
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.003 # Default 3 mm thickness
        marker.points = []
        marker.colors = []
        for pt, s in zip(spline_pts, params):
            point = Point()
            point.x, point.y, point.z = pt[0], pt[1], pt[2]
            marker.points.append(point)
            color = ColorRGBA()
            # black to white
            color.a = 1.0
            color.r = s
            color.g = s
            color.b = s
            marker.colors.append(color)
        marker.lifetime = rospy.Duration(0)
        marker.frame_locked = True
        self.reconstr_pub.publish(marker)
    
    def grasp_service(self, pose, primitive):
        response = self.grasp_service_handle(pose, primitive)
        if response == None:
            raise GraspError()

    def simple_grasp(self, grasp_s):
        self.grasp_service(self.get_pose(grasp_s), CAPTURE)
        self.grasp_service(self.get_pose(grasp_s), GRASP)

    def robust_grasp(self, grasp_s):
        SAMPLES = 100
        BASE_GUIDE_PROB = 0.99

        # Compute capture point
        candidates = np.linspace(0, 1, SAMPLES)
        candidate_idxs = np.arange(SAMPLES)
        reliabilities = self.reliability(candidates)
        grasp_idx = np.round((SAMPLES-1) * grasp_s)
        distances = np.abs(grasp_idx - candidate_idxs)
        guide_probs = BASE_GUIDE_PROB ** distances
        grasp_probs = reliabilities * guide_probs
        capture_idx = np.argmax(grasp_probs)
        capture_s = candidates[capture_idx]
        
        # Get guide waypoints
        guide_dir = np.sign(grasp_idx - capture_idx)
        guide_s = [] if guide_dir == 0 else candidates[int(capture_idx-guide_dir):int(grasp_idx):int(guide_dir)]

        # Execute
        self.grasp_service(self.get_pose(capture_s), CAPTURE)
        for s in guide_s:
            self.grasp_service(self.get_pose(s), GUIDE)
        self.grasp_service(self.get_pose(grasp_s), GRASP)
    
    def get_pose(self, s):
        # Get grasp point in camera frame
        point = self.spline(s) * MM_TO_M

        # Choose grasp pose aligned with thread and easy for PSM to grasp
        dspline = self.spline.derivative()
        z = dspline(s)

        ANG_THRESH = 30
        # Choose best y direction
        angle2cam = np.arccos(np.abs(np.dot(point/np.linalg.norm(point), z/np.linalg.norm(z))))
        # grasp from PSM direction if angle to camera is too small
        if angle2cam < np.radians(ANG_THRESH):
            pose_base = self.tf_buf.lookup_transform("dvrk_cam", "PSM%d_base" % (PSM,), rospy.Time(0))
            pos_base = np.array([pose_base.transform.translation.x, pose_base.transform.translation.y, pose_base.transform.translation.z])
            base2point = point - pos_base
            x = np.cross(base2point, z)
            y = np.cross(z, x)
        else:
            x = np.cross(point, z)
            y = np.cross(z, x)

        # Represent as matrix
        R = np.eye(4)
        R[:3, 0] = x / np.linalg.norm(x)
        R[:3, 1] = y / np.linalg.norm(y)
        R[:3, 2] = z / np.linalg.norm(z)
        # Offset on ree y-axis to handle ree-to-tip distance
        REE_TO_TIP = 0.0102 # Taken from dvrk manual
        point -= R[:3, 1] * REE_TO_TIP/2 # target point is halfway to tip distance
        # Convert to PoseStamped
        quat_cam = quaternion_from_matrix(R)
        pose = PoseStamped()
        pose.header.stamp = self.stamp
        pose.header.frame_id = "dvrk_cam"
        pose.pose.position.x = point[0]
        pose.pose.position.y = point[1]
        pose.pose.position.z = point[2]
        pose.pose.orientation.x = quat_cam[0]
        pose.pose.orientation.y = quat_cam[1]
        pose.pose.orientation.z = quat_cam[2]
        pose.pose.orientation.w = quat_cam[3]

        return pose

    def trace(self, request):
        # Republish spline, for data collection purposes
        self.publish_spline()
        
        if self.spline == None:
            rospy.logwarn("thread_reconstr_node: No reconstruction found")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None

        # Get grasp policy
        if request.policy == SIMPLE:
            grasp = self.simple_grasp
        elif request.policy == ROBUST:
            grasp = self.robust_grasp
        else:
            rospy.logwarn("thread_reconstr_node: Grasp policy not recognized")

        # Construct trace path
        num_waypoints = 1 + np.abs(request.stop-request.start) / request.step
        direction = np.sign(request.stop-request.start)
        waypoint_s = request.start + np.arange(0, num_waypoints)*request.step*direction

        # Execute trace
        if request.record:
            self.record_service(True, 0.005)

        try:
            for s in waypoint_s:
                grasp(s)
        except GraspError as e:
            rospy.logwarn("Trajectory failed")

        if request.record:
            self.record_service(False, 0.005)
        return TraceThreadResponse()



if __name__ == "__main__":
    rospy.init_node("thread_reconstr_node", anonymous=True)
    
    rospy.loginfo("thread_reconstr_node: Waiting for grasp and record services...")
    rospy.wait_for_service("exec_grasp_node/grasp")
    grasp_service = rospy.ServiceProxy("exec_grasp_node/grasp", Grasp)
    rospy.wait_for_service("record_psm_path_node/toggle_recording")
    record_service = rospy.ServiceProxy("record_psm_path_node/toggle_recording", RecordPSMPath)
    rospy.loginfo("thread_reconstr_node: Services found!")
    
    node = ThreadReconstrNode(grasp_service, record_service)
    reconstr_service = rospy.Service('thread_reconstr_node/reconstruct', Reconstruct, node.reconstruct)
    trace_service = rospy.Service('thread_reconstr_node/trace', TraceThread, node.trace)


    # import time
    # time.sleep(2.0)

    # plt.figure(0)
    # plt.imshow(cv2.remap(node.left, node.map1x, node.map1y, cv2.INTER_LINEAR))
    # plt.show()
    # points = get_camera2markers_pose(cv2.remap(node.left, node.map1x, node.map1y, cv2.INTER_LINEAR), node.P1)
    
    # for point in points:

    #     z = np.array([0, 1, 0])
        
    #     x = np.cross(point, z)#np.cross(base2point, z)
    #     y = np.cross(z, x)

    #     # Represent as matrix
    #     R = np.eye(4)
    #     R[:3, 0] = x / np.linalg.norm(x)
    #     R[:3, 1] = y / np.linalg.norm(y)
    #     R[:3, 2] = z / np.linalg.norm(z)
    #     # Offset on ree y-axis to handle ree-to-tip distance
    #     REE_TO_TIP = 0.0102 # Taken from dvrk manual
    #     point -= R[:3, 1] * REE_TO_TIP # target point is halfway to tip distance

    #     # offset = 0.01
    #     # point[2] += offset

    #     # Convert to PoseStamped
    #     quat_cam = quaternion_from_matrix(R)
    #     pose = PoseStamped()
    #     pose.header.stamp = node.stamp
    #     pose.header.frame_id = "dvrk_cam"
    #     pose.pose.position.x = point[0]
    #     pose.pose.position.y = point[1]
    #     pose.pose.position.z = point[2]
    #     pose.pose.orientation.x = quat_cam[0]
    #     pose.pose.orientation.y = quat_cam[1]
    #     pose.pose.orientation.z = quat_cam[2]
    #     pose.pose.orientation.w = quat_cam[3]

    #     node.grasp_service(pose, CAPTURE)
    #     node.grasp_service(pose, GRASP)

    rospy.spin()
