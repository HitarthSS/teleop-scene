import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.interpolate as interp
import cv2
import os
import torch
import time
from collections import deque

import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


from src.thread_reconstruction.segmenter import UNetSegmenter
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


"""
img1: RGB left camera image (np array)
img2: RGB right camera image (np array)
calib: filename for camera calibration file (string)
segmenter: segmentation object (see segmenter.py)
"""
class fitEvalNode(Node):
    def __init__(self, left_topic, right_topic, calib, segmenter):
        super().__init__('image_processing_node')
        self.calib = calib
        self.segmenter = segmenter
        self.hz = 0 # processing speed
        self.times = deque(maxlen=10)
        self.last = None
        self.bridge = CvBridge()
        self.image_received = False

        self.sub_l = Subscriber(self,
                                Image,
                                left_topic,
        )        
        self.sub_r = Subscriber(self,
                                Image,
                                right_topic,
        )

        self.ts = ApproximateTimeSynchronizer(
            [self.sub_l, self.sub_r],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.callback)

        self.get_logger().info("Fit eval node started.")

    def callback(self, sub_l_img, sub_r_img):
        try:
            img_l = self.bridge.imgmsg_to_cv2(sub_l_img, desired_encoding='bgr8')
            img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)

            img_r = self.bridge.imgmsg_to_cv2(sub_r_img, desired_encoding='bgr8')
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
            merged = cv2.hconcat([img_l, img_r])

            cv2.imshow("Stereo Image Viewer", merged)

            self.fit_eval(img_l, img_r, self.calib, self.segmenter)

            self.hz_callback()
            cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}")

    def fit_eval(self, img1, img2, calib, segmenter):
        # Read in camera matrix
        cv_file = cv2.FileStorage(calib, cv2.FILE_STORAGE_READ)
        K1 = cv_file.getNode("K1").mat()
        D1 = cv_file.getNode("D1").mat()
        K2 = cv_file.getNode("K2").mat()
        D2 = cv_file.getNode("D2").mat()
        R = cv_file.getNode("R").mat()
        T = cv_file.getNode("T").mat()
        ImageSize = cv_file.getNode("ImageSize").mat()
        img_size = (int(ImageSize[0][1]), int(ImageSize[0][0]))
        new_size = (640, 480)

        # Rectify image and store necessary matrices
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=new_size)
        cam2img = P1[:,:-1]
        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, new_size, cv2.CV_16SC2)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, new_size, cv2.CV_16SC2)
        img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
        img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
        
        # Segment stereo images
        mask1 = segmenter.segmentation(img1)
        mask2 = segmenter.segmentation(img2)
        stack_mask1 = np.stack((mask1, mask1, mask1), axis=-1)
        img1 = np.where(stack_mask1>0, img1, 0)
        stack_mask2 = np.stack((mask2, mask2, mask2), axis=-1)
        img2 = np.where(stack_mask2>0, img2, 0)
        
        # Convert from btyes to float
        img1 = np.float32(img1)
        img2 = np.float32(img2)
        
        # Perform reconstruction
        img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents = keypt_selection(img1, img2, mask1, Q)
        img_3D, keypoints, grow_paths, order = keypt_ordering(img1, img_3D, cluster_map, keypoints, grow_paths, adjacents)
        spline, reliability = optim(mask1, mask2, mask1, mask2, keypoints, order, cam2img, P1, P2)
        return spline
    
    def hz_callback(self):
        now = time.perf_counter()
        if self.last is None:
            self.last = now
            return None

        dt = now - self.last
        self.last = now

        if dt <= 0:
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None

        self.times.append(dt)
        avg_dt = sum(self.times) / len(self.times)

        self.hz = 1.0 / avg_dt
        self.get_logger().info(f"Hz: {self.hz:.2f}")
        return self.hz

if __name__ == "__main__":
    USE_SAM = False

    # inp_folder = "/media/emmah/PortableSSD/Arclab_data/meat_thread_data_9_26/"
    inp_folder = "/media/arclab/PortableSSD/Arclab_data/4_1_26/"
    trial = "trial_01"
    # prefixes = ["left_recif_", "right_recif_"]
    prefixes = ["left_", "right_"]
    left_right_folder = ["/left_rgb/", "/right_rgb/"]
    start = 0
    ext = ".png" # ".jpg"
    calib = os.path.dirname(__file__) + "/../assets/camera_calibration_fei.yaml"
    if USE_SAM:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_type = "vit_h"
        segmenter = SAMSegmenter(device, model_type)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        segmenter = UNetSegmenter(device)
    left_topic = "stereo/left/rectified_downscaled_image"
    right_topic = "stereo/right/rectified_downscaled_image"

    rclpy.init()
    node = fitEvalNode(left_topic, right_topic, calib, segmenter)
    while rclpy.ok() and not node.image_received:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.get_logger().info("Image stream ready, continuing...")

    try:
        rclpy.spin(node)
    except:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


'''
    def fit_eval_ros(self, 
                     img1,
                     img2,
                     img1_t_mask,
                     img2_t_mask,
                     frame,
                     needle_mask=None,
                     ):
        # pr.enable()
        if needle_mask is None:
            needle_pos_file=None
        lr_folder = ["left_rgb/", "right_rgb/"]
        ext = ".png"

        # img1_t_mask = cv2.cvtColor(img1_t_mask, cv2.COLOR_BGR2RGB)
        # img1_n_mask = cv2.cvtColor(cv2.imread(img1_n_mask), cv2.COLOR_BGR2RGB)
        # img2_t_mask = cv2.cvtColor(img2_t_mask, cv2.COLOR_BGR2RGB)
        # img2_n_mask = cv2.cvtColor(cv2.imread(img2_n_mask), cv2.COLOR_BGR2RGB)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)





        warm_thread_path, warm_thread_specs_path = self.seek_warm_start(frame)

        # Segment stereo images
        # mask1_t = cv2.cvtColor(img1_t_mask, cv2.COLOR_RGB2GRAY)
        # mask2_t = cv2.cvtColor(img2_t_mask, cv2.COLOR_RGB2GRAY)
        mask1_n, mask2_n = None, None
        # mask1_n = cv2.cvtColor(img1_n_mask, cv2.COLOR_RGB2GRAY)
        # mask2_n = cv2.cvtColor(img2_n_mask, cv2.COLOR_RGB2GRAY)

        # mask1 = mask1_t+mask1_n
        # mask2 = mask2_t+mask2_n
        # mask1 = mask1_t
        # mask2 = mask2_t
        mask1 = img1_t_mask
        mask2 = img2_t_mask
        mask1_t = img1_t_mask
        mask2_t = img2_t_mask
        stack_mask1 = np.stack((mask1, mask1, mask1), axis=-1)
        img1 = np.where(stack_mask1>0, img1, 0)
        # img1 = stack_mask1
        stack_mask2 = np.stack((mask2, mask2, mask2), axis=-1)
        img2 = np.where(stack_mask2>0, img2, 0)
        # img2 = stack_mask2
        
        # Convert from btyes to float
        img1 = np.float32(img1)
        img2 = np.float32(img2)

        # print("fit eval cv2 version", cv2.__version__)
        # print(f"img1 shape {img1.shape}, mask1 shape {mask1.shape}")

        # merged = cv2.hconcat([img1, img2])
        # cv2.imshow("img1, img2", merged)
        # cv2.imshow("mask1 of fit_eval_ros", mask1)
        # cv2.waitKey(0)

        img_3D, __, cluster_map, keypoints, grow_paths, adjacents =  keypt_selection(img1, img2, mask1, self.Q)
        
        warm = False
        if (self.speedy or self.ros_enable) and warm_thread_path is not None:
            warm = True
        elif self.ros_enable and warm_thread_path is None:
            warm = False

        # pdb.set_trace()
        if warm and warm_thread_path is not None:
            # warm_keypts = None
            warm_keypoints, order, warm_keypts = warm_start_keypoints(mask1, keypoints, 
                                                                thread_file_path=warm_thread_path, 
                                                                thread_specs_path=warm_thread_specs_path, 
                                                                P1=self.P1, 
                                                                speedy=self.speedy, 
                                                                ros_enable=self.ros_enable)
            if warm_keypts == None: # if warm start fails, resort to hand ordering
                keypoints, order = hand_ordering(img1, img_3D, keypoints, needle_pos_file, self.P1) # clusters is not used in keypt_ordering, hence removed
                spline, spline_specs = optim(mask1_t, mask2_t, mask1_n, mask2_n, 
                                            keypoints, 
                                            order, 
                                            self.cam2img, 
                                            self.P1, self.P2, 
                                            needle_pos_file)
                return spline, spline_specs
            spline, spline_specs = optim_warm_start(mask1_t, mask2_t, 
                                                    mask1_n, mask2_n, 
                                                    warm_keypoints, 
                                                    order, 
                                                    self.cam2img, 
                                                    self.P1, 
                                                    thread_file_path=warm_thread_path, 
                                                    warm_start_keypts=warm_keypts, 
                                                    speedy=self.speedy, 
                                                    ros_enable=self.ros_enable, 
                                                    needle_pos_file=needle_pos_file
                                                    )
        else:
            if self.hand_order:
                keypoints, order = hand_ordering(img1, img_3D, keypoints, needle_pos_file, self.P1) # clusters is not used in keypt_ordering, hence removed
            else: # original method auto ordering
                __, keypoints, __, order = keypt_ordering(img1, img_3D, cluster_map, keypoints, grow_paths, adjacents)

            spline, spline_specs = optim(mask1_t, mask2_t, mask1_n, mask2_n, 
                                        keypoints, 
                                        order, 
                                        self.cam2img, 
                                        self.P1, self.P2, 
                                        needle_pos_file)
        # pr.disable()

        return spline, spline_specs

'''