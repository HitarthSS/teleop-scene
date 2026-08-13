#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
import cv2

import argparse
from pathlib import Path
import pdb

class ImagePublishNode(Node):
    def __init__(self, trial_path, trial_name, start_frame=0, end_frame=1, hz=1):
        super().__init__('image_publish_node')

        self.pub_l = self.create_publisher(Image, '/image_left', 10)
        self.pub_r = self.create_publisher(Image, '/image_right', 10)
        self.pub_mask_l = self.create_publisher(Image, '/mask_left', 10)
        self.pub_mask_r = self.create_publisher(Image, '/mask_right', 10)

        self.bridge = CvBridge()

        self.trial_path = trial_path
        self.trial_name = trial_name
        self.frame_idx = start_frame
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.hz = hz
        self.ext = ".png"
        self.path_option = 1

        lr_folder = ["left_rgb/", "right_rgb/"]
        self.img_l_path = trial_path + "/" + lr_folder[0]
        self.img_r_path = trial_path + "/" + lr_folder[1]
        
        t_mask_1 = Path(trial_path) / lr_folder[0] / f"{trial_name}_{start_frame:03d}_mask{self.ext}"
        t_mask_2 = Path(trial_path) / lr_folder[0] / f"binary_masks/{trial_name}_{start_frame:03d}{self.ext}"

        if t_mask_1.exists():
            self.img_l_mask_path = trial_path + lr_folder[0]
            self.img_r_mask_path = trial_path + lr_folder[1]
            self.path_option = 1
        elif t_mask_2.exists():
            self.img_l_mask_path = trial_path + lr_folder[0] + f"binary_masks/"
            self.img_r_mask_path = trial_path + lr_folder[1] + f"binary_masks/"
            self.path_option = 2
        else:
            raise FileNotFoundError(f"File not found:\n{t_mask_1}\n{t_mask_2}")

        # publish at 1 Hz
        self.timer = self.create_timer(1/self.hz, self.timer_callback)

    def timer_callback(self):
        try:
            if self.frame_idx > self.end_frame:
                self.frame_idx = self.start_frame

            now = self.get_clock().now().to_msg()
            if self.path_option == 1:
                img_l_mask = cv2.imread(str(self.img_l_mask_path + f"{self.trial_name}_{self.frame_idx:03d}_mask{self.ext}"), cv2.IMREAD_GRAYSCALE)
                img_r_mask = cv2.imread(str(self.img_r_mask_path + f"{self.trial_name}_{self.frame_idx:03d}_mask{self.ext}"), cv2.IMREAD_GRAYSCALE)
            else:
                img_l_mask = cv2.imread(str(self.img_l_mask_path + f"{self.trial_name}_{self.frame_idx:03d}{self.ext}"), cv2.IMREAD_GRAYSCALE)
                img_r_mask = cv2.imread(str(self.img_r_mask_path + f"{self.trial_name}_{self.frame_idx:03d}{self.ext}"), cv2.IMREAD_GRAYSCALE)

            img_l = cv2.imread(str(self.img_l_path + f"{self.trial_name}_{self.frame_idx:03d}{self.ext}"))
            img_r = cv2.imread(str(self.img_r_path + f"{self.trial_name}_{self.frame_idx:03d}{self.ext}"))

            ros_img_l = self.bridge.cv2_to_imgmsg(img_l, encoding='bgr8')
            ros_img_r = self.bridge.cv2_to_imgmsg(img_r, encoding='bgr8')
            ros_mask_l = self.bridge.cv2_to_imgmsg(img_l_mask, encoding='mono8')
            ros_mask_r = self.bridge.cv2_to_imgmsg(img_r_mask, encoding='mono8')
            ros_img_l.header = Header()
            ros_img_r.header = Header()
            ros_mask_l.header = Header()
            ros_mask_r.header = Header()
            ros_img_l.header.stamp = now
            ros_img_r.header.stamp = now
            ros_mask_l.header.stamp = now
            ros_mask_r.header.stamp = now

            ros_img_l.header.frame_id = f"{self.trial_name}_{self.frame_idx:03d}{self.ext}"
            ros_img_r.header.frame_id = f"{self.trial_name}_{self.frame_idx:03d}{self.ext}"
            ros_mask_l.header.frame_id = f"{self.trial_name}_{self.frame_idx:03d}_mask{self.ext}"
            ros_mask_r.header.frame_id = f"{self.trial_name}_{self.frame_idx:03d}_mask{self.ext}"
            
            # print("publish cv2 version", cv2.__version__)
            # cv_merged = cv2.hconcat([img_l_mask, img_r_mask])
            # cv2.imshow("cv images before publish", cv_merged)
            # cv2.waitKey(0)
            self.pub_l.publish(ros_img_l)
            self.pub_r.publish(ros_img_r)
            self.pub_mask_l.publish(ros_mask_l)
            self.pub_mask_r.publish(ros_mask_r)

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}")
        
        self.frame_idx += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_path', required=True)
    parser.add_argument('--trial_name', type=str, required=True)
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=1)
    parser.add_argument('--hz', type=int, default=10)

    args, unknown = parser.parse_known_args()


    rclpy.init(args=unknown)
    node = ImagePublishNode(args.trial_path, args.trial_name, args.start_frame, args.end_frame, args.hz)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
