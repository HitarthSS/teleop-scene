#!/usr/bin/env python3
from pathlib import Path
import sys
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))
sys.path.append("/home/emmah/miniconda3/envs/sam/lib/python3.10/site-packages")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
import cv2
import argparse
import pdb


from message_filters import Subscriber, ApproximateTimeSynchronizer
import time
from collections import deque
import time
from fit_eval_exp import FitEvalClass
import re
import traceback


class ImageProcessingNode(Node):
    def __init__(self, trial_path, trial_name, hz_window=10):
        super().__init__('image_processing_node')
        self.trial_path = trial_path
        self.trial_name = trial_name
        self.hz = 0 # processing speed
        self.times = deque(maxlen=hz_window)
        self.last = None
        self.bridge = CvBridge()

        self.sub_l = Subscriber(self,
                                Image,
                                '/image_left',
        )        
        self.sub_r = Subscriber(self,
                                Image,
                                '/image_right',
        )
        self.sub_mask_l = Subscriber(self,
                                Image,
                                '/mask_left',
        )        
        self.sub_mask_r = Subscriber(self,
                                Image,
                                '/mask_right',
        )
        # initialize fit eval processes
        self.fit_eval = FitEvalClass(trial_path=trial_path,
                                trial_name=trial_name,
                                speedy=False,
                                ros_enable=True
                                )
        
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_l, self.sub_r, self.sub_mask_l, self.sub_mask_r],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.callback)

        self.get_logger().info("Image subscriber node started.")


    def callback(self, img_l, img_r, img_mask_l, img_mask_r):
        try:
            frame_id_l = img_l.header.frame_id
            frame_id_r = img_r.header.frame_id
            frame = int(re.search(r'trial_(\d+)(?:_video)?_(\d+)', frame_id_l).group(2)) #extract frame number

            # convert from ROS to CV
            cv_img_l = self.bridge.imgmsg_to_cv2(img_l, desired_encoding='bgr8')
            cv_img_r = self.bridge.imgmsg_to_cv2(img_r, desired_encoding='bgr8')
            cv_mask_l = self.bridge.imgmsg_to_cv2(img_mask_l, desired_encoding='mono8')
            cv_mask_r = self.bridge.imgmsg_to_cv2(img_mask_r, desired_encoding='mono8')

            self.get_logger().info(
                f"Synchronized images received: "
                f"{frame_id_l} {img_l.header.stamp.sec}.{img_l.header.stamp.nanosec}, "
                f"{frame_id_r} {img_r.header.stamp.sec}.{img_r.header.stamp.nanosec}"
            )
            # Resize images to the same height if necessary
            # if cv_mask_l.shape[:2] != cv_mask_r.shape[:2]:
            #     height = min(cv_mask_l.shape[0], cv_mask_r.shape[0])
            #     width_l = int(cv_mask_l.shape[1] * height / cv_mask_l.shape[0])
            #     width_r = int(cv_mask_r.shape[1] * height / cv_mask_r.shape[0])
            #     cv_mask_l = cv2.resize(cv_mask_l, (width_l, height))
            #     cv_mask_r = cv2.resize(cv_mask_r, (width_r, height))

            # Merge images side by side
            # merged = cv2.hconcat([cv_mask_l, cv_mask_r])
            merged = cv2.hconcat([cv_img_l, cv_img_r])
            # add text for frame
            cv2.putText(
                merged,
                str(frame_id_l+' | '+frame_id_r+f" Hz: {self.hz:.2f}"),
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Display merged image
            cv2.imshow("Stereo", merged)

            # # process spline
            try:
                spline, spline_specs = self.fit_eval.fit_eval_ros(img1=cv_img_l, img2=cv_img_r, img1_t_mask=cv_mask_l, img2_t_mask=cv_mask_r, frame=frame)
                # self.fit_eval.save_spline(spline, spline_specs, frame)
            except Exception as e:
                print(f"Error with frame {frame}: {e}")
                print(traceback.format_exc())
                pass

            # log image rate
            self.hz_callback()

            cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}")

    def hz_callback(self):
        now = time.perf_counter()
        if self.last is None:
            self.last = now
            return None

        dt = now - self.last
        self.last = now

        if dt <= 0:
            return None

        self.times.append(dt)
        avg_dt = sum(self.times) / len(self.times)

        self.hz = 1.0 / avg_dt
        self.get_logger().info(f"Hz: {self.hz:.2f}")
        return self.hz

def main():

    BASE_DIR="/media/emmah/PortableSSD/Arclab_data"
    PARENT_FOLDER="thread_meat_3_21"
    TRIAL_NAME="trial_33_video"
    IMAGE_DIR=f"{BASE_DIR}/{PARENT_FOLDER}/{TRIAL_NAME}/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_path', default=IMAGE_DIR, help="full path to the directory of the trial used")
    parser.add_argument('--trial_name', default=TRIAL_NAME, help='trial name is usually trial_xx or trial_xx_video')
    args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = ImageProcessingNode(trial_path=args.trial_path, trial_name=args.trial_name)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
