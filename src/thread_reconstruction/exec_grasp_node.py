#! /home/autosurg/anaconda3/envs/thread_ros/bin/python

import rospy
from geometry_msgs.msg import PoseStamped
from __thread_reconstruction.srv import Grasp, GraspResponse

import numpy as np
import time

from read_dvrk_msg.ros_dvrk import ROSdVRK
from psm_control.psm_control import PsmControl

from tf.transformations import quaternion_matrix
import os
import pdb

# Drop into pdb at every error / degenerate return in this module, so a
# silently-skipped frame stops instead of being swallowed by the caller's
# fallback path.  Export THREAD_RECON_BREAK=0 to disable every breakpoint
# in the package at once (unattended runs, live ROS sessions).
DEBUG_BREAK_ON_ERROR = os.environ.get("THREAD_RECON_BREAK", "1") != "0"


CAPTURE = 0
GUIDE = 1
GRASP = 2
PSM = 2
OTHER_PSM = 2 if PSM==1 else 1

class ExecGraspNode:
    def __init__(self, psm_control):
        self.psm_control = psm_control
        self.grasp_sub = rospy.Subscriber("/thread_reconstr/grasp_pose", PoseStamped, self.execute_grasp, queue_size=1)
    
    def execute_grasp(self, request):
        pose = request.graspPoint
        
        # Set up approach on capture
        if request.primitive == CAPTURE:
            R = quaternion_matrix([
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w
            ])
            approach_len = 0.015
            grasp_point = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
            approach_point = grasp_point - R[:3, 1] * approach_len
            approach_pose_cam_ree = np.array([
                pose.pose.orientation.w,
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                approach_point[0],
                approach_point[1],
                approach_point[2]
            ])

            self.psm_control.openGripper(PSM)
            success = self.psm_control.controlPoseReeInCam(PSM, approach_pose_cam_ree, timeout=10.0)
            if not success:
                if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
                return None

        # Move to goal
        goal_pose_cam_ree = np.array([
            pose.pose.orientation.w,
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z
        ])
        success = self.psm_control.controlPoseReeInCam(PSM, goal_pose_cam_ree, timeout=10.0)
        if not success:
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None

        # Set final gripper state
        if request.primitive == CAPTURE:
            self.gripper_capture()
        elif request.primitive == GRASP:
            self.psm_control.closeGripper(PSM)
            time.sleep(0.5)
            self.psm_control.openGripper(PSM)
        
        return GraspResponse()


    def gripper_capture(self):
        if PSM == 1:
            self.psm_control._setGripper(
                    self.psm_control.set_gripper1_pub, 
                    end_pos=20, 
                )
        elif PSM == 2:
            self.psm_control._setGripper(
                    self.psm_control.set_gripper2_pub, 
                    end_pos=20, 
                )
        time.sleep(1.)

if __name__ == '__main__': 
    ros_dvrk = ROSdVRK( # Calls "init_node" internally
        control=True, 
        joints=True, 
    )
    psm_control = PsmControl(ros_dvrk)
    time.sleep(2.0)
    psm_control.closeGripper(OTHER_PSM, tight=True)
    node = ExecGraspNode(psm_control)
    service = rospy.Service('exec_grasp_node/grasp', Grasp, node.execute_grasp)
    rospy.spin()
