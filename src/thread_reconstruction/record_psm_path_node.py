#! /home/autosurg/anaconda3/envs/thread_ros/bin/python

import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from __thread_reconstruction.srv import RecordPSMPath, RecordPSMPathResponse

import numpy as np

from read_dvrk_msg.ros_dvrk import ROSdVRK
from psm_control import utils

PSM = 1

class RecordPSMPathNode:
    def __init__(self, ros_dvrk):
        self.ros_dvrk = ros_dvrk
        self.path = []
        self.recording = False
        self.spacing = 0.005 # Half cm by default
        self.path_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)

        # store and reuse visualization message for convenience
        self.marker = Marker()
        self.marker.header.frame_id = "dvrk_cam"
        self.marker.header.stamp = rospy.Time()
        self.marker.ns = "psm_path"
        self.marker.id = 0
        self.marker.type = Marker.LINE_STRIP
        self.marker.action = Marker.MODIFY
        self.marker.pose.position.x = 0
        self.marker.pose.position.y = 0
        self.marker.pose.position.z = 0
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0
        self.marker.scale.x = 0.001 # Default 1 mm thickness
        self.marker.color.a = 1.0
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0
        self.marker.points = []
        self.marker.lifetime = rospy.Duration(0)
        self.marker.frame_locked = True

    def toggle_recording(self, request):
        if request.record == True and self.recording == False:
            self.path = []
            self.marker.points = []
            self.spacing = request.spacing
            self.recording = True
            return RecordPSMPathResponse("Recording started")
        elif request.record == False and self.recording == True:
            self.recording = False
            return RecordPSMPathResponse("Recording stopped")
        else:
            return RecordPSMPathResponse("Invalid inputs. Current recording state is: " + str(self.recording))
    
    def build_path(self, event):
        if self.recording:
            # Get current PSM position
            ros_msg = self.ros_dvrk.getSyncMsg()
            current_pose_cam_base = ros_msg['pose_cam_base{}'.format(PSM)]
            current_pose_base_ree = ros_msg['pose_base_ee{}'.format(PSM)]
            REE_TO_TIP = 0.0102 # Taken from dvrk manual
            current_pose_ree_tip = [1, 0, 0, 0, 0, REE_TO_TIP/2, 0]
            current_H_cam_base = utils.posquat2H(
                current_pose_cam_base[-3:], 
                current_pose_cam_base[:4], 
            )
            current_H_base_ree = utils.posquat2H(
                current_pose_base_ree[-3:], 
                current_pose_base_ree[:4], 
            )
            current_H_ree_tip= utils.posquat2H(
                current_pose_ree_tip[-3:], 
                current_pose_ree_tip[:4], 
            )
            current_H_cam_tip = current_H_cam_base @ current_H_base_ree @ current_H_ree_tip
            current_pos_cam_tip, current_quat_cam_tip = utils.matrix2PosQuat(current_H_cam_tip)

            # Add points if spacing distance is met
            if len(self.path) == 0 or np.linalg.norm(self.path[-1] - current_pos_cam_tip) > self.spacing:
                self.path.append(current_pos_cam_tip)
                point = Point()
                point.x, point.y, point.z = current_pos_cam_tip[0], current_pos_cam_tip[1], current_pos_cam_tip[2]
                self.marker.points.append(point)
                self.path_pub.publish(self.marker)

if __name__ == '__main__': 
    ros_dvrk = ROSdVRK( # Calls "init_node" internally
        control=True, 
        joints=True, 
    )
    node = RecordPSMPathNode(ros_dvrk)
    service = rospy.Service('record_psm_path_node/toggle_recording', RecordPSMPath, node.toggle_recording)
    timer = rospy.Timer(rospy.Duration.from_sec(0.2), node.build_path)
    rospy.spin()