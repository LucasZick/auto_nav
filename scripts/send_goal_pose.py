#!/usr/bin/env python

import rospy
import actionlib
import os
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose
from std_msgs.msg import Empty

debug_mode = os.getenv('AUTO_NAV_DEBUG', 'False').lower() == 'true' 

def active_cb():
    if debug_mode:
        rospy.loginfo("Goal pose being processed")

def feedback_cb(feedback):
    if debug_mode:
        rospy.loginfo("Current location: " + str(feedback))

def done_cb(status, result):
    if status == 3:
        rospy.loginfo("Goal reached")
    if status == 2 or status == 8:
        rospy.loginfo("Goal cancelled")
    if status == 4:
        rospy.loginfo("Goal aborted")

rospy.init_node('goal_pose')

navclient = actionlib.SimpleActionClient('move_base', MoveBaseAction)
navclient.wait_for_server()

def position_callback(pose_msg):
    x = pose_msg.position.x
    y = pose_msg.position.y
    z = pose_msg.position.z
    orientation_x = pose_msg.orientation.x
    orientation_y = pose_msg.orientation.y
    orientation_z = pose_msg.orientation.z
    orientation_w = pose_msg.orientation.w

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.position.z = z
    goal.target_pose.pose.orientation.x = orientation_x
    goal.target_pose.pose.orientation.y = orientation_y
    goal.target_pose.pose.orientation.z = orientation_z
    goal.target_pose.pose.orientation.w = orientation_w
    
    navclient.send_goal(goal, done_cb, active_cb, feedback_cb)

position_subscriber = rospy.Subscriber('hand_position', Pose, position_callback)

def restart_robot_callback():
    navclient.cancel_all_goals()
    rospy.loginfo("Restarting the robot")

restart_robot_subscriber = rospy.Subscriber('restart_robot', Empty, restart_robot_callback)
rospy.spin()
