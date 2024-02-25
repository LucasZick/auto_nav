#!/usr/bin/env python

import rospy
import actionlib
import os
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Empty
from auto_nav.msg import RobotInfo

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

def send_goal_callback(goal_msg):
    navclient = actionlib.SimpleActionClient(f'robot{goal_msg.robot_id}/move_base', MoveBaseAction)
    navclient.wait_for_server()
    
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = goal_msg.pose.position.x
    goal.target_pose.pose.position.y = goal_msg.pose.position.y
    goal.target_pose.pose.position.z = goal_msg.pose.position.z
    goal.target_pose.pose.orientation.x = goal_msg.pose.orientation.x
    goal.target_pose.pose.orientation.y = goal_msg.pose.orientation.y
    goal.target_pose.pose.orientation.z = goal_msg.pose.orientation.z
    goal.target_pose.pose.orientation.w = goal_msg.pose.orientation.w
    
    navclient.send_goal(goal, done_cb, active_cb, feedback_cb)

def restart_robot_callback(robot_id):
    navclient = actionlib.SimpleActionClient(f'robot{robot_id}/move_base', MoveBaseAction)
    navclient.wait_for_server()
    navclient.cancel_all_goals()
    rospy.loginfo("Restarting the robot")

position_subscriber = rospy.Subscriber('send_goal', RobotInfo, send_goal_callback)
restart_robot_subscriber = rospy.Subscriber('restart_robot', Empty, restart_robot_callback)
rospy.spin()
