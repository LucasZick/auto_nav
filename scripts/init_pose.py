#!/usr/bin/env python3

import os
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry

num_robots = int(os.getenv('NUM_ROBOTS', '3'))
debug_mode = os.getenv('AUTO_NAV_DEBUG', 'False').lower() == 'true'

# Node initialization
rospy.init_node('init_pose')

for robot_id in range(1, num_robots+1):
    pub = rospy.Publisher(f'robot{robot_id}/initialpose', PoseWithCovarianceStamped, queue_size = 1)
    init_msg = PoseWithCovarianceStamped()
    init_msg.header.frame_id = "map"

    # Get initial pose from Gazebo
    odom_msg = rospy.wait_for_message(f'robot{robot_id}/odom', Odometry)
    init_msg.pose.pose.position.x = odom_msg.pose.pose.position.x
    init_msg.pose.pose.position.y = odom_msg.pose.pose.position.y
    init_msg.pose.pose.orientation.x = odom_msg.pose.pose.orientation.x
    init_msg.pose.pose.orientation.y = odom_msg.pose.pose.orientation.y
    init_msg.pose.pose.orientation.z = odom_msg.pose.pose.orientation.z
    init_msg.pose.pose.orientation.w = odom_msg.pose.pose.orientation.w

    # Publish message
    if debug_mode:
        rospy.loginfo(f"setting initial pose for robot {robot_id}")
    rospy.sleep(0.2)
    pub.publish(init_msg)
    if debug_mode:
        rospy.loginfo(f"initial pose is set for robot {robot_id}")

    # Delay before clearing costmaps
    rospy.sleep(0.2)

    # Clear costmaps by calling the service
    if debug_mode:
        rospy.loginfo(f"clearing costmaps for robot {robot_id}")
    rospy.wait_for_service(f'/move_base{robot_id}/clear_costmaps')
    clear_costmaps = rospy.ServiceProxy(f'/move_base{robot_id}/clear_costmaps', Empty)
    clear_costmaps()
    if debug_mode:
        rospy.loginfo(f"costmaps cleared for robot {robot_id}")
    rospy.sleep(0.2)

if debug_mode:
        rospy.loginfo("finished positional setup")