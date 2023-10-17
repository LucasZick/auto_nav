#!/usr/bin/env python

import math
import os
import numpy as np
import rospy
import cv2
import mediapipe as mp
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import Odometry

debug_mode = os.getenv('AUTO_NAV_DEBUG', 'False').lower() == 'true' 

robot_marker = Marker()
robot_marker.header.frame_id = "map"
robot_marker.type = Marker.MESH_RESOURCE
robot_marker.mesh_resource = "package://auto_nav/models/robot.stl"
robot_marker.action = Marker.ADD
robot_marker.id = 1
robot_marker.scale.x = 0.01
robot_marker.scale.y = 0.01
robot_marker.scale.z = 0.01
robot_marker.color.r = 1
robot_marker.color.g = 0
robot_marker.color.b = 0
robot_marker.color.a = 0.4

hand_marker = Marker()
hand_marker.header.frame_id = "map"
hand_marker.type = Marker.MESH_RESOURCE
hand_marker.mesh_resource = "package://auto_nav/models/open_hand.stl"
hand_marker.action = Marker.ADD
hand_marker.id = 2
hand_marker.scale.x = 0.005
hand_marker.scale.y = 0.005
hand_marker.scale.z = 0.005
hand_marker.color.r = 1
hand_marker.color.g = 0
hand_marker.color.b = 1
hand_marker.color.a = 1

def angle_to_quaternion(angle):
    quaternion = Quaternion()
    quaternion.z = math.sin(angle / 2)
    quaternion.w = math.cos(angle / 2)

    return quaternion

def map_coordinates_to_grid(hand_x, hand_y, frame, grid_size=256):
    hand_x_normalized = hand_x / frame.shape[1]
    hand_y_normalized = hand_y / frame.shape[0]

    grid_x = int(hand_x_normalized * grid_size)
    grid_y = int(hand_y_normalized * grid_size)

    return grid_x, grid_y

def grid_to_map(grid_x, grid_y):
    scale_factor = 2.0
    x = (grid_x / 256) * 5 - 2.5
    y = (grid_y / 256) * 5 - 2.5
    x *= scale_factor
    y *= scale_factor

    return -y,-x

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    return distance

def calculate_hand_angle(landmarks, point1_idx, point2_idx, is_left_hand):
    x1, y1 = landmarks.landmark[point1_idx].x, landmarks.landmark[point1_idx].y
    x2, y2 = landmarks.landmark[point2_idx].x, landmarks.landmark[point2_idx].y

    if is_left_hand:
        x1, x2 = -x1, -x2

    vec = np.array([x2 - x1, y2 - y1])
    angle = np.arctan2(vec[1], vec[0])
    angle_deg = np.degrees(angle)

    while angle_deg > 90:
        angle_deg -= 180
    while angle_deg < -90:
        angle_deg += 180

    return -angle_deg if is_left_hand else angle_deg 

def publish_marker():
    rospy.init_node('marker_publisher')
    hand_marker_pub = rospy.Publisher('hand_marker1', Marker, queue_size=10)
    robot_marker_pub = rospy.Publisher('robot_marker', Marker, queue_size=10)
    position_publisher = rospy.Publisher('hand_position', Pose, queue_size=10)

    rate = rospy.Rate(30)

    orientation_yaw = 0
    x, y, z = 0, 0, 0

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Hand recognition', cv2.WINDOW_NORMAL)

    close_distance_threshold = 0.10
    hand_closed = False
    prev_hand_closed = False
    marker_following_hand = False
    marker_released = False
    alpha = 0.5
    robot_marker_position = rospy.wait_for_message('/odom', Odometry)
    robot_marker_x = robot_marker_position.pose.pose.position.x
    robot_marker_y = robot_marker_position.pose.pose.position.y
    hand_marker_x = 0
    hand_marker_y = 0
    robot_marker.pose = robot_marker_position.pose.pose
    robot_marker_pub.publish(robot_marker)

    while not rospy.is_shutdown() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                hand_caption_x = landmarks.landmark[0].x * frame.shape[1]
                hand_caption_y = landmarks.landmark[0].y * frame.shape[0]
                hand_grid_x, hand_grid_y = map_coordinates_to_grid(hand_caption_x, hand_caption_y, frame, grid_size=256)
                old_hand_marker_x, old_hand_marker_y = hand_marker_x, hand_marker_y
                hand_marker_x, hand_marker_y = grid_to_map(hand_grid_x,hand_grid_y)
                distance_to_marker = calculate_distance((robot_marker_x,robot_marker_y),(hand_marker_x, hand_marker_y))

                distances = []
                for finger in [4, 8, 12, 16, 20]:
                    x, y = landmarks.landmark[finger].x, landmarks.landmark[finger].y
                    distances.append((x, y))
                avg_distance = sum([(x2 - x1) ** 2 + (y2 - y1) ** 2 for x1, y1 in distances for x2, y2 in distances]) ** 0.5 / 5

                prev_hand_closed = hand_closed
                hand_closed = avg_distance < close_distance_threshold

                if marker_following_hand:
                    x = (1 - alpha) * robot_marker_x + alpha * hand_marker_x
                    y = (1 - alpha) * robot_marker_y + alpha * hand_marker_y
                    robot_marker_x = x
                    robot_marker_y = y
                    robot_marker.pose.position = Point(x, y, z)
                    robot_marker.pose.orientation = angle_to_quaternion(orientation_yaw)
                    robot_marker_pub.publish(robot_marker)

                    if prev_hand_closed and not hand_closed: 
                        marker_released = True

                if hand_closed and not prev_hand_closed and distance_to_marker < (2 if marker_following_hand else 0.5):
                    marker_following_hand = True
                elif not hand_closed:
                    marker_following_hand = False

                if marker_released:
                    position_msg = Pose()
                    position_msg.position = Point(x, y, z)
                    position_msg.orientation = angle_to_quaternion(orientation_yaw)
                    position_publisher.publish(position_msg)
                    marker_released=False
                    orientation_yaw = 0

                x = (1 - alpha) * hand_marker_x + alpha * old_hand_marker_x
                y = (1 - alpha) * hand_marker_y + alpha * old_hand_marker_y
                
                hand_marker.mesh_resource = f"package://auto_nav/models/{'closed' if hand_closed else 'open'}_hand.stl"    
                hand_marker.pose.position = Point(x, y, 10)
                
                is_left_hand = (landmarks.landmark[0].x < 0.5)
                angle = calculate_hand_angle(landmarks, 9, 13, is_left_hand)
                if 40 < angle < 100:
                    orientation_yaw -= 0.1
                elif -40 > angle > -100:
                    orientation_yaw += 0.1

                hand_marker.pose.orientation = angle_to_quaternion(orientation_yaw)

                hand_marker_pub.publish(hand_marker)
                rate.sleep()

        if cv2.waitKey(1) & 0xFF == 27:
            break
        cv2.imshow('Hand recognition', frame)
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    try:
        publish_marker()
    except rospy.ROSInterruptException:
        pass
