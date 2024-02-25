#!/usr/bin/env python

import math
import os
import time
import numpy as np
import rospy
import cv2
import mediapipe as mp
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Quaternion
from nav_msgs.msg import Odometry
from auto_nav.msg import RobotInfo

num_robots = int(os.getenv('NUM_ROBOTS', '3'))
debug_mode = os.getenv('AUTO_NAV_DEBUG', 'False').lower() == 'true'

class Robot:
    def __init__(self, id):
        self.id = id
        self.marker = Marker()
        self.marker.header.frame_id = "map"
        self.marker.type = Marker.MESH_RESOURCE
        self.marker.mesh_resource = "package://auto_nav/models/robot.stl"
        self.marker.action = Marker.ADD
        self.marker.id = id
        self.marker.scale.x = 0.01
        self.marker.scale.y = 0.01
        self.marker.scale.z = 0.01
        self.marker.color.r = 1 if id == 1 else 0
        self.marker.color.g = 1 if id == 2 else 0
        self.marker.color.b = 1 if id == 3 else 0
        self.marker.color.a = 0.4
        self.marker_publisher = rospy.Publisher(f"robot_marker{id}", Marker, queue_size=10)

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
    scale_factor = 3
    x = (grid_x / 256) * 5 - 2.5
    y = (grid_y / 256) * 5 - 2.5
    x *= scale_factor
    y *= scale_factor

    return -y,-x

def closest_distance_info(robots, target_point):
    closest_index, _ = min(enumerate(robots, start=1), key=lambda x: math.dist((x[1].marker.pose.position.x, x[1].marker.pose.position.y), target_point))
    closest_distance = math.dist((robots[closest_index - 1].marker.pose.position.x, robots[closest_index - 1].marker.pose.position.y), target_point)
    
    return closest_index, closest_distance

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

robots = []
for robot in range(num_robots):
    robots.append(Robot(id=robot+1))

hand_marker = Marker()
hand_marker.header.frame_id = "map"
hand_marker.type = Marker.MESH_RESOURCE
hand_marker.mesh_resource = "package://auto_nav/models/open_hand.stl"
hand_marker.action = Marker.ADD
hand_marker.id = 2
hand_marker.scale.x = 0.005
hand_marker.scale.y = 0.005
hand_marker.scale.z = 0.005
hand_marker.color.r = 0.9
hand_marker.color.g = 0.6
hand_marker.color.b = 0.4
hand_marker.color.a = 1

def publish_marker():
    rospy.init_node('marker_publisher')
    hand_marker_pub = rospy.Publisher('hand_marker', Marker, queue_size=10)
    goal_publisher = rospy.Publisher('send_goal', RobotInfo, queue_size=10)
    rate = rospy.Rate(30)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Hand recognition', cv2.WINDOW_NORMAL)

    close_distance_threshold = 0.10
    hand_closed = False
    prev_hand_closed = False
    marker_following_hand = None
    marker_released = False
    smooth = 0.5

    orientation_yaw = 0
    hand_marker_x, hand_marker_y = 0, 0
    
    for robot in robots:
        robot_marker_position = rospy.wait_for_message(f'robot{robot.id}/odom', Odometry)
        robot.marker.pose = robot_marker_position.pose.pose
        robot.marker_publisher.publish(robot.marker)

    start_time = time.time()
    start_operating_time = time.time()
    operation_time = 0
    recognizing_hands = False

    while not rospy.is_shutdown() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            recognizing_hands = True
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                hand_caption_x = landmarks.landmark[0].x * frame.shape[1]
                hand_caption_y = landmarks.landmark[0].y * frame.shape[0]
                hand_grid_x, hand_grid_y = map_coordinates_to_grid(hand_caption_x, hand_caption_y, frame, grid_size=256)
                old_hand_marker_x, old_hand_marker_y = hand_marker_x, hand_marker_y
                hand_marker_x, hand_marker_y = grid_to_map(hand_grid_x,hand_grid_y)

                if marker_following_hand:
                    closest_robot = marker_following_hand
                else:
                    closest_robot_id, distance_to_marker = closest_distance_info(robots,(hand_marker_x, hand_marker_y))
                    closest_robot = robots[closest_robot_id-1]

                distances = []
                for finger in [4, 8, 12, 16, 20]:
                    x, y = landmarks.landmark[finger].x, landmarks.landmark[finger].y
                    distances.append((x, y))
                avg_distance = sum([(x2 - x1) ** 2 + (y2 - y1) ** 2 for x1, y1 in distances for x2, y2 in distances]) ** 0.5 / 5

                prev_hand_closed = hand_closed
                hand_closed = avg_distance < close_distance_threshold

                if marker_following_hand and marker_following_hand.id == closest_robot.id:
                    closest_robot.marker.pose.position.x = (1 - smooth) * closest_robot.marker.pose.position.x + smooth * hand_marker_x
                    closest_robot.marker.pose.position.y = (1 - smooth) * closest_robot.marker.pose.position.y + smooth * hand_marker_y
                    closest_robot.marker.pose.orientation = angle_to_quaternion(orientation_yaw)
                    closest_robot.marker_publisher.publish(closest_robot.marker)

                    if prev_hand_closed and not hand_closed: 
                        marker_released = True

                if hand_closed and not prev_hand_closed and distance_to_marker < (2 if marker_following_hand else 0.5):
                    marker_following_hand = closest_robot
                elif not hand_closed:
                    marker_following_hand = None

                if marker_released:
                    goal_info = RobotInfo()
                    goal_info.pose.position = Point(closest_robot.marker.pose.position.x, closest_robot.marker.pose.position.y, 0)
                    goal_info.pose.orientation = angle_to_quaternion(orientation_yaw)
                    goal_info.robot_id = closest_robot.id
                    goal_publisher.publish(goal_info)
                    marker_released=False
                    orientation_yaw = 0

                x = (1 - smooth) * hand_marker_x + smooth * old_hand_marker_x
                y = (1 - smooth) * hand_marker_y + smooth * old_hand_marker_y
                
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
        else:
            recognizing_hands = False
        if recognizing_hands:
            operation_time += (time.time() - start_operating_time)  # Incrementa o tempo
            start_operating_time = time.time()
        else:
            start_operating_time = time.time()
        if cv2.waitKey(1) & 0xFF == 27:
            print("\033[92mTotal execution time:", round(time.time()-start_time,2), "seconds\033[0m")
            print("\033[92mTotal operation time:", round(operation_time,2), "seconds\033[0m")
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
