# TurtleBot3 Multi-Robot Control

This project implements the control of multiple TurtleBot3 robots with hand recognition using MediaPipe and autonomous navigation through the ROS Navigation Stack. The entire system is coordinated by ROS, allowing intuitive interaction and autonomous movement of the robots.

## Table of Contents

1. About the Project
2. Prerequisites
3. Installation
4. How to Use
5. Repository Structure
6. Contributing
7. License
8. Contact

## About the Project

The TurtleBot3 Multi-Robot Control is an innovative system that combines hand recognition through the MediaPipe library with the ROS Navigation Stack to provide an intuitive control experience for mobile robots. The project aims to facilitate human-robot interaction and enable autonomous navigation in dynamic environments.

## Prerequisites

Make sure you have the following tools and libraries installed:

- ROS Noetic
- MediaPipe
- ROS dependencies for TurtleBot3

## Installation

Follow the steps below to install and set up the project locally:

1. Clone the repository:
   git clone https://github.com/LucasZick/turtlebot3-multirobot-control.git

2. Navigate to the project folder:
   cd turtlebot3-multirobot-control

3. Install the necessary dependencies (adjust as needed):
   rosdep install --from-paths src --ignore-src -r -y

## How to Use

To run the project, follow the instructions below:

1. Start the ROS Master:
   roscore

2. Next, start the TurtleBot3 simulation:
   roslaunch turtlebot3_gazebo turtlebot3_world.launch

3. Run the control node:
   rosrun [your_package] [your_node]

4. Use hand recognition control commands.

## Repository Structure

The structure of the repository is as follows:

turtlebot3-multirobot-control/
- .gitignore               # Ignores unwanted files in the repository
- CMakeLists.txt           # CMake configuration file
- package.xml              # Information about the ROS package
- README.md                # Project documentation
- models/                  # Models used
- results/                 # Generated results
- scripts/                 # Test scripts
- worlds/                  # Simulation environments
- msg/                     # Custom messages

## Contributing

Contributions are welcome! If you want to collaborate on the project, feel free to open an issue or submit a pull request.

## License

Distributed under the MIT License. See the LICENSE file for more information.

## Contact

Lucas Alexandre Zick  
Email: lucas.zick07@edu.udesc.br  
GitHub: LucasZick
