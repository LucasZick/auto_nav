
# Autonav

**Autonav** is a multi-robot control system developed to simplify operator interaction by using autonomous navigation and a simplified control model. This project is designed to operate with multiple TurtleBot3 robots but can be adapted to work with other robots with minimal adjustments.

## Project Purpose

Autonav focuses on:
- Reducing operator interaction time through autonomous navigation.
- Facilitating control of multiple robots with an intuitive interface and simplified commands, recognized by hand point tracking.

## Requirements

To run Autonav, the following components are needed:
- **ROS** (recommended: Noetic)
- **Rviz** (for visualization and interface with the robots)
- **Gazebo** (for simulation)
- **Mediapipe** (for hand point recognition, used to generate navigation commands)

## Project Structure

The project follows the structure of a ROS package, with:
- **scripts**: main code for the control system and commands.
- **launch**: launch files to initialize the environment and robots.
- **worlds**: files to configure simulation environments.

### Test Results

The test results are available in the `results` folder. Note: the project has not yet been tested in a real-world environment.

## Execution Instructions

To run the project, follow these steps:

1. Run the multi-robot environment launch file:
   ```bash
   roslaunch autonav environment_multiple_robots.launch
   ```
2. In another terminal, start the control system:
   ```bash
   roslaunch autonav autonav.launch
   ```

   Alternatively, you can use the command below to run both in a single terminal:
   ```bash
   roslaunch autonav main.launch
   ```