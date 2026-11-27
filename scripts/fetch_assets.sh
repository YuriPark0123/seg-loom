#!/usr/bin/env bash
# Third-party robot descriptions used by configs/robots/*.yaml (pinned commits).
set -e
cd "$(dirname "$0")/.."
mkdir -p assets/src && cd assets/src
get() { [ -d "$2" ] || git clone -q "$1" "$2"; git -C "$2" checkout -q "$3"; echo "$2 @ $3"; }
get https://github.com/TheRobotStudio/SO-ARM100.git SO-ARM100 141aea7
get https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git Universal_Robots_ROS2_Description 0eb1a86
get https://github.com/Gepetto/example-robot-data.git example-robot-data d9fd825
get https://github.com/agilexrobotics/piper_ros.git piper_ros ac41fcb

# ---- extra real robots (scripts/onboard_robots.py -> configs/robots/<name>.yaml, configs/robots/extra_splits.yaml)
mkdir -p extra && cd extra
get https://github.com/ianathompson/eDO_description.git edo 17b3f92
get https://github.com/robot-descriptions/fanuc_m710ic_description.git fanuc_m710ic_description d12af44
get https://github.com/ros-industrial/fanuc.git fanuc d8f42bd
get https://github.com/Gepetto/example-robot-data.git example-robot-data d0d9098
get https://github.com/Kinovarobotics/ros2_kortex.git ros2_kortex ca433b0
get https://github.com/facebookresearch/differentiable-robot-model differentiable-robot-model d7bd1b3
get https://github.com/Kinovarobotics/kinova-ros.git kinova-ros 924781d
get https://github.com/flexivrobotics/flexiv_description.git flexiv_description f33331a
get https://github.com/ROBOTIS-GIT/open_manipulator.git open_manipulator bc555a9
get https://github.com/xArm-Developer/xarm_ros2.git xarm_ros2 5bb832f
get https://github.com/robot-descriptions/i2rt.git i2rt 7809b5b
get https://github.com/Interbotix/interbotix_ros_manipulators.git interbotix_ros_manipulators 0bb2b0e
get https://github.com/RethinkRobotics/sawyer_robot.git sawyer_robot d1ef3d0
# drake is huge: only the iiwa model folder
[ -d drake ] || { git clone -q --filter=blob:none --sparse https://github.com/RobotLocomotion/drake.git drake && \
  git -C drake sparse-checkout set manipulation/models/iiwa_description; }
git -C drake checkout -q 7abea0556 && echo "drake @ 7abea0556 (iiwa_description only)"
