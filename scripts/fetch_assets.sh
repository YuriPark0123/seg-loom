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
