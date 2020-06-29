#!/bin/zsh

# This script automates the setup required for the nsight profiler to work properly with autorally code specifically the mppi controller

# Follow these instructions for solving permission issue with Performance Counters
# https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters

# check script launched as sudo
if [ "$EUID" -ne 0 ]
  then echo "Please run script as sudo!"
  exit
fi

source /opt/ros/melodic/setup.zsh
source ~/catkin_ws/devel/setup.zsh
source ~/catkin_ws/src/autorally/autorally_util/setupEnvLocal.zsh

/usr/local/cuda-10.2/bin/nsight -vm /usr/lib/jvm/jre1.8.0_151/bin/java &

roscore
