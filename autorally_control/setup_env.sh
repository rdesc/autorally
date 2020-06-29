#!/bin/bash

# this script shows the various steps for setting up autorally + ROS
# ideally these steps are appended to the end of the .bashrc file 

# useful alias
alias cs="cd ~/catkin_ws"

# setup ROS env variables
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash

# setup autorally env variables
source ~/catkin_ws/src/autorally/autorally_util/setupEnvLocal.sh

# setup CUDA env variables
export PATH=$PATH:/usr/local/cuda-10.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
