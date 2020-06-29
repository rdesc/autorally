#!/bin/zsh

# this script shows the various steps for setting up autorally + ROS
# ideally these steps are appended to the end of the .zshrc file 

# useful alias
alias cs="cd ~/catkin_ws"

# setup ROS env variables
source /opt/ros/melodic/setup.zsh
source ~/catkin_ws/devel/setup.zsh

# setup autorally env variables
source ~/catkin_ws/src/autorally/autorally_util/setupEnvLocal.zsh

# setup CUDA env variables
export PATH=$PATH:/usr/local/cuda-10.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
