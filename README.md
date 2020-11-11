# AutoRally

Forked repo of the [AutoRally](https://github.com/AutoRally/autorally) project. Go to the parent repository to read through the original README.md which includes setup/build instructions.

The following items are the contributions made in this forked repo:
  - Removed some coupling between MPPI and ROS. No longer need the `roslaunch` wrapper to launch [path_integral_nn](https://github.com/rdesc/autorally/blob/rdesc-melodic-devel/autorally_control/src/path_integral/path_integral_main.cu), can now just run it as a standalone binary.
  - [PROFILE.md](https://github.com/rdesc/autorally/blob/rdesc-melodic-devel/autorally_control/profiler.md) provides specific instructions on how to setup the [NVIDIA Visual Profiler](https://developer.nvidia.com/nvidia-visual-profiler) to debug + profile MPPI. 
  - Adds a scalable framework to train a neural network dynamics model so that MPPI supports other vehicle models other than AutoRally. [ML Pipeline directory](https://github.com/rdesc/autorally/tree/rdesc-melodic-devel/autorally_control/src/path_integral/scripts/ml_pipeline)
  - Adds setup instructions to configure [SSL Vision](https://github.com/RoboCup-SSL/ssl-vision) and python scripts to measure the sensor noise of the overhead vision system. [SSL Vision directory](https://github.com/rdesc/autorally/tree/rdesc-melodic-devel/autorally_control/src/path_integral/scripts/ssl_vision)
