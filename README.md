# AutoRally

Forked repo of the [AutoRally](https://github.com/AutoRally/autorally) project. Go to the parent repository to read through the original README.md which includes setup/build instructions.

The following items are the contributions made in this forked repo:
  - __Removed ROS coupling__ - Removed some coupling between MPPI and ROS. No longer need the `roslaunch` wrapper to launch path_integral_nn, can now just run it as a standalone binary. [Link](https://github.com/rdesc/autorally/blob/rdesc-melodic-devel/autorally_control/src/path_integral/path_integral_main.cu)
  - __Profiler instructions__ - PROFILE.md provides specific instructions on how to setup the NVIDIA Visual Profiler to debug + profile MPPI. [Link](https://github.com/rdesc/autorally/blob/rdesc-melodic-devel/autorally_control/profiler.md)
  - __ML Pipeline__ - Adds a scalable framework to train a neural network dynamics model so that MPPI supports other vehicle models other than AutoRally. [Link](https://github.com/rdesc/autorally/tree/rdesc-melodic-devel/autorally_control/src/path_integral/scripts/ml_pipeline)
  - __SSL Vision__ - Adds setup instructions to configure SSL Vision and python scripts to measure the sensor noise of the overhead vision system. [Link](https://github.com/rdesc/autorally/tree/rdesc-melodic-devel/autorally_control/src/path_integral/scripts/ssl_vision)
