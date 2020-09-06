# NVIDIA Visual Profiler
This document provides information on how to setup the NVIDIA profiler with autorally CUDA code, specifically how to profile
__path_integral_nn.launch__.

## Setting up 
1. Follow the instructions from the [NVIDIA documentation](https://docs.nvidia.com/cuda/nsight-eclipse-plugins-guide/index.html#using-nsight)
 to setup and install Nsight Eclipse edition. Make sure these instructions match your CUDA Toolkit version. [Here](https://docs.nvidia.com/cuda/nsight-eclipse-edition-getting-started-guide/index.html#abstract)
 are the same instructions but for the CUDA Toolkit v10.2.89 which requires the annoying step of installing Java Runtime Environment 1.8.
2. Make sure the environment variables PATH and LD_LIBRARY_PATH are properly configured! The following is an example of what needs to be added to your .bashrc (.zshrc) file.
         
         # setup CUDA env variables
         export PATH=$PATH:/usr/local/cuda-10.2/bin
         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
3. Launch the script [nsight.sh](https://github.com/rdesc/autorally/blob/rdesc-melodic-devel/autorally_control/nsight.sh) (or nsight.zsh depending on shell) via `sudo nsight.zsh`. Note, line 18  `/usr/local/cuda-10.2/bin/nsight -vm /usr/lib/jvm/jre1.8.0_151/bin/java &`
 most likely needs to be modified to match your setup. Key thing is that this part of the script launches the nsight IDE in the background so that `roscore` can be executed in the next line.
4. Follow the instructions [here](http://wiki.ros.org/IDEs#Eclipse) starting from step 2 "Creating the Eclipse project files" to setup Eclipse with ROS.
 For step 3, if there are building or indexing errors, they can be fixed by importing the entire catkin workspace into eclipse instead of just the AutoRally package.
 This post from [ros answers](https://answers.ros.org/question/52013/catkin-and-eclipse/?answer=174069#post-id-174069) also has some good discussion
 for debugging ROS + eclipse setup.
5. Modify the code you'd like to profile such that it can be executed as a binary without calling `roslaunch` or `rosrun`. This usually 
means removing the code's dependency on the ros param server. This was done for __path_integral_nn.launch__ by using a C++ XML parser to
manually parse the key-value parameters from the roslaunch file. Note, the parameter value type needs to be specified in the launch file for each parameter (default is string).
The code that parses the launch file is inside [param_getter.cpp](https://github.com/rdesc/autorally/blob/rdesc-melodic-devel/autorally_control/src/path_integral/param_getter.cpp#L75).
6. Modify the code you'd like to profile such that it terminates cleanly (returns exit code 0) after some finite time. For __path_integral_nn.launch__,
if the parameter [profiler_max_iter](https://github.com/rdesc/autorally/blob/rdesc-melodic-devel/autorally_control/launch/path_integral_nn.launch#L19) is set then
MPPI will terminate cleanly after running the specified number of iterations.
7. Test launching the binary version of the ROS code by executing it in a configured ROS environment with a running rosmaster. 
ROS binaries are either found in _catkin_ws/devel/lib/$package/_ or _catkin_ws/install/lib/$package/_. For the case of __path_integral_nn.launch__
execute `./path_integral_nn` inside the directory _catkin_ws/devel/lib/autorally_control/_.

# Running profiler
1. Make sure Nsight eclipse is open with a properly configured project
2. Build project and check to see if the binaries such as _path_integral_nn_ show up (they may need to be added manually).
3. Select the binary and click Profile.

More information about the different components to Nsight Eclipse https://developer.nvidia.com/nsight-eclipse-edition.
