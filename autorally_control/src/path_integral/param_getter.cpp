/*
* Software License Agreement (BSD License)
* Copyright (c) 2013, Georgia Institute of Technology
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/**********************************************
 * @file param_getter.cpp
 * @author Grady Williams <gradyrw@gmail.com>
 * @date May 24, 2017
 * @copyright 2017 Georgia Institute of Technology
 * @brief Function for grabbing parameters from the ROS
 * parameter server.
 ***********************************************/

#include <autorally_control/path_integral/param_getter.h>

namespace autorally_control {

void loadParams(SystemParams* params, ros::NodeHandle nh, bool from_roslaunch)
{
  if(from_roslaunch) {
    params->debug_mode = getRosParam<bool>("debug_mode", nh);
    params->hz = getRosParam<int>("hz", nh);
    params->num_timesteps = getRosParam<int>("num_timesteps", nh);
    params->num_iters = getRosParam<int>("num_iters", nh);
    params->x_pos = getRosParam<double>("x_pos", nh);
    params->y_pos = getRosParam<double>("y_pos", nh);
    params->heading = getRosParam<double>("heading", nh);
    params->gamma = getRosParam<double>("gamma", nh);
    params->init_steering = getRosParam<double>("init_steering", nh);
    params->init_throttle = getRosParam<double>("init_throttle", nh);
    params->steering_std = getRosParam<double>("steering_std", nh);
    params->throttle_std = getRosParam<double>("throttle_std", nh);
    params->max_throttle = getRosParam<double>("max_throttle", nh);
    params->model_path = getRosParam<std::string>("model_path", nh);
    params->use_only_actual_state_controller = getRosParam<bool>(
            "use_only_actual_state_controller", nh);
    params->use_only_predicted_state_controller = getRosParam<bool>(
            "use_only_predicted_state_controller", nh);
  } else {
    // TODO: get rid of hardcoded params -> use a config.txt file maybe?
    printf("here");
    params->debug_mode = true;
    params->hz = int(50);
    params->num_timesteps = int(100);
    params->num_iters = int(1);
    params->x_pos = double(0.0);
    params->y_pos = double(0.0);
    params->heading = double(2.35);
    params->gamma = double(0.15);
    params->init_steering = double(0.0);
    params->init_throttle = double(0.0);
    params->steering_std = double(0.275);
    params->throttle_std = double(0.3);
    params->max_throttle = double(0.65);
    params->model_path = "/home/rdesc/catkin_ws/src/autorally/autorally_control/src/path_integral/params/models/autorally_nnet_09_12_2018.npz";
    params->use_only_actual_state_controller = false;
    params->use_only_predicted_state_controller = false;
  }
}

}


