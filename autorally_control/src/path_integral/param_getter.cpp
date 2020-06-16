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

void loadParams(std::map<std::string,XmlRpc::XmlRpcValue>* params, ros::NodeHandle nh)
{
  std::string ros_namespace = nh.getNamespace().c_str();
  ros_namespace += "/";

  // get all param keys from ros param server
  std::vector<std::string> keys;
  nh.getParamNames(keys);

  std::string key;
  XmlRpc::XmlRpcValue val;

  ROS_INFO("Loading parameters in namespace '%s' into map...", ros_namespace.c_str());
  ROS_INFO("XmlRpcValue types: 1 == Boolean; 2 == Int; 3 == Double; 4 == String");

  for (int i = 0; i < keys.size(); i++) {
    key = keys[i];
    nh.getParam(key, val);
    // only add param if key is inside the 'mppi_controller' ROS namespace
    if (key.find(ros_namespace) != std::string::npos) {
      // remove the namespace before setting the key
      key = key.substr(ros_namespace.length());
      // add to map
      (*params)[key] = val;
      ROS_INFO("Loaded param '%s' with type '%d'", key.c_str(), val.getType());
    }
  }
}

}


