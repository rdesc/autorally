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

// TODO: rename
void loadParams(std::map<std::string,XmlRpc::XmlRpcValue>* params, ros::NodeHandle nh)
{
  std::string ros_namespace = nh.getNamespace();
  ros_namespace += "/";

  // get all param keys from ros param server
  std::vector<std::string> keys;
  nh.getParamNames(keys);

  XmlRpc::XmlRpcValue val;

  ROS_INFO("Loading parameters from ROS parameter server in namespace '%s' into map...", ros_namespace.c_str());
  ROS_INFO("XmlRpcValue types: 1 == Boolean; 2 == Int; 3 == Double; 4 == String");

  for (auto & key : keys) {
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

// FIXME: not the prettiest
namespace pt = boost::property_tree;
const pt::ptree& empty_ptree()
{
  static pt::ptree t;
  return t;
}
void parseXML(std::map<std::string,XmlRpc::XmlRpcValue>* params, const std::string& filename)
{
  // Create empty property tree object
  pt::ptree tree;

  // Parse the XML into the property tree.
  pt::read_xml(filename, tree);

  // Get the part of the xml containing the mppi node params (NOTE: assumes it is the first one in the roslaunch file)
  const pt::ptree & formats = tree.get_child("launch.node", empty_ptree());

  // init some variables
  std::string key;
  std::string string_val;
  XmlRpc::XmlRpcValue val;
  std::string param_type = "str"; // default is 'str'

  // list of configured parameter types
  std::vector<std::string> param_types{"str", "int", "double", "bool"};

  ROS_INFO("Loading parameters from roslaunch file '%s' into map...", filename.c_str());
  ROS_INFO("XmlRpcValue types: 1 == Boolean; 2 == Int; 3 == Double; 4 == String");

  BOOST_FOREACH(const pt::ptree::value_type & f, formats) {
    // get the params and iterate over each one
    const pt::ptree & attributes = f.second.get_child("<xmlattr>", empty_ptree());
    BOOST_FOREACH(const pt::ptree::value_type &v, attributes) {
      if ((std::string)v.first == "name") {
        // set key name
        key = v.second.data();
      } else if ((std::string)v.first == "type") {
        // set the param type
        param_type = v.second.data();
        // check if param type is known
        if (!(std::find(std::begin(param_types), std::end(param_types), param_type) != std::end(param_types))) {
          ROS_WARN("Not configured for parameter type '%s'! Setting type to default: 'str'", param_type.c_str());
          param_type = "str";
        }
      } else if ((std::string)v.first == "value") {
        // set value
        string_val = v.second.data();
      } else {
        ROS_WARN("Not configured for parameter attribute '%s'! Will ignore...", v.first.data());
      }
    }
    // convert value to correct type
    if (param_type == "int") val = std::stoi(string_val);
    else if (param_type == "double") val = std::stod(string_val);
    else if (param_type == "bool") val = (string_val == "true");
    else val = string_val;

    // add to map if not yet added
    if (params->find(key) == params->end() && !key.empty()) {
      (*params)[key] = val;
      ROS_INFO("Loaded param '%s' with type '%d'", key.c_str(), val.getType());
    }
    }
  }
}
