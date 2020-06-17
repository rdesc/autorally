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
 * @file param_getter.h
 * @author Grady Williams <gradyrw@gmail.com>
 * @date May 24, 2017
 * @copyright 2017 Georgia Institute of Technology
 * @brief Function for grabbing parameters from the ROS
 * parameter server.
 ***********************************************/

#ifndef PARAM_GETTER_H_
#define PARAM_GETTER_H_

#include <unistd.h>
#include <string>
#include <vector_types.h>
#include <ros/ros.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <iomanip>


namespace autorally_control {

inline bool fileExists (const std::string& name) {
    return ( access( name.c_str(), F_OK ) != -1 );
}

/**
 * @brief Queries ros param server to get value of parameter
 * @tparam T The parameter value type (e.g. int, double, bool etc.)
 * @param paramName Name of parameter to query
 * @param nh ROS node handle
 */
template <typename T>
T getRosParam(std::string paramName, ros::NodeHandle nh)
{
  std::string key;
  T val;
  bool found = nh.searchParam(paramName, key);
  if (!found){
    ROS_ERROR("Could not find parameter name '%s' in tree of node '%s'", 
              paramName.c_str(), nh.getNamespace().c_str());
  }
  else {
    nh.getParam(key, val);
  }
  return val;
}

/**
 * @brief Load params into map by querying ROS param server
 * @param params A pointer to the params map which contains the runtime configured parameters
 * @param nh ROS node handle
 */
void loadParams(std::map<std::string,XmlRpc::XmlRpcValue>* params, ros::NodeHandle nh);

/**
 * @brief Load params into map by parsing roslaunch (xml) file
 * @param params A pointer to the params map which contains the runtime configured parameters
 * @param file_path File path of the roslaunch file to parse (NOTE: assumes mppi controller node is the first node in the tree)
 */
void loadParams(std::map<std::string,XmlRpc::XmlRpcValue>* params, const std::string& file_path);
}

#endif /*PARAM_GETTER_H_*/


