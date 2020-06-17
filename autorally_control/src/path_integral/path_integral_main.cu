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
 * @file path_integral_main.cpp
 * @author Grady Williams <gradyrw@gmail.com>
 * @date May 24, 2017
 * @copyright 2017 Georgia Institute of Technology
 * @brief Main file model predictive path integral control.
 *
 ***********************************************/

//Some versions of boost require __CUDACC_VER__, which is no longer defined in CUDA 9. This is
//the old expression for how it was defined, so should work for CUDA 9 and under.
#define __CUDACC_VER__ __CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 + __CUDACC_VER_BUILD__

#include <autorally_control/path_integral/meta_math.h>
#include <autorally_control/path_integral/param_getter.h>
#include <autorally_control/path_integral/autorally_plant.h>
#include <autorally_control/PathIntegralParamsConfig.h>
#include <autorally_control/path_integral/costs.cuh>

//Including neural net model
#ifdef MPPI_NNET_USING_CONSTANT_MEM__
__device__ __constant__ float NNET_PARAMS[param_counter(6,32,32,4)];
#endif
#include <autorally_control/path_integral/neural_net_model.cuh>
#include <autorally_control/path_integral/car_bfs.cuh>
#include <autorally_control/path_integral/car_kinematics.cuh>
#include <autorally_control/path_integral/generalized_linear.cuh>
#include <autorally_control/path_integral/mppi_controller.cuh>
#include <autorally_control/path_integral/run_control_loop.cuh>

#include <ros/ros.h>
#include <atomic>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

using namespace autorally_control;

#ifdef USE_NEURAL_NETWORK_MODEL__ /*Use neural network dynamics model*/
const int MPPI_NUM_ROLLOUTS__ = 1920; // number of trajectories to sample
const int BLOCKSIZE_X = 8;
const int BLOCKSIZE_Y = 16;
typedef NeuralNetModel<7,2,3,6,32,32,4> DynamicsModel; // this is where you specify size of the neural network layers (6, 32, 32 4)
#elif USE_BASIS_FUNC_MODEL__ /*Use the basis function model* */
const int MPPI_NUM_ROLLOUTS__ = 2560;
const int BLOCKSIZE_X = 16;
const int BLOCKSIZE_Y = 4;
typedef GeneralizedLinear<CarBasisFuncs, 7, 2, 25, CarKinematics, 3> DynamicsModel;
#endif

//Convenience typedef for the MPPI Controller.
typedef MPPIController<DynamicsModel, MPPICosts, MPPI_NUM_ROLLOUTS__, BLOCKSIZE_X, BLOCKSIZE_Y> Controller;

int main(int argc, char** argv) {
  //ROS node initialization
  ros::init(argc, argv, "mppi_controller");
  ros::NodeHandle mppi_node("~");

  //Name of roslaunch file containing parameter config
  std::string config_file = "path_integral_nn.launch";

  //Load setup parameters
  std::map<std::string,XmlRpc::XmlRpcValue> params;
  loadParams(&params, "/home/rdesc/catkin_ws/src/autorally/autorally_control/launch/" + config_file); // FIXME: path is hardcoded

  //Define the mppi costs
  MPPICosts* costs = new MPPICosts(&params);

  //Define the internal dynamics model for mppi
  float2 control_constraints[2] = {make_float2(-.99, .99), make_float2(-.99, (double)params["max_throttle"])};
  //Init dynamics model object
  DynamicsModel* model = new DynamicsModel(1.0 / (int)params["hz"], control_constraints);
  //Load the dynamics model parameters from the specified file path
  model->loadParams((std::string)params["model_path"]);

  //Init control arrays
  float init_u[2] = {(float)(double)params["init_steering"], (float)(double)params["init_throttle"]};
  float exploration_std[2] = {(float)(double)params["steering_std"], (float)(double)params["throttle_std"]};
  //Define the controller
  Controller* actual_state_controller = new Controller(model, costs, exploration_std, init_u, &params);
  Controller* predicted_state_controller = new Controller(model, costs, exploration_std, init_u, &params);

  //Define the autorally plant (a plant model is the mathematical model of the system)
  AutorallyPlant* robot = new AutorallyPlant(mppi_node, &params);

  //Setup dynamic reconfigure callback
  dynamic_reconfigure::Server<PathIntegralParamsConfig> server;
  dynamic_reconfigure::Server<PathIntegralParamsConfig>::CallbackType callback_f;
  callback_f = boost::bind(&AutorallyPlant::dynRcfgCall, robot, _1, _2);
  server.setCallback(callback_f);

  //Get maximum number of iterations if running with profiler
  std::string max_iter_key = "profiler_max_iter";
  int max_iter = (params.count(max_iter_key)) ? (int)params[max_iter_key] : INT_MAX;

  boost::thread optimizer;

  std::atomic<bool> is_alive(true);
  optimizer = boost::thread(
      &runControlLoop<Controller>, predicted_state_controller,
      actual_state_controller, robot, &params, &is_alive, max_iter);

  ros::spin();

  //Shutdown procedure
  is_alive.store(false);
  optimizer.join();
  robot->shutdown();
  actual_state_controller->deallocateCudaMem();
  predicted_state_controller->deallocateCudaMem();
  delete robot;
  delete actual_state_controller;
  delete predicted_state_controller;
  delete costs;
  delete model;
}
