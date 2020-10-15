# autorally_control path_integral scripts

## ml_pipeline
From README:
>This directory contains all the files associated with the neural network dynamics model ML pipeline. MPPI requires a model of
the system dynamics (i.e. what will be the vehicle's next state if 30% steering and 60% throttle are applied). The ICRA 2017 paper,
[Information Theoretic MPC for Model-Based Reinforcement Learning](https://ieeexplore.ieee.org/document/7989202), used a shallow and narrow neural network for their experiment with the AutoRally vehicle. Details on how this network was trained
can be found in the paper and in the [models README](https://github.com/rdesc/autorally/tree/rdesc-melodic-devel/autorally_control/src/path_integral/params/models#autorally_nnet_09_12_2018npz).
The __ml_pipeline__ directory provides a robust and scalable framework for generating models of different vehicle dynamics in order to add MPPI support for additional robots.

Read the full [ml_pipeline README here](https://github.com/rdesc/autorally/blob/rdesc-melodic-devel/autorally_control/src/path_integral/scripts/ml_pipeline/README.md).

## ssl_vision
From README:
>In order to test the ml pipeline, real world ground truth vehicle data is required. __ssl_vision__ contains the work associated
with setting up and validating an overhead vision system to collect ground truth data.

Read the [ssl_vision README here](https://github.com/rdesc/autorally/blob/rdesc-melodic-devel/autorally_control/src/path_integral/scripts/ssl_vision/README.md).

## scripts from AutoRally/autorally
The following scripts come from the parent repo:
- __lap_stats.py__ - rospy node which publishes running lap statistics. It is launched by default in [path_integral_nn.launch](https://github.com/rdesc/autorally/blob/rdesc-melodic-devel/autorally_control/launch/path_integral_nn.launch).
- __track_converter.py__ - converts a .txt map file to .npz file
- __track_generator.py__ - generates a costmap from a map image

NOTE: __track_converter.py__ and __track_generator.py__ lack documentation from AutoRally/autorally
