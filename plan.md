This file contains the plan for the implementation of tube-mpc, as specified in the 2018 paper

## TODO
[x] Stop feeding state into MPPI, simply interpolate along generated expected state from previous controls instead
[] Use DDP as ancillary controller
[] Changed DDP to iLQR to match paper? (**CHECK WITH IAN FIRST**)
[] Add a second instance of the MPPI controller that optimizes from the robot current state (instead of expected state) and arbitrate between them via the algorithm described in the paper.

## Notes
