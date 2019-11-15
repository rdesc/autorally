This file contains the plan for the implementation of tube-mpc, as specified in the 2018 paper

## TODO
[] Create a simple PID controller to generate feedback terms to track the generated controls
[] Stop feeding state into MPPI, simply interpolate along generated expected state from previous controls instead
[] Generalize simple PID controller to a iLQG controller
[] Add a second instance of the MPPI controller that optimizes from the robot current state (instead of expected state) and overwrite the first MPPI controller with this second one if it's better

## Notes
