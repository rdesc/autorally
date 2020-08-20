"""Compares neural network output to ODE integration
"""
import numpy as np
import torch
import pandas as pd
import os
from scipy.integrate import odeint

from utils import setup_model, npz_to_torch_model, compute_state_ders, state_variable_plots, state_der_plots


def load_model(f, nn_layers=None, from_npz=False):
    """
    Loads neural network architecture specified in MPPI code, and loads weights and biases
    :param f: path of either .npz file or torch model
    :type f: str
    :param nn_layers: list consisting of number of nodes for each layer in network
    :param from_npz: if True, load .npz file which contains weights and biases of neural network
    :return: torch model
    """
    # setup model architecture
    model = setup_model(layers=nn_layers, verbose=False)
    # load torch model
    if not from_npz:
        model.load_state_dict(torch.load(f))

    else:
        # load npz file
        model = npz_to_torch_model(f, model)

    # load model onto GPU
    model.to(device)

    return model


def vehicle_dynamics_sys_ode(state, t, p):
    """
    Defines the first order differential equations for the vehicle model equations of motion
    :param state: vector of the state variables: state = [x_pos, y_pos, yaw, roll, u_x, u_y, yaw_mder]
    :param t: time
    :param p: vector of the parameters: p = [st, thr, a1, a2]
    """
    st, thr, a1, a2 = p
    f = [np.cos(state[2]) * state[4] - np.sin(state[2]) * state[5],
         np.sin(state[2]) * state[4] + np.cos(state[2]) * state[5],
         -1 * state[6], 0, a2 * thr, 0, a1 * st]  # ignores roll and u_y
    return f


def model_vehicle_dynamics(steering, throttle, time_horizon, nn_model_path="", nn_layers=None, time_step=0.01, state_dim=7,
                           init_cond=None, linear_varying_ctrls=False):
    """
    Calculates vehicle dynamics when steering and throttle controls are applied
    :param steering: steering control input to apply
    :param throttle: throttle control input to apply
    :param time_horizon: total time to propagate dynamics
    :param nn_model_path: path to neural network model
    :param nn_layers: list consisting of number of nodes for each layer in network
    :param time_step: the time interval between each update step (dt)
    :param state_dim: the size of state space
    :param init_cond: initial conditions to states (length must match state_dim)
    :param linear_varying_ctrls: if True, apply control inputs linearly varying
    """
    # define number of steps for integration
    num_steps = int(time_horizon / time_step + 1)
    prefix = "st=" + str(steering) + "_thr=" + str(throttle)
    # create directory to store trajectory files
    dir_path = "trajectory_files/" + prefix + "/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # initial conditions
    if init_cond is None:
        init_cond = np.zeros(state_dim)

    # init array to store controls
    ctrl = np.full((num_steps, 2), 0, np.float)
    ctrl[:, 0] = steering
    ctrl[:, 1] = throttle

    x_pos_0, y_pos_0, yaw_0, roll_0, u_x_0, u_y_0, yaw_mder_0 = init_cond
    plt_title = ("steering=%s, throttle=%s, time_horizon=%s\n"
                 "x_pos_0=%s, y_pos_0=%s, yaw_0=%s, roll_0=%s, u_x_0=%s, u_y_0=%s, yaw_mder_0=%s" %
                 (steering, throttle, time_horizon, x_pos_0, y_pos_0, yaw_0, roll_0, u_x_0, u_y_0, yaw_mder_0))

    # integrate using scipys odeint
    # NOTE: these constants are not very well tuned
    # constant to multiply steering by
    a1 = 2
    # constant to multiply throttle by
    a2 = 7

    # ODE solver parameters
    abserr = 1.0e-8
    relerr = 1.0e-6
    time = np.linspace(0, time_horizon, num_steps)

    # init param vectors
    p = [steering, throttle, a1, a2]
    # call the ODE solver
    sol = odeint(vehicle_dynamics_sys_ode, init_cond, time, args=(p,), atol=abserr, rtol=relerr)

    # convert to pandas DataFrames
    cols = ["time", "x_pos", "y_pos", "yaw", "roll", "u_x", "u_y", "yaw_mder", "steering", "throttle"]
    df_ode = pd.DataFrame(data=np.concatenate((np.reshape(time, (len(time), 1)), sol, ctrl), axis=1), columns=cols)
    # save to disk
    df_ode.to_csv(dir_path + "odeint_state_variables.csv", index=False, header=True)

    # apply pre-trained neural network
    # init state variables
    state = np.full((num_steps, state_dim), 0, np.float)  # x_pos, y_pos, yaw, roll, u_x, u_y, yaw_mder
    # set initial conditions
    state[0] = init_cond
    # array to store all state derivatives
    state_ders = np.full((num_steps, state_dim), 0, np.float)

    if linear_varying_ctrls:
        # apply linearly increasing throttle and steering
        ctrl[:, 0] = np.linspace(0, steering, num_steps)
        ctrl[:, 1] = np.linspace(0, throttle, num_steps)

    # load model
    model = load_model(nn_model_path, nn_layers=nn_layers)
    with torch.no_grad():
        model.eval()

        # iterate through each step of trajectory
        for idx in range(num_steps - 1):
            # prep input to feed to neural network [roll, u_x, u_y, yaw_mder, steering, throttle]
            x = torch.tensor(
                [state[idx][3], state[idx][4], state[idx][5], state[idx][6], ctrl[idx][0], ctrl[idx][1]])

            # get the current state
            curr_state = state[idx]

            # get output of neural network
            y_pred = model(x.float().to(device))
            # convert to numpy array
            y_pred = y_pred.detach().cpu().numpy()

            # compute the state derivatives
            state_der = compute_state_ders(curr_state, y_pred)

            # update states
            state[idx + 1] = curr_state + state_der * time_step

            # save state derivatives
            state_ders[idx + 1] = state_der

    # add state derivative column names
    der_cols = ["x_pos_der", "y_pos_der", "yaw_der", "roll_der", "u_x_der", "u_y_der", "yaw_mder_der"]
    # convert numpy arrays to pandas DataFrames
    df_nn = pd.DataFrame(data=np.concatenate((np.reshape(time, (len(time), 1)), state, ctrl, state_ders), axis=1), columns=np.concatenate((cols, der_cols)))
    # save to disk
    df_nn.to_csv(dir_path + "nn_state_variables.csv", index=False, header=True)

    # plot trajectories
    cols.remove("time")
    state_variable_plots(df1=df_ode, df2=df_nn, dir_path=dir_path, plt_title=plt_title, cols_to_include=cols)
    # plot state derivatives
    state_der_plots(df_nn, dir_path=dir_path, cols_to_include=der_cols)


def main():
    nn_layers = [6, 32, 32, 4]
    torch_model_path = "../params/models/torch_model_autorally_nnet.pt"
    model_from_npz = False
    if model_from_npz:
        file_path = "../params/models/autorally_nnet_09_12_2018.npz"
        m = load_model(file_path, nn_layers=nn_layers, from_npz=True)
        torch.save(m.state_dict(), torch_model_path)

    # control constraints to match path_integral_nn.launch
    # throttle range [-0.99, 0.65]
    # steering range [-0.99, 0.99]
    # initial conditions
    cond = [0, 0, 0, 0, 0, 0, 0]  # x_pos, y_pos, yaw, roll, u_x, u_y, yaw_mder
    horizon = 5
    st = [0.0, 0.0, 0.5, 0.99, -0.5]
    thr = [0.65, -0.99, 0.5, 0.5, 0.3]

    for s, t in zip(st, thr):
        print("Modeling vehicle dynamics with controls: steering=%.2f, throttle=%.2f..." % (s, t))
        model_vehicle_dynamics(nn_model_path=torch_model_path, nn_layers=nn_layers, steering=s, throttle=t, time_horizon=horizon, init_cond=cond)


if __name__ == '__main__':
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # TODO: use arg parser
    main()
