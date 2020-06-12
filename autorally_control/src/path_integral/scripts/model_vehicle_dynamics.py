import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.integrate import odeint


def load_model(f, from_npz=False):
    """
    Loads neural network architecture specified in MPPI code, and loads weights and biases
    :param f: path of either .npz file or torch model
    :param from_npz: if True, load .npz file which contains weights and biases of neural network
    :return: torch model
    """
    # setup model architecture
    model = nn.Sequential(nn.Linear(6, 32),
                          nn.Tanh(),
                          nn.Linear(32, 32),
                          nn.Tanh(),
                          nn.Linear(32, 4))
    # load torch model
    if not from_npz:
        model.load_state_dict(torch.load(f))

    else:
        # load npz file
        layers = np.load(f)
        bias_names = ["dynamics_b1", "dynamics_b2", "dynamics_b3"]
        layer_names = ["dynamics_W1", "dynamics_W2", "dynamics_W3"]
        # load weights and biases into appropriate layers
        for i in range(3):
            model[i * 2].bias = nn.Parameter(torch.from_numpy(layers[bias_names[i]]).float(), requires_grad=False)
            model[i * 2].weight = nn.Parameter(torch.from_numpy(layers[layer_names[i]]).float(), requires_grad=False)

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
    f = [np.cos(state[2])*state[4] - np.sin(state[2])*state[5],
         np.sin(state[2])*state[4] + np.cos(state[2])*state[5],
         -1 * state[6], 0, a2 * thr, 0, a1 * st]  # ignores roll and u_y
    return f


def model_vehicle_dynamics(steering, throttle, time_horizon, f="", time_step=0.01, state_dim=7, init_cond=None, neural_net=True, linear_varying_ctrls=False):
    """
    Calculates vehicle dynamcis when steering and throttle controls are applied
    :param steering: steering control input to apply
    :param throttle: throttle control input to apply
    :param time_horizon: total time to propagate dynamics
    :param f: path to neural network model
    :param time_step: the time interval between each update step
    :param state_dim: the size of state space
    :param init_cond: initial conditions to states (length must match state_dim)
    :param neural_net: if True, get dynamics from neural network, otherwise solve sys of ODEs directly with scipy odeint
    :param linear_varying_ctrls: if True, apply control inputs linearly varying
    """
    # define number of steps for integration
    num_steps = int(time_horizon / time_step + 1)
    suffix = "_odeint" if not neural_net else "_nn"
    prefix = "st=" + str(steering) + "_thr=" + str(throttle)
    # create directory to store trajectory files
    dir_path = "trajectory_files/" + prefix + suffix + "/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # initial conditions
    if init_cond is None:
        init_cond = np.zeros(state_dim)

    x_pos_0, y_pos_0, yaw_0, roll_0, u_x_0, u_y_0, yaw_mder_0 = init_cond
    plt_title = ("steering=%s, throttle=%s, time_horizon=%s\n"
                 "x_pos_0=%s, y_pos_0=%s, yaw_0=%s, roll_0=%s, u_x_0=%s, u_y_0=%s, yaw_mder_0=%s" %
                 (steering, throttle, time_horizon, x_pos_0, y_pos_0, yaw_0, roll_0, u_x_0, u_y_0, yaw_mder_0))

    if not neural_net:
        # integrate using scipys odeint
        # constant to multiply steering by
        a1 = 1
        # constant to multiply throttle by
        a2 = 1

        # ODE solver parameters
        abserr = 1.0e-8
        relerr = 1.0e-6
        t = np.linspace(0, time_horizon, num_steps)

        # init param vectors
        p = [steering, throttle, a1, a2]
        # call the ODE solver
        sol = odeint(vehicle_dynamics_sys_ode, init_cond, t, args=(p,), atol=abserr, rtol=relerr)

        # convert to pandas DataFrames
        data_df = pd.DataFrame(data=np.concatenate((np.reshape(t, (len(t), 1)), sol), axis=1),
                               columns=["time", "x_pos", "y_pos", "yaw", "roll", "u_x", "u_y", "yaw_mder"])
        # save to disk
        data_df.to_csv(dir_path + "odeint_state_variables.csv", index=False, header=True)
        # plot trajectories
        state_variable_plots(data_df, dir_path, plt_title=plt_title, cols_to_exclude=["time"])  # TODO roll, u_y?
        return

    ############################
    # apply pre-trained neural network
    # init state variables
    state = np.full((num_steps, state_dim), 0, np.float)  # x_pos, y_pos, yaw, roll, u_x, u_y, yaw_mder
    # set initial conditions
    state[0] = init_cond
    # init array for state derivatives
    state_der = np.zeros(state_dim)  # d/dt [x_pos, y_pos, yaw, roll, u_x, u_y, yaw_mder]
    # init array to store controls
    ctrl = np.full((num_steps, 2), 0, np.float)

    if linear_varying_ctrls:  # TODO: would this even provide us anything interesting?
        # apply linearly increasing throttle and steering
        ctrl[:, 0] = np.linspace(0, steering, num_steps)
        ctrl[:, 1] = np.linspace(0, throttle, num_steps)
    else:
        ctrl[:, 0] = steering
        ctrl[:, 1] = throttle

    # init array to store outputs of neural network
    # nn_output = np.full((num_steps, 4), 0, np.float)  # d/dt[roll, u_x, u_y, yaw_mder]

    # load model
    model = load_model(f)
    with torch.no_grad():
        model.eval()

        # iterate through each step of trajectory
        for idx in range(num_steps - 1):
            # prep input to feed to neural network [roll, u_x, u_y, yaw_mder, steering, throttle]
            x = torch.tensor([state[idx][3], state[idx][4], state[idx][5], state[idx][6], ctrl[idx][0], ctrl[idx][1]])

            # compute kinematics (match implementation with NeuralNetModel::computeKinematics in neural_net_model.cu)
            state_der[0] = np.cos(state[idx][2])*state[idx][4] - np.sin(state[idx][2])*state[idx][5]
            state_der[1] = np.sin(state[idx][2])*state[idx][4] + np.cos(state[idx][2])*state[idx][5]
            state_der[2] = -1. * state[idx][6]

            # get output of neural network
            y_pred = model(x.float().to(device))
            # convert to numpy array
            y_pred = y_pred.detach().cpu().numpy()

            # compute dynamics
            state_der[3], state_der[4], state_der[5], state_der[6] = y_pred

            # update states
            state[idx + 1] = state[idx] + state_der * time_step

            # ensure state derivative set back to zero
            state_der = np.zeros(state_dim)

    # convert numpy arrays to pandas DataFrames
    time = np.linspace(0, time_horizon, num_steps)
    data_df = pd.DataFrame(data=np.concatenate((np.reshape(time, (len(time), 1)), state), axis=1),
                           columns=["time", "x_pos", "y_pos", "yaw", "roll", "u_x", "u_y", "yaw_mder"])
    # save to disk
    data_df.to_csv(dir_path + "nn_state_variables.csv", index=False, header=True)
    # plot trajectories
    state_variable_plots(data_df, dir_path, plt_title=plt_title, cols_to_exclude=["time"])


def state_variable_plots(df, dir_path, plt_title="", cols_to_exclude=None):
    """
    Outputs trajectory plot, states vs. time, and state vs. state pair plots
    :param df: pandas DataFrame must contain columns 'x_pos', 'y_pos', and 'time'
    :param dir_path: Path of directory to store plots
    :param plt_title: Title of plots
    :param cols_to_exclude: Columns to exclude from state vs. time and state vs. state plots
    """
    # plot trajectory in fixed global frame
    fig = plt.figure(figsize=(8, 6))
    plt.xlabel("x position")
    plt.axis('equal')
    plt.ylabel("y position")
    plt.title("2D trajectory\n" + plt_title)
    plt.plot(df['x_pos'], df['y_pos'])
    plt.savefig(dir_path + "trajectory.png")

    # plot all state variables along a common time axis
    fig = plt.figure(figsize=(8, 10))
    count_states = len(df.columns) - len(cols_to_exclude)
    idx = 1
    for _, column in enumerate(df.columns):
        # skip columns to exclude
        if column in cols_to_exclude:
            continue
        ax = fig.add_subplot(count_states, 1, idx)
        ax.set_ylabel(column)
        plt.plot(df["time"], df[column])
        # plt.grid(True, which='both', axis='both')
        if not (idx == count_states):
            ax.set_xticklabels([])
        idx += 1
    plt.xlabel("time")
    plt.suptitle("states vs. time\n" + plt_title)
    plt.savefig(dir_path + "states_vs_time.png", dpi=300)

    # plot state vs state
    pd.plotting.scatter_matrix(df.drop(cols_to_exclude, axis=1), alpha=0.8, figsize=(10, 10), diagonal='hist')
    plt.suptitle("state vs. state pair plot\n" + plt_title)
    plt.savefig(dir_path + "state_vs_state.png", dpi=300)


if __name__ == '__main__':
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    torch_model_path = "../params/models/torch_model_autorally_nnet.pt"
    model_from_npz = False
    if model_from_npz:
        file_path = "../params/models/autorally_nnet_09_12_2018.npz"
        m = load_model(file_path, from_npz=True)
        torch.save(m.state_dict(), torch_model_path)

    # control constraints to match path_integral_nn.launch
    # throttle range [-0.99, 0.65]
    # steering range [0.99, -0.99]
    initial_conditions = [0, 0, 0, 0, 0, 0, 0]  # x_pos, y_pos, yaw, roll, u_x, u_y, yaw_mder

    # neural net
    model_vehicle_dynamics(f=torch_model_path, steering=0.5, throttle=0.6, time_horizon=10, init_cond=initial_conditions,
                           neural_net=True, linear_varying_ctrls=False)

    # scipy odeint
    model_vehicle_dynamics(steering=0.5, throttle=0.6, time_horizon=10, init_cond=initial_conditions, neural_net=False)

