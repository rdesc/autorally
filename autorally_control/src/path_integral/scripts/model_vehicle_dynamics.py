import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
from scipy.integrate import odeint


# TODO: create requirements.txt file
class AutoRallyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]


def make_data_loader(data):
    dataset = AutoRallyDataset(data)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)


def load_model(f, from_npz=False):
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


def vehicle_dynamics_sys_ode(w, t, p):
    """
    Defines the first order differential equations for the vehicle model equations of motion
    :param w: vector of the state variables: w = [x1, x2, x3, x4, x5, x6]
    :param t: time
    :param p: vector of the parameters: p = [st, thr, a1, a2]
    """
    x1, x2, x3, x4, x5, x6 = w  # head_rate, lat_vel, long_vel, x, y, yaw
    st, thr, a1, a2 = p
    # create f = (x1', x3', x4', x5', x6'):
    f = [a1 * st, 0, a2 * thr, x3 * np.cos(x6), x3 * np.sin(x6), x1]
    return f


def model_vehicle_dynamics(f, steering, throttle, time_horizon, time_step=0.01, input_dim=6, init_cond=None, forward_euler=True, linear_varying_ctrls=True):
    # define number of steps for integration
    num_steps = int(time_horizon / time_step + 1)
    prefix = "odeint_" if not forward_euler else "nn_"
    suffix = "thr=" + str(throttle) + "_st=" + str(steering)
    # create directory to store trajectory files
    dir_path = "trajectory_files/" + prefix + suffix + "/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # initial conditions
    if init_cond is None:
        head_rate_0 = 0
        lat_vel_0 = 0
        long_vel_0 = 0
        x_0 = 0
        y_0 = 0
        yaw_0 = 0
    else:
        head_rate_0, lat_vel_0, long_vel_0, x_0, y_0, yaw_0 = init_cond

    plt_title = ("throttle=%s, steering=%s, time_horizon=%s\nhead_rate_0=%s, lat_vel_0=%s, long_vel_0=%s, x_0=%s, y_0=%s, yaw_0=%s" %
                 (throttle, steering, time_horizon, head_rate_0, lat_vel_0, long_vel_0, x_0, y_0, yaw_0))

    if not forward_euler:
        # integrate using scipys odeint
        # constant to multiply steering by
        a1 = 1
        # constant to multiply throttle by
        a2 = 1

        # ODE solver parameters
        abserr = 1.0e-8
        relerr = 1.0e-6
        t = np.linspace(0, time_horizon, num_steps)

        # init state variables and param vectors
        w0 = [head_rate_0, lat_vel_0, long_vel_0, x_0, y_0, yaw_0]
        p = [steering, throttle, a1, a2]
        # call the ODE solver
        wsol = odeint(vehicle_dynamics_sys_ode, w0, t, args=(p,), atol=abserr, rtol=relerr)

        # convert to pandas DataFrames
        data_df = pd.DataFrame(data=np.concatenate((np.reshape(t, (len(t), 1)), wsol), axis=1),
                               columns=["time", "head_rate", "lat_vel", "long_vel", "x", "y", "yaw"])
        # save to disk
        data_df.to_csv(dir_path + "odeint_state_variables.csv", index=False, header=True)
        # plot trajectories
        state_variable_plots(data_df, dir_path, plt_title=plt_title, cols_to_exclude=["time"])
        return

    ############################
    # do forward euler
    # init state variable data
    data = np.full((num_steps, input_dim), 0, np.float)
    # set initial conditions
    data[0] = [0, long_vel_0, lat_vel_0, head_rate_0, 0, 0]  # [roll, long_vel, lat_vel, head_rate, steering, throttle]

    if linear_varying_ctrls:
        # apply linearly increasing throttle and steering
        steering_array = np.linspace(0, steering, num_steps)
        throttle_array = np.linspace(0, throttle, num_steps)
        data[:, 4] = steering_array
        data[:, 5] = throttle_array
    else:
        data[:, 4] = steering
        data[:, 5] = throttle

    # init array to store state variables yaw, x and y positions w.r.t fixed ref, and x_dot, and y_dot
    pos_yaw_vars = np.full((num_steps, 5), 0, np.float)
    # set initial conditions
    pos_yaw_vars[0] = [yaw_0, x_0, y_0, 0, 0]  # [yaw, x, y, x_dot, y_dot]
    # init array to store outputs of neural network
    nn_output = np.full((num_steps, 4), 0, np.float)  # d/dt[roll, long_vel, lat_vel, head_rate]
    # init first data loader
    data_loader = make_data_loader(data[0:1])
    # load model
    model = load_model(f)
    with torch.no_grad():
        model.eval()

        # iterate through each step of trajectory
        for idx in tqdm(range(num_steps - 1)):
            for _, sample in enumerate(data_loader):
                # get output of neural network
                y_pred = model(sample.float().to(device))
                y_pred = y_pred.detach().cpu().numpy()[0]
                # update state variables
                long_vel = data[idx][1] + y_pred[1] * time_step
                lat_vel = data[idx][2] + y_pred[2] * time_step
                head_rate = data[idx][3] + y_pred[3] * time_step
                yaw = pos_yaw_vars[idx][0] + head_rate * time_step
                x_dot = -1. * lat_vel * np.sin(yaw) + long_vel * np.cos(yaw)
                y_dot = lat_vel * np.cos(yaw) + long_vel * np.sin(yaw)
                x = pos_yaw_vars[idx][1] + x_dot * time_step
                y = pos_yaw_vars[idx][2] + y_dot * time_step
                # store in arrays, ignore updating roll state variable
                data[idx + 1] = [y_pred[0], long_vel, lat_vel, head_rate, data[idx + 1][4], data[idx + 1][5]]
                pos_yaw_vars[idx + 1] = [yaw, x, y, x_dot, y_dot]
                nn_output[idx + 1] = y_pred

            # get new data loader with updated data
            data_loader = make_data_loader(data[idx + 1:idx + 2])

    # convert numpy arrays to pandas DataFrames
    time = np.linspace(0, time_horizon, num_steps)
    data_df = pd.DataFrame(data=np.concatenate((data, nn_output, pos_yaw_vars, np.reshape(time, (len(time), 1))), axis=1),
                           columns=["roll", "long_vel", "lat_vel", "head_rate", "steering", "throttle", "der_roll",
                                    "der_long_vel", "der_lat_vel", "der_head_rate", "yaw", "x", "y", "x_dot", "y_dot", "time"])
    # match column ordering to odeint method
    cols = ["head_rate", "lat_vel", "long_vel", "x", "y", "yaw", "time", "roll", "steering", "throttle", "der_roll",
            "der_long_vel", "der_lat_vel", "der_head_rate", "x_dot", "y_dot"]
    data_df = data_df.reindex(columns=cols)
    # save to disk
    data_df.to_csv(dir_path + "nn_state_variables.csv", index=False, header=True)
    # plot trajectories
    state_variable_plots(data_df, dir_path, plt_title=plt_title, cols_to_exclude=cols[6:])


def state_variable_plots(df, dir_path, plt_title="", cols_to_exclude=None):
    # plot trajectory in fixed global frame
    fig = plt.figure()
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("2D trajectory\n" + plt_title)
    plt.plot(df['x'], df['y'])
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
    # TODO: add parser args
    # TODO: add docs
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    torch_model_path = "../params/models/torch_model_autorally_nnet.pt"
    model_from_npz = False
    if model_from_npz:
        file_path = "../params/models/autorally_nnet_09_12_2018.npz"
        model = load_model(file_path, from_npz=True)
        torch.save(model.state_dict(), torch_model_path)
        del model

    # control constraints to match path_integral_nn.launch
    # throttle range [-0.99, 0.65]
    # steering range [0.99, -0.99]
    initial_conditions = [0, 0, 0, 0, 0, 0]  # head_rate, lat_vel, long_vel, x, y, yaw
    model_vehicle_dynamics(torch_model_path, steering=0.0, throttle=0.5, time_horizon=5, init_cond=initial_conditions,
                           forward_euler=False, linear_varying_ctrls=False)
