import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
from scipy.integrate import solve_ivp
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
            model[i*2].bias = nn.Parameter(torch.from_numpy(layers[bias_names[i]]).float(), requires_grad=False)
            model[i*2].weight = nn.Parameter(torch.from_numpy(layers[layer_names[i]]).float(), requires_grad=False)

    # load model onto GPU
    model.to(device)

    return model


# nn input: roll, longitudinal velocity, lateral velocity, heading rate (state variables) and steering + throttle
# nn output: time derivative of state variables
def generate_output(f, steering, throttle, time_horizon, time_step=0.01, input_dim=6,
                    forward_euler=True, linear_varying_ctrls=True, save_states=False):
    if not forward_euler:
        c_st = 1./time_horizon*steering*30*np.pi/180
        c_thr = 1./time_horizon*throttle*8
        # yaw_sol = solve_ivp(lambda t, y: t/time_horizon*steering*30*np.pi/180, t_span=[0, time_horizon], y0=[0])
        x_sol = solve_ivp(lambda t, y: c_thr*t**2/2*np.cos(c_st*t**2/2), t_span=[0, time_horizon], y0=[0])
        return NotImplementedError

    num_steps = int(time_horizon / time_step + 1)
    # init data
    data = np.full((num_steps, input_dim), 0, np.float)
    # init state variables to 0
    data[0] = [0, 0, 0, 0, 0, 0]
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
    pos_yaw_vars[0] = [0, 0, 0, 0, 0]  # [yaw, x, y, x_dot, y_dot] initial yaw ~ 3*pi/4 set in path_integral_nn
    # init array to store outputs of neural network
    nn_output = np.full((num_steps, 4), 0, np.float)
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
                y_dot = lat_vel * np.cos(yaw) + long_vel * np.cos(yaw)
                x = pos_yaw_vars[idx][1] + x_dot * time_step
                y = pos_yaw_vars[idx][2] + y_dot * time_step
                # store in arrays, ignore updating roll state variable
                data[idx + 1] = [y_pred[0], long_vel, lat_vel, head_rate, data[idx + 1][4], data[idx + 1][5]]
                pos_yaw_vars[idx + 1] = [yaw, x, y, x_dot, y_dot]
                nn_output[idx + 1] = y_pred

            # get new data loader with updated data
            data_loader = make_data_loader(data[idx+1:idx+2])

    # convert numpy arrays to pandas DataFrames
    time = np.linspace(0, time_horizon, num_steps)
    data_df = pd.DataFrame(data=np.concatenate((data, nn_output, pos_yaw_vars, np.reshape(time, (len(time), 1))), axis=1),
                           columns=["roll", "long_vel", "lat_vel", "head_rate", "steering", "throttle",
                                    "der_roll", "der_long_vel", "der_lat_vel", "der_head_rate",
                                    "yaw", "x", "y", "x_dot", "y_dot", "time"])

    suffix = "thr=" + str(throttle) + "_st=" + str(steering)
    # create directory to store trajectory files
    dir_path = "../params/models/trajectory_files/" + suffix + "/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if save_states:
        data_df.to_csv(dir_path + "state_variables.csv", index=False, header=True)

    # title = "2D trajectory\n" + "throttle=" + str(throttle) + ", steering=" + str(steering) + "\n" \
    #        "time_horizon=" + str(time_horizon) + ", time_step=" + str(time_step)
    file_name = "2d_traj"
    plot_trajectory(data_df, file_name, dir_path)


def plot_trajectory(df, file_name, dir_path):
    # plot all state variables along a common time axis
    fig = plt.figure(figsize=(10,20))
    count_states = len(df.columns) - 3
    idx = 1
    for _, column in enumerate(df.columns):
        # skip time and control inputs
        if column in ["time", "steering", "throttle"]:
            continue
        ax = fig.add_subplot(count_states, 1, idx)
        ax.set_ylabel(column)
        plt.plot(df["time"], df[column])
        # plt.grid(True, which='both', axis='both')
        if not(idx == count_states):
            ax.set_xticklabels([])
        idx += 1
    plt.xlabel("time")
    plt.suptitle("states vs. time")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(dir_path + file_name + "_states_vs_time.png", dpi=600)

    # plot control inputs along a time axis
    fig = plt.figure()
    for idx, column in enumerate(["steering", "throttle"]):
        ax = fig.add_subplot(2, 1, idx+1)
        ax.set_ylabel(column)
        plt.plot(df["time"], df[column])
    plt.xlabel("time")
    plt.suptitle("controls vs. time")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(dir_path + file_name + "_ctrl_vs_time.png", dpi=300)

    # plot state vs state
    # TODO: not sure which state vs state is interesting, missing roll, yaw
    pd.plotting.scatter_matrix(df.drop(["time", "steering", "throttle"], axis=1), alpha=0.8, figsize=(30, 30), diagonal='hist')
    plt.suptitle("state vs. state pair plot")
    plt.savefig(dir_path + file_name + "_state_vs_state.png", dpi=300)
    #pd.plotting.scatter_matrix(df[["x", "y", "x_dot", "y_dot"]], alpha=0.8, figsize=(10, 10), diagonal='hist')
    #plt.suptitle("state vs. state pair plot")
    #plt.savefig(dir_path + file_name + "_state_vs_state_2.png", dpi=300)


if __name__ == '__main__':
    # TODO: add parser args
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
    generate_output(torch_model_path, steering=0.0, throttle=0.65, time_horizon=2.5, linear_varying_ctrls=False, save_states=True)
