import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm


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
def generate_output(f, steering, throttle, time_horizon, time_step=0.01, input_dim=6, linear_varying_ctrls=True, save_states=False):
    # init data
    data = np.full((int(time_horizon / time_step + 1), input_dim), 0, np.float)
    # init state variables to 0
    data[0] = [0, 0, 0, 0, 0, 0]
    if linear_varying_ctrls:
        # apply linearly increasing throttle and steering
        steering_array = np.linspace(0, steering, int(time_horizon / time_step) + 1)
        throttle_array = np.linspace(0, throttle, int(time_horizon / time_step) + 1)
        data[:, 4] = steering_array
        data[:, 5] = throttle_array
    else:
        data[:, 4] = steering
        data[:, 5] = throttle
    # init array to store state variables yaw, x and y positions w.r.t fixed ref, and x_dot, and y_dot
    pos_yaw_vars = np.full((int(time_horizon / time_step + 1), 5), 0, np.float)
    pos_yaw_vars[0] = [0, 0, 0, 0, 0]  # [yaw, x, y, x_dot, y_dot]
    # init first data loader
    data_loader = make_data_loader(data[0:1])
    # load model
    model = load_model(f)
    with torch.no_grad():
        model.eval()

        # iterate through each step of trajectory
        for idx in tqdm(range(int(time_horizon / time_step))):
            for _, sample in enumerate(data_loader):
                # get output of neural network
                y_pred = model(sample.float().to(device))
                y_pred = y_pred.detach().cpu().numpy()[0]
                # update state variables
                long_vel = data[idx][1] + y_pred[1] * time_step
                lat_vel = data[idx][2] + y_pred[2] * time_step
                head_rate = data[idx][3] + y_pred[3] * time_step
                yaw = pos_yaw_vars[idx][0] + head_rate * time_step
                x_dot = -1*lat_vel * np.sin(yaw) + long_vel*np.cos(yaw)
                y_dot = lat_vel*np.cos(yaw) + long_vel*np.cos(yaw)
                x = pos_yaw_vars[idx][1] + x_dot * time_step
                y = pos_yaw_vars[idx][2] + y_dot * time_step
                # store in arrays, ignore roll variable, keep throttle and steering inputs fixed for now
                data[idx + 1] = [y_pred[0], long_vel, lat_vel, head_rate, data[idx + 1][4], data[idx + 1][5]]
                pos_yaw_vars[idx + 1] = [yaw, x, y, x_dot, y_dot]

            # get new data loader with updated data
            data_loader = make_data_loader(data[idx+1:idx+2])

    if save_states:
        suffix = "thr=" + str(throttle) + "_st=" + str(steering)
        np.save("../params/models/pos_yaw_vars_" + suffix + ".npy", pos_yaw_vars)
        np.save("../params/models/nn_state_variables_" + suffix + ".npy", data)

    title = "2D trajectory\n" + "throttle=" + str(throttle) + ", steering=" + str(steering) + "\n" \
            "time_horizon=" + str(time_horizon) + ", time_step=" + str(time_step)
    file_name = "2d_traj_thr=" + str(throttle) + "_st=" + str(steering) + ".png"
    plot_trajectory(pos_yaw_vars, title, file_name)


def plot_trajectory(data, title, file_name):
    fig = plt.figure()
    # plt.ylim(data[:, 1:3].min(), data[:, 1:3].max())
    plt.ylabel("y position")
    # plt.xlim(data[:, 1:3].min(), data[:, 1:3].max())
    plt.xlabel("x position")
    plt.title(title)
    plt.plot(data[:, 1], data[:, 2])
    # plt.show()
    plt.savefig(file_name)


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
    generate_output(torch_model_path, steering=0.99, throttle=0.65, time_horizon=8, linear_varying_ctrls=True)
