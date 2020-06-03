import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class AutoRallyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]


# TODO: what are domains of the inputs
# steering and throttle [-1, 1]
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


# input: roll, longitudinal velocity, lateral velocity, heading rate (state variables) and steering + throttle
# output: time derivative of state variables
def generate_output(f, steering=0.0, throttle=0.9, time_horizon=2.5, time_step=0.01, input_dim=6):
    # init data
    data = np.full((int(time_horizon / time_step + 1), input_dim), 0, np.float)
    # init state variables to 0 and apply throttle
    data[0] = [0, 0, 0, 0, steering, throttle]
    # init array to store positions
    positions = np.full((int(time_horizon / time_step + 1), 2), 0, np.float)
    positions[0] = [0, 0]
    # init first data loader
    data_loader = make_data_loader(data[0:1])
    # load model
    model = load_model(f)
    with torch.no_grad():
        model.eval()

        # iterate through each step of trajectory
        for idx in range(int(time_horizon / time_step)):
            print("index %s" % idx)
            for _, sample in enumerate(data_loader):
                print(sample)
                # get output of neural network
                y_pred = model(sample.float().to(device))
                y_pred = y_pred.detach().cpu().numpy()[0]
                print(y_pred)
                # v = v0 + at
                data[idx+1] = [y_pred[0], data[idx][1] + y_pred[1] * time_step, data[idx][2] + y_pred[2] * time_step, y_pred[3], data[idx][4], data[idx][5]]
                # x = x0 + vt + at^2/2
                positions[idx + 1] = [positions[idx][0] + data[idx][1] * time_step + y_pred[1] * time_step ** 2 / 2,
                                      positions[idx][1] + data[idx][2] * time_step + y_pred[2] * time_step ** 2 / 2]

            # get new data loader with updated data
            data_loader = make_data_loader(data[idx+1:idx+2])

    #np.save("../params/models/positions.npy", positions)
    title = "2D trajectory\n" \
            "throttle=" + str(throttle) + ", steering=" + str(steering)+ "\n" \
            "time_horizon=" + str(time_horizon)+ ", time_step=" + str(time_step)
    plot_trajectory(positions, title)


def plot_trajectory(data, title):
    fig = plt.figure()
    plt.ylim(data.min(), data.max())
    plt.ylabel("y position")
    plt.xlim(data.min(), data.max())
    plt.xlabel("x position")
    plt.title(title)
    plt.plot(data[:,0], data[:,1])
    plt.show()


if __name__ == '__main__':
    #TODO: add parser args
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

    generate_output(torch_model_path, steering=0.0, throttle=0.99)
