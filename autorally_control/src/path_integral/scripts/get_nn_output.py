import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class AutoRallyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]


def make_data_loader():
    data = np.full((2, 6), 0)
    # init all state variables to 0
    data[0] = [0., 0., 0., 0., 0., 1.]
    data[1] = [0., 0., 0., 0., 0., 2.]
    dataset = AutoRallyDataset(data)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)


def load_model(f):
    # setup model architecture
    model = nn.Sequential(nn.Linear(6, 32),
                          nn.Tanh(),
                          nn.Linear(32, 32),
                          nn.Tanh(),
                          nn.Linear(32, 4))
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
def generate_output(f):
    # get a data loader
    data_loader = make_data_loader()
    # load model with pre-trained weights
    model = load_model(f)
    # init variable to store neural network outputs
    output = np.zeros(10)
    with torch.no_grad():
        model.eval()

        for idx, samples in enumerate(data_loader):
            print(samples)
            y_pred = model(samples.float().to(device))
            print(y_pred)
            y_pred = y_pred.detach().cpu().numpy()
            print(y_pred)


if __name__ == '__main__':
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    file_paths = ["../params/models/autorally_nnet_09_12_2018.npz"]  # ,"../params/models/gazebo_nnet_09_12_2018.npz"]
    for f in file_paths:
        generate_output(f)
