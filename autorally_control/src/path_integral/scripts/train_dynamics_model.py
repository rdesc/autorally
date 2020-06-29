"""Trains neural network for dynamics model
"""
import gc
import os
import yaml
from datetime import datetime
from shutil import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model_vehicle_dynamics import compute_state_ders, state_variable_plots

torch.manual_seed(0)


class VehicleDynamicsDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def setup_model(layers=None, activation=nn.Tanh()):
    """
    Sets up a simple feed forward neural network
    :param layers: A list specifying the number of nodes for each layer (includes input and output layer)
    :param activation: The activation function to apply after each layer (non-linearity), default is tanh
    :return: torch model loaded on gpu
    """
    # if no layers specified, set default to [6, 32, 32, 4]
    if not layers:
        model = nn.Sequential(nn.Linear(6, 32),
                              activation,
                              nn.Linear(32, 32),
                              activation,
                              nn.Linear(32, 4))
    else:
        # initialize model
        model = nn.Sequential()
        for idx, layer in enumerate(layers):
            # skip last iteration
            if idx == len(layers) - 1:
                continue
            model.add_module("nn" + str(idx), nn.Linear(layers[idx], layers[idx+1]))
            # dont add activation to final layer
            if idx != len(layers) - 2:
                model.add_module("act" + str(idx), activation)

    print(model)
    # load model onto GPU
    model.to(device)

    return model


def make_data_loader(indices, batch_size=10, input_cols=None, label_cols=None):
    """
    Sets up data loader
    """
    df = pd.read_csv("trajectory_files/st=0.5_thr=0.65/nn_state_variables.csv").loc[indices]  # FIXME: temporary training data
    inputs = df[["roll", "u_x", "u_y", "yaw_mder", "steering", "throttle"]] if input_cols is None else df[input_cols]
    labels = df[["roll_der", "u_x_der", "u_y_der", "yaw_mder_der"]] if label_cols is None else df[label_cols]

    dataset = VehicleDynamicsDataset(inputs.to_numpy(), labels.to_numpy())  # convert to numpy arrays
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)


# TODO: change to x and y and not x_pos and y_pos
def make_test_data_loader(data_path=None):
    """
    Sets up test data loader
    """
    df = pd.read_csv("trajectory_files/st=0.5_thr=0.65/nn_state_variables.csv")
    inputs = df[["roll", "u_x", "u_y", "yaw_mder", "steering", "throttle"]]
    labels = df[["roll_der", "u_x_der", "u_y_der", "yaw_mder_der"]]

    dataset = VehicleDynamicsDataset(inputs.to_numpy(), labels.to_numpy())  # convert to numpy arrays
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)


# TODO: generate predictions, then use code from model_vehicle_dynamics.py to compare nn trajectories to truth trajectories
# TODO: color coding for forward, reverse direction
# this data will be time ordered
def generate_predictions(data_path, nn_layers, state_dim=7):
    print("Generating predictions from trained model...")

    # FIXME: how to get proper timestep
    time_step = 0.05

    # init df
    df_nn = pd.read_csv(data_path)
    preds = []
    labels = []
    # init state variables
    state = np.full((len(df_nn), state_dim), 0, np.float)  # x_pos, y_pos, yaw, roll, u_x, u_y, yaw_mder
    cols = ["x_pos", "y_pos", "yaw", "roll", "u_x", "u_y", "yaw_mder"]

    # setup data loader
    test_loader = make_test_data_loader(data_path)

    # load model from disk
    model = setup_model(nn_layers)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))

    with torch.no_grad():
        # set model to eval mode
        model.eval()

        idx = 0
        for sample, label in test_loader:
            # get the current state
            curr_state = state[idx]

            # get predictions of neural network
            pred = model(sample.float().to(device))
            # save on cpu
            pred = pred.detach().cpu()
            # append to lists
            preds.append(pred)
            labels.append(label)

            # compute the state derivatives
            state_der = compute_state_ders(curr_state, pred)

            # FIXME: update states
            state[idx + 1] = state[idx] + state_der * time_step

            # increase index
            idx += 1

    # make tensors for predictions and labels to calculate loss
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0).float()
    loss_func = torch.nn.MSELoss()  # regression mean squared loss
    # compute loss
    test_loss = loss_func(preds, labels)

    print("Test loss: %0.5f" % test_loss.item())

    # save loss to file
    f = open(model_dir + "/loss.txt", "w+")
    f.write(str(test_loss.item()))

    # load nn state data into data frame
    for idx, col in enumerate(cols):
        df_nn[col] = state[idx]

    # load ground truth data
    df_truth = pd.read_csv("dynamics_model/ground_truthstate.csv")[["x", "y"]]
    df_truth.rename(columns={"x": "x_pos", "y": "y_pos"})

    # TODO: plot title and cols to exclude
    # plot trajectories and state variables
    state_variable_plots(df_truth, df_nn, truth_label="ground_truth", dir_path=model_dir, plt_title="")


def val_phase(model, data_loader):
    """
    Get validation loss
    """
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    with torch.no_grad():
        # set to eval mode
        model.eval()
        preds = []
        labels = []
        for sample, label in data_loader:
            # get predictions
            pred = model(sample.float().to(device))
            preds.append(pred.cpu())
            labels.append(label.unsqueeze(1))

        # check to see if epoch contains multiple batches
        if len(preds) > 1:
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0).float()
        else:
            preds = preds[0]
            labels = labels[0].float()

        # calculate the L2 loss
        val_loss = loss_func(preds, labels)

        # back to training
        model.train()

        return val_loss


def train(train_loader, validation_loader, nn_layers, epochs, lr):
    """
    Train model
    """
    # set up model
    model = setup_model(nn_layers)  # default layers and activation
    # set up optimizer
    opt = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = np.inf
    loss_func = torch.nn.MSELoss()  # mean squared loss

    for epoch in range(epochs):
        preds = []
        labels = []
        # clear gradients
        opt.zero_grad()

        for sample, label in train_loader:
            # number of samples to load at each iteration is determined by the batch size set in the data loader
            # get predictions
            pred = model(sample.float().to(device))
            preds.append(pred.cpu())
            labels.append(label.unsqueeze(1))

        # check to see if epoch contains multiple batches
        if len(preds) > 1:
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0).float()
        else:
            preds = preds[0]
            labels = labels[0].float()

        # calculate the L2 loss TODO: try L1 loss maybe?
        tr_loss = loss_func(preds, labels)
        # calculate the gradients
        tr_loss.backward()
        # update all the parameters based on the gradients calculated
        opt.step()

        # get validation loss
        val_loss = val_phase(model, validation_loader)

        # update best model
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
            best_val_loss = val_loss

        print("Epoch %i, Train loss is %0.5f, Validation loss is %0.5f" % (epoch, tr_loss.item(), val_loss.item()))
        gc.collect()

    print("Best val loss is %0.5f" % best_val_loss)


# TODO: add step that starts recording rosbag data
# TODO: stop using global vars to reduce coupling
if __name__ == '__main__':
    # load config file into args
    config = "./config.yml"
    with open(config, "r") as yaml_file:
        args = yaml.load(yaml_file)

    # Append time/date to directory name
    creation_time = str(datetime.now().strftime('%m-%d_%H:%M/'))
    model_dir = args["results_dir"] + creation_time

    # setup directory to save models
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # save training details
    copy("config.yml", model_dir)

    device = torch.device('cpu')
    # check if cuda enabled gpu is available
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # generate indices for training and validation
    a = np.arange(args["dataset_size"])
    tr_ind, val_ind = train_test_split(a, train_size=0.8, test_size=0.2, shuffle=True)
    # init training data loader
    tr_loader = make_data_loader(indices=tr_ind, batch_size=args["batch_size"])
    # init validation data loader
    val_loader = make_data_loader(indices=val_ind, batch_size=args["batch_size"])

    # start training
    train(tr_loader, val_loader, args["nn_layers"], args["epochs"], args["lr"])

    # test phase
    # generate_predictions("", args["nn_layers"], args["state_dim"])
