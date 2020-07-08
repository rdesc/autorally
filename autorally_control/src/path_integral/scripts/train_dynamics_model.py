"""Trains and evaluates neural network for dynamics model # TODO: docs
"""
import gc
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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


def setup_model(device, layers=None, activation=nn.Tanh()):
    """
    Sets up a simple feed forward neural network
    :param layers: A list specifying the number of nodes for each layer (includes input and output layer)
    :param activation: The activation function to apply after each layer (non-linearity), default is tanh
    :return: torch model loaded on device
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
            model.add_module("nn" + str(idx), nn.Linear(layers[idx], layers[idx + 1]))
            # dont add activation to final layer
            if idx != len(layers) - 2:
                model.add_module("act" + str(idx), activation)

    print(model)
    # load model onto GPU
    model.to(device)

    return model


def make_data_loader(data_path, indices, batch_size=32, feature_cols=None, label_cols=None):
    df = pd.read_csv(data_path).loc[indices]
    inputs = df if feature_cols is None else df[feature_cols]
    labels = df if label_cols is None else df[label_cols]

    dataset = VehicleDynamicsDataset(inputs.to_numpy(), labels.to_numpy())  # convert to numpy arrays
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)


def make_test_data_loader(data_path, feature_cols=None, label_cols=None):
    df = pd.read_csv(data_path)
    inputs = df if feature_cols is None else df[feature_cols]
    labels = df if label_cols is None else df[label_cols]

    dataset = VehicleDynamicsDataset(inputs.to_numpy(), labels.to_numpy())  # convert to numpy arrays
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)


# TODO: generate predictions, then use code from model_vehicle_dynamics.py to compare nn trajectories to truth trajectories
# TODO: color coding for forward, reverse direction
# this data will be time ordered
def generate_predictions(device, data_path, nn_layers, model_dir, state_dim=7):
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


def train(device, train_loader, val_loader, nn_layers, epochs, lr, model_dir, criterion=torch.nn.L1Loss()):
    # get start time
    start = time.time()
    # set up model
    model = setup_model(device, nn_layers)  # default activation
    # set up optimizer
    opt = optim.Adam(model.parameters(), lr=lr)
    # set up data loaders
    data_loaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": train_loader.dataset.__len__(), "val": val_loader.dataset.__len__()}

    best_val_loss = np.inf

    for epoch in range(epochs):
        print('\nEpoch %i/%i' % (epoch, epochs - 1))
        print('-' * 10)
        for phase in ["train", "val"]:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # clear gradients
                opt.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    # get output from model
                    outputs = model(inputs.float())
                    # calculate loss
                    loss = criterion(outputs, labels.float())

                    if phase == "train":
                        # calculate the gradients
                        loss.backward()
                        # update all the parameters based on the gradients calculated
                        opt.step()

                    # updating stats
                    running_loss += loss.item()*inputs.size(0)  # multiply by batch size since calculated loss was the mean

            # calculate loss for epoch
            epoch_loss = running_loss/dataset_sizes[phase]
            print('%s Loss: %.4f' % (phase, epoch_loss))

            # update new best val loss and model
            if phase == "val" and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'loss': best_val_loss}, os.path.join(model_dir, "model.pt"))

        gc.collect()

    time_elapsed = time.time() - start
    print("\nTraining complete in %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))
    print("Best val loss %0.5f" % best_val_loss)

