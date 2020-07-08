"""Trains and evaluates neural network for dynamics model # TODO: docs
"""
import gc
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


class TestDataset(Dataset):
    def __init__(self, states, state_cols, ctrl_data, ctrl_cols, time_data, time_col):
        self.states = states
        self.state_cols = state_cols
        self.ctrls = ctrl_data
        self.ctrl_cols = ctrl_cols
        self.time = time_data
        self.time_col = time_col

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.ctrls[idx], self.time[idx]


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


def make_test_data_loader(data_path, batch_size, state_cols, ctrl_cols, indices=None, time_col='time'):
    df = pd.read_csv(data_path) if indices is None else pd.read_csv(data_path).loc[indices]
    states = df[state_cols]
    ctrl_data = df[ctrl_cols]
    time_data = df[time_col]

    dataset = TestDataset(states.to_numpy(), state_cols, ctrl_data.to_numpy(), ctrl_cols, time_data.to_numpy(), time_col)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)  # data needs to be time ordered


# TODO: color coding for forward, reverse direction
# rename feature and label cols
# TODO: calculate drift in x and y in the end
def generate_predictions(device, data_path, nn_layers, model_dir, state_cols, ctrl_cols, time_col='time', time_horizon=2.5, state_dim=7):
    print("\nGenerating predictions from trained model...")

    # get time step from data
    time_step = pd.read_csv(data_path).head(2)[time_col].values[1]
    print("time step: %.04f" % time_step)

    # determine batch size
    batch_size = int(np.ceil(time_horizon / time_step))
    print("batch size: %.0f" % batch_size)

    # setup data loader
    data_loader = make_test_data_loader(data_path, batch_size, state_cols, ctrl_cols, indices=np.arange(300), time_col=time_col)

    # number of batches to do
    total_batches = data_loader.dataset.__len__() // batch_size

    # load model
    model = setup_model(device, nn_layers)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"))["model_state_dict"])

    with torch.no_grad():
        # set model to eval mode
        model.eval()

        # keep track of current batch number
        batch_num = 0

        # generate a trajectory for each batch
        for truth_states, ctrls, time_data in data_loader:
            print('\nBatch %i/%i' % (batch_num, total_batches))
            print('-' * 10)
            batch_num += 1
            num_steps = truth_states.size(0)
            # skip last batch if it is less than batch size
            if num_steps < batch_size:
                print("Skipping final batch...")
                continue

            # init state variables
            nn_states = np.full((num_steps, state_dim), 0, np.float)
            # set initial conditions
            nn_states[0] = truth_states[0].cpu().numpy()
            # array to store all state derivatives
            state_ders = np.full((num_steps, state_dim), 0, np.float)

            # iterate through each step of trajectory
            # FIXME: loop is hardcoded to specific problem
            for idx in range(num_steps - 1):
                # prep input to feed to neural network
                x = torch.tensor(
                    [nn_states[idx][3], nn_states[idx][4], nn_states[idx][5], nn_states[idx][6], ctrls[idx][0], ctrls[idx][1]])

                # get the current state
                curr_state = nn_states[idx]

                # get output of neural network
                output = model(x.float().to(device))
                # convert to numpy array
                output = output.cpu().numpy()

                # compute the state derivatives
                state_der = compute_state_ders(curr_state, output)

                # update states
                nn_states[idx + 1] = curr_state + state_der * time_step

                # save state derivatives
                state_ders[idx] = state_der

            # TODO: print some stats
            # TODO: make dir for each batch

            # convert time data to numpy
            time_data = time_data.cpu().numpy()
            # convert control data to numpy
            ctrls = ctrls.cpu().numpy()
            # make der cols
            state_der_cols = [col + '_der' for col in state_cols]
            # TODO: add nn ders
            # create pandas DataFrames
            df_nn = pd.DataFrame(data=np.concatenate((np.reshape(time_data, (len(time_data), 1)), nn_states, ctrls), axis=1),
                                 columns=np.concatenate(([time_col], state_cols, ctrl_cols)))
            df_nn.to_csv(model_dir + "nn_state_variables.csv", index=False, header=True)

            # TODO: add truth ders
            # load ground truth data
            df_truth = pd.DataFrame(data=np.concatenate((np.reshape(time_data, (len(time_data), 1)), truth_states, ctrls), axis=1),
                                    columns=np.concatenate(([time_col], state_cols, ctrl_cols)))
            df_truth.to_csv(model_dir + "truth_state_variables.csv", index=False, header=True)

            # plot trajectories
            # TODO: need tick labels for time
            state_variable_plots(df_truth, df_nn, truth_label="ground truth", dir_path=model_dir, plt_title='',
                                 cols_to_exclude=["time"], suffix=str(batch_num))

        # TODO: output the mean error in final trajectories


def train(device, train_loader, val_loader, nn_layers, epochs, lr, model_dir, criterion=torch.nn.L1Loss()):
    # get start time
    start = time.time()
    # set up model
    model = setup_model(device, nn_layers)  # default activation
    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # set up data loaders
    data_loaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": train_loader.dataset.__len__(), "val": val_loader.dataset.__len__()}
    # store train and val losses for plotting
    losses = {"train": [], "val": []}

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
                optimizer.zero_grad()

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
                        optimizer.step()

                    # updating stats
                    running_loss += loss.item()*inputs.size(0)  # multiply by batch size since calculated loss was the mean

            # calculate loss for epoch
            epoch_loss = running_loss/dataset_sizes[phase]
            losses[phase].append(epoch_loss)
            print('%s Loss: %.4f' % (phase, epoch_loss))

            # update new best val loss and model
            if phase == "val" and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_val_loss}, os.path.join(model_dir, "model.pt"))

        gc.collect()

    time_elapsed = time.time() - start
    print("\nTraining complete in %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))
    print("Best val loss %0.5f" % best_val_loss)

    # plot loss monitoring
    fig = plt.figure()
    x = np.arange(epochs)
    plt.plot(x, losses['train'], 'b-', x, losses['val'], 'r-')
    plt.title("Loss")
    plt.legend(['train', 'val'], loc='upper right')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    fig.savefig(os.path.join(model_dir, "loss.pdf"), format="pdf")
