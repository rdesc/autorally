"""Trains neural network for dynamics model
"""
import gc
import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


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


def make_data_loader(indices, input_cols=None, label_cols=None):
    """
    Sets up data loader
    :param indices: array or list of indices to index the data
    :param input_cols: list of column names for inputs
    :param label_cols: list of column names for labels
    :return: data loader
    """
    df = pd.read_csv("trajectory_files/st=0.5_thr=0.65/nn_state_variables.csv").loc[indices]  # FIXME: temporary training data
    inputs = df[["roll", "u_x", "u_y", "yaw_mder", "steering", "throttle"]] if input_cols is None else df[input_cols]
    labels = df[["roll_der", "u_x_der", "u_y_der", "yaw_mder_der"]] if label_cols is None else df[label_cols]

    dataset = VehicleDynamicsDataset(inputs.to_numpy(), labels.to_numpy())  # convert to numpy arrays
    return DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1, drop_last=False)


def val_phase(model, data_loader):
    """
    Get validation loss
    :param model: model during training phase
    :param data_loader: data loader for validation phase
    :return: validation loss
    """
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
        val_loss = (preds - labels)**2

        # back to training
        model.train()

        return val_loss


def train(train_loader, validation_loader, args):
    """
    Train model
    :param train_loader: data loader for training
    :param validation_loader:
    :param args: args containing training hyperparameters
    """
    # set up model
    model = setup_model()  # default layers and activation
    # set up optimizer
    opt = optim.Adam(model.parameters(), lr=args['lr'])

    best_val_loss = np.inf

    for epoch in range(args['epochs']):
        preds = []
        labels = []
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
        tr_loss = (preds - labels)**2
        # calculate the gradients
        tr_loss.sum().backward()
        # update all the parameters based on the gradients calculated
        opt.step()

        # get validation loss
        val_loss = val_phase(model, validation_loader)

        # update best model
        val_loss = val_loss.sum().item()
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(args["results_dir"], "model"))
            best_val_loss = val_loss

        print("Epoch %i, Train loss is %d, Validation loss is %d" % (epoch, tr_loss.sum().item(), val_loss))
        gc.collect()

    print("Best val loss is %d" % best_val_loss)


# TODO: add step that starts recording rosbag data
if __name__ == '__main__':
    # load config file into args
    config = "./config.yml"
    with open(config, "r") as yaml_file:
        args = yaml.load(yaml_file)

    # setup directory to save models
    if not os.path.exists(args["results_dir"]):
        os.makedirs(args["results_dir"])

    device = torch.device('cpu')
    # check if cuda enabled gpu is available
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # generate indices for training and validation
    dataset_size = 250
    a = np.arange(dataset_size)
    tr_ind, val_ind = train_test_split(a, train_size=0.8, test_size=0.2, shuffle=True)
    # init training data loader
    tr_loader = make_data_loader(indices=tr_ind)
    # init validation data loader
    val_loader = make_data_loader(indices=val_ind)
    # start training
    train(tr_loader, val_loader, args)
