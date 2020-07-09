"""Utility functions for model training, validation and testing phase"""
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_dataset_classes import VehicleDynamicsDataset, TestDataset


def setup_model(device, layers=None, activation=nn.Tanh()):
    """
    Sets up a feed forward neural network model
    :param device: cuda device
    :param layers: List of integers to specify the architecture of nn
    :param activation: Activation function to apply after each layer (except the last)
    :return: model loaded on device
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
    # load model onto device
    model.to(device)

    return model


def make_data_loader(data_path, indices, batch_size=32, feature_cols=None, label_cols=None):
    """
    Data loader for training and validation phase
    :type data_path: str
    :type indices: list[int]|ndarray
    :type batch_size: int
    :type feature_cols: list[str]|None
    :type label_cols: list[str]|None
    """
    df = pd.read_csv(data_path).loc[indices]
    inputs = df if feature_cols is None else df[feature_cols]
    labels = df if label_cols is None else df[label_cols]

    dataset = VehicleDynamicsDataset(inputs.to_numpy(), labels.to_numpy())  # convert to numpy arrays
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)


def make_test_data_loader(data_path, batch_size, state_cols, ctrl_cols, indices=None, time_col='time'):
    """
    Data loader for test phase
    :type data_path: str
    :type batch_size: int
    :type state_cols: list[str]
    :type ctrl_cols: list[str]
    :type indices: list[int]|ndarray|None
    :type time_col: str
    """
    df = pd.read_csv(data_path) if indices is None else pd.read_csv(data_path).loc[indices]
    states = df[state_cols]
    ctrl_data = df[ctrl_cols]
    time_data = df[time_col]

    dataset = TestDataset(states.to_numpy(), state_cols, ctrl_data.to_numpy(), ctrl_cols, time_data.to_numpy(), time_col)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)  # data needs to be time ordered
