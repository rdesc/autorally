"""Torch dataset classes"""
from torch.utils.data import Dataset


class VehicleDynamicsDataset(Dataset):
    """Dataset class for training and validation phase"""
    def __init__(self, inputs, labels, input_cols=None, label_cols=None):
        self.inputs = inputs
        self.labels = labels
        self.input_cols = input_cols
        self.label_cols = label_cols

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class TestDataset(Dataset):
    """Dataset class for test phase"""
    def __init__(self, states, state_cols, state_ders, state_der_cols, ctrl_data, ctrl_cols, time_data, time_col):
        self.states = states
        self.state_cols = state_cols
        self.state_ders = state_ders
        self.state_der_cols = state_der_cols
        self.ctrls = ctrl_data
        self.ctrl_cols = ctrl_cols
        self.time = time_data
        self.time_col = time_col

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.state_ders[idx], self.ctrls[idx], self.time[idx]
