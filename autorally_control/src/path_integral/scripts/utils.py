"""Utility functions for model training, validation and testing phase"""
import os
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_dataset_classes import VehicleDynamicsDataset, TestDataset

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def setup_model(layers=None, activation=nn.Tanh(), verbose=True):
    """
    Sets up a feed forward neural network model
    :param layers: List of integers to specify the architecture of nn
    :param activation: Activation function to apply after each layer (except the last)
    :param verbose: If true, print model configuration
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

    if verbose:
        print(model)

    return model


def npz_to_torch_model(filename, model):
    """
    Loads weights and biases from npz file to a torch model
    :param filename: path of the npz file
    :param model: torch model
    """
    npz = np.load(filename)
    files = npz.files
    # load weights and biases into appropriate layers
    for f in files:
        idx = (int(f[-1]) - 1)*2  # assumes activation layers are between nn layers
        if '_W' in f:
            model[idx].weight = nn.Parameter(torch.from_numpy(npz[f]).double(), requires_grad=False)
        elif '_b' in f:
            model[idx].bias = nn.Parameter(torch.from_numpy(npz[f]).double(), requires_grad=False)

    return model


def torch_model_to_npz(model, model_dir):
    """
    Converts torch model to a npz file configured for mppi
    From MPPI wiki "Model parameters need to be saved as double precision floating point numbers in order for them to be read in correctly."
    :param model: torch model
    :param model_dir: path to save npz file
    """
    weight_name = "dynamics_W"
    w_idx = 1
    bias_name = "dynamics_b"
    b_idx = 1

    files = {}
    # iterate over each set of weights and biases
    for name, param in model.named_parameters():
        if 'weight' in name:
            files[weight_name + str(w_idx)] = param.cpu().detach().numpy()
            w_idx += 1
        elif 'bias' in name:
            files[bias_name + str(b_idx)] = param.cpu().detach().numpy()
            b_idx += 1

    np.savez(os.path.join(model_dir, 'model.npz'), **files)


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

    dataset = VehicleDynamicsDataset(inputs.to_numpy(), labels.to_numpy(), input_cols=feature_cols, label_cols=label_cols)  # convert to numpy arrays
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)


def make_test_data_loader(data_path, batch_size, state_cols, state_der_cols, ctrl_cols, indices=None, time_col='time'):
    """
    Data loader for test phase
    :type data_path: str
    :type batch_size: int
    :type state_cols: list[str]
    :type state_der_cols: list[str]
    :type ctrl_cols: list[str]
    :type indices: list[int]|ndarray|None
    :type time_col: str
    """
    df = pd.read_csv(data_path) if indices is None else pd.read_csv(data_path).loc[indices]
    states = df[state_cols]
    state_ders = df[state_der_cols]
    ctrl_data = df[ctrl_cols]
    time_data = df[time_col]

    dataset = TestDataset(states.to_numpy(), state_cols, state_ders.to_numpy(), state_der_cols, ctrl_data.to_numpy(), ctrl_cols,
                          time_data.to_numpy(), time_col)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)  # data needs to be time ordered


def compute_state_ders(curr_state, y_pred, negate_yaw_der=True):
    """
    Takes in the current state and the output of the model to generate state time derivatives
    :param curr_state: the current state of the model
    :param y_pred: the neural network predictions of the model dynamics
    :param negate_yaw_der: in original mppi implementation yaw derivative needs to be negated
    :return: array of the state derivatives
    """
    # init array for state derivatives
    state_der = np.zeros(len(curr_state))  # d/dt [x_pos, y_pos, yaw, roll, u_x, u_y, yaw_mder]
    # compute kinematics (match implementation with NeuralNetModel::computeKinematics in neural_net_model.cu)
    state_der[0] = np.cos(curr_state[2]) * curr_state[4] - np.sin(curr_state[2]) * curr_state[5]
    state_der[1] = np.sin(curr_state[2]) * curr_state[4] + np.cos(curr_state[2]) * curr_state[5]
    state_der[2] = curr_state[6]
    if negate_yaw_der:
        state_der[2] *= -1  # from mppi comments 'pose estimate actually gives the negative yaw derivative'

    # compute dynamics
    state_der[3], state_der[4], state_der[5], state_der[6] = y_pred

    return state_der


def state_variable_plots(df1, df1_label="ode", df2=None, df2_label="nn", dir_path="", plt_title="",
                         cols_to_include="all", time_col="time", suffix=""):
    """
    Outputs trajectory plot and state vs. time plots
    :param df1: state and control data
    :param df1_label: label of df1 e.g. "ground truth"
    :param df2: secondary state and control data
    :param df2_label: label of df2 e.g. "neural network"
    :param dir_path: path to store plots
    :param plt_title: title of plots
    :type cols_to_include: list[str] or "all"
    :param time_col: name of time column
    :param suffix: string to append to plots in case multiple calls are made with same dir_path
    """
    # plot trajectory in fixed global frame
    fig = plt.figure(figsize=(8, 6))
    plt.xlabel("x position (m)")
    plt.axis('equal')
    plt.ylabel("y position (m)")
    plt.title("2D trajectory\n" + plt_title)
    plt.plot(df1['x_pos'], df1['y_pos'], color="blue", label=df1_label)

    # check if second df is specified
    if df2 is not None:
        plt.plot(df2['x_pos'], df2['y_pos'], color="red", label=df2_label)
    plt.legend()
    plt.savefig(dir_path + "trajectory" + suffix + ".pdf", format="pdf")
    plt.close(fig)

    fig = state_plot_helper(cols_to_include, df1, df1_label, df2, df2_label, time_col)
    plt.xlabel("time (s)")
    plt.legend()
    plt.suptitle("states vs. time\n" + plt_title)
    plt.savefig(dir_path + "states_vs_time" + suffix + ".pdf", dpi=300, format="pdf")
    plt.close(fig)


def state_der_plots(df1, df1_label="ode", df2=None, df2_label="nn", dir_path="", plt_title="", cols_to_include="all", time_col="time"):
    """
    Plots state derivatives against time
    :param df1: state and control data
    :param df1_label: label of df1 e.g. "ground truth"
    :param df2: secondary state and control data
    :param df2_label: label of df2 e.g. "neural network"
    :param dir_path: path to store plots
    :param plt_title: title of plots
    :type cols_to_include: list[str] or "all"
    :param time_col: name of time column
    """
    fig = state_plot_helper(cols_to_include, df1, df1_label, df2, df2_label, time_col)
    plt.xlabel("time (s)")
    plt.legend()
    plt.suptitle("state der vs. time\n" + plt_title)
    plt.savefig(dir_path + "state_der_vs_time.pdf", dpi=300, format="pdf")
    plt.close(fig)


def state_plot_helper(cols_to_include, df1, df1_label, df2, df2_label, time_col):
    """
    Helper method for the methods state_variable_plots and state_der_plots
    """
    # plot all state variables along a common time axis
    fig = plt.figure(figsize=(8, 10))
    # get time data
    time_data = df1[time_col]
    # if columns to include is not all extract specified columns
    if cols_to_include is not 'all':
        df1 = df1[cols_to_include]
        if df2 is not None:
            df2 = df2[cols_to_include]
    else:
        cols_to_include = df1.columns()
    count_states = len(cols_to_include)
    for idx, col in enumerate(cols_to_include):
        ax = fig.add_subplot(count_states, 1, idx + 1)
        ax.set_ylabel(col)
        plt.plot(time_data, df1[col], color="blue", label=df1_label)
        if df2 is not None:
            plt.plot(time_data, df2[col], color="red", label=df2_label)
        # plt.grid(True, which='both', axis='both')
        if not (idx == count_states - 1):
            ax.set_xticklabels([])
    return fig


def multi_step_error_plots(error_data, time_data, x_idx, y_idx, yaw_idx, dir_path="", time_horizon=2.5, num_box_plots=5,
                           plot_hists=True, num_hist=5, track_width=3, bin_width=0.5):
    """
    Plots position and heading errors
    :param error_data: numpy 3d array of raw errors, shape is (# batches, # time steps, # states)
    :param time_data: numpy array containing time data
    :param x_idx: error data index containing errors for x position
    :param y_idx: error data index containing errors for y position
    :param yaw_idx: error data index containing errors for yaw
    :param dir_path: path to store plots
    :param time_horizon: total time to propagate dynamics which will show up as a dashed vertical line on plots
    :param num_box_plots: number of box plots to generate
    :param plot_hists: Optional arg to plot histogram of errors at specific time steps
    :param num_hist: Number of histograms to plot
    :param track_width: Width of the track, useful for comparing position errors
    :param bin_width: Width of the bins for errors smaller than track width
    """
    # calculate mean errors
    mean_errors = np.mean(error_data, axis=0)
    # calculate error std
    std_errors = np.std(error_data, axis=0)

    # figure out how many box plots to show
    errorevery = int((len(time_data)-1)/num_box_plots)

    # get the time step size and time step where time = time_horizon
    step_size = time_data[1]
    time_step = int(np.ceil(time_horizon/step_size))

    # init fig
    fig = plt.figure(figsize=(9, 6))

    print("\nMulti-step errors with time horizon %.2f" % time_horizon)
    plot_idx = 1
    # start looping over the different error data
    for idx, c, unit in zip([x_idx, y_idx, yaw_idx], ["x_pos", "y_pos", "yaw"], ["m", "m", "rad"]):
        print("Mean absolute error for %s: %.4f %s (SD=%.4f)" % (c, mean_errors[:, idx][time_step], unit, std_errors[:, idx][time_step]))
        ax = fig.add_subplot(1, 3, plot_idx)
        ax.set_ylabel("Mean absolute error (%s)" % unit)
        ax.set_xlabel("time (s)")
        # plot the mean errors
        plt.plot(time_data, mean_errors[:, idx], label=c)
        # compute the time steps where box plots will be generated
        indices = np.arange(errorevery, len(time_data), errorevery)
        data = error_data[:, indices, idx]
        # plot boxplots
        plt.boxplot(data, positions=time_data[indices], showmeans=True, meanline=True)
        # update x ticks and labels
        locs = np.arange(0, time_horizon+1, 0.5)
        ax.set_xticks(locs)
        ax.set_xticklabels(locs)
        ax.axvline(x=time_horizon, ls="--", lw=1, color='k', label="time horizon")
        # add legend
        ax.legend(loc="upper left")
        plot_idx += 1

    # title
    plt.suptitle("Multi-step prediction error on vehicle dynamics")
    # adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # save fig
    fig.savefig(os.path.join(dir_path, "multi_step_error_plot.pdf "), format="pdf")
    plt.close(fig)

    # generate histograms of errors at specific time steps if specified
    # NOTE: assumes the shape of hist data is (# batches, # time steps, # states)
    if plot_hists:
        # get the step index of the error data where a histogram will be generated
        step = np.floor(len(time_data)/num_hist)
        indices = np.arange(step-1, len(time_data), step, dtype=int)

        for i, idx in enumerate(indices):
            fig = plt.figure()
            # get error position data for all batches at the time step idx
            pos_err_data = error_data[:, idx][:, 0:2]
            j = 0
            for col, color in zip(["x_pos", "y_pos"], ['b', 'r']):
                curr_err = pos_err_data[:, j]
                ax = fig.add_subplot(1, 2, j+1)
                upper_range = max(track_width, np.ceil(np.max(curr_err)))
                bins = np.arange(0, track_width, bin_width)
                bins = np.append(bins, np.arange(track_width, upper_range+0.1, track_width))
                ax.hist(curr_err, bins=bins, label=col, density=True, color=color, edgecolor='black', alpha=0.5)
                ax.legend(loc="upper right")
                ax.set_xlabel("Error (m)")
                ax.set_ylabel("Density")
                ax.set_xticks(ticks=np.arange(0, upper_range, 1), minor=True)
                if upper_range > 10:
                    ax.set_xticks(ticks=np.arange(0, upper_range, 3))
                j += 1

            fig.suptitle("Histogram of errors at time %.02f s\nSample size = %.0f, Track width = %0.2f m" %
                         (time_data[idx], error_data.shape[0], track_width))
            # adjust spacing
            plt.subplots_adjust(wspace=0.3)
            # save fig
            fig.savefig(os.path.join(dir_path, "hist_" + str(i) + ".pdf"), format="pdf")
            plt.close(fig)


def inst_error_plots(inst_errors, state_der_cols, test_phase_dir):
    """
    Plots histogram of the instantaneous errors
    :param inst_errors: np array of the raw instantaneous errors
    :param state_der_cols: the labels
    :param test_phase_dir: dir to store plot
    """
    # make a hist for each state_der
    fig = plt.figure()
    for idx, state_der in enumerate(state_der_cols):
        ax = fig.add_subplot(2, np.ceil(len(state_der_cols)/2), idx+1)
        ax.hist(inst_errors[:, idx], label=state_der, bins=50)
        ax.set_xlabel("signed error")
        ax.set_ylabel("frequency")
        ax.set_title("Instantaneous error %s" % state_der)

    plt.tight_layout()
    plt.savefig(os.path.join(test_phase_dir, "inst_error_hist.pdf"), format="pdf")
    plt.close(fig)
