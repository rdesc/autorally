"""Trains and evaluates neural network for dynamics model"""
import gc
import matplotlib.pyplot as plt
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from model_vehicle_dynamics import compute_state_ders, state_variable_plots, state_error_plots, state_der_plots
from utils import setup_model, make_test_data_loader

torch.manual_seed(0)


def train(device, model_dir, train_loader, val_loader, nn_layers, epochs, lr, weight_decay=0.0, criterion=torch.nn.L1Loss(), loss_weights=None):
    """
    Model training and validation phase
    :param device: torch device object
    :type model_dir: str
    :param train_loader: data loader with test data
    :param val_loader: data loader with validation data
    :type nn_layers: list[int]
    :type epochs: int
    :type lr: float
    :type weight_decay: float
    :param criterion: loss function
    :type loss_weights: list[float]
    """
    # get start time
    start = time.time()
    # set up model
    model = setup_model(layers=nn_layers)  # use default activation
    # load model onto device
    model.to(device)
    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # weight_decay is L2 penalty term to loss
    # set up data loaders
    data_loaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": train_loader.dataset.__len__(), "val": val_loader.dataset.__len__()}
    # store train and val losses for plotting
    losses = {"train": [], "val": []}

    best_val_loss = np.inf

    if loss_weights is None:
        loss_weights = np.ones(nn_layers[-1])  # if no weights are specified for the loss just set the weights to 1

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
                    # apply the specified loss weights and compute the loss
                    loss = criterion(torch.t(torch.mul(outputs, torch.tensor(loss_weights, dtype=torch.float).to(device))),
                                     torch.t(torch.mul(labels.float(), torch.tensor(loss_weights, dtype=torch.float).to(device))))

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
    plt.ylim(bottom=0.0)
    plt.title("Loss")
    plt.legend(['train', 'val'], loc='upper right')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    fig.savefig(os.path.join(model_dir, "loss.pdf"), format="pdf")
    plt.close(fig)


def generate_predictions(device, model_dir, data_path, nn_layers, state_cols, state_der_cols, ctrl_cols, time_col='time',
                         time_horizon=2.5, state_dim=7, data_frac=1.0):
    """
    Model test phase. Generates truth and nn predicted trajectory for each batch
    NOTE: many parts of this test phase are currently hard coded to a specific problem
    :param device: torch device object
    :type model_dir: str
    :type data_path: str
    :type nn_layers: list[int]
    :type state_cols: list[str]
    :type state_der_cols: list[str]
    :type ctrl_cols: list[str]
    :type time_col: str
    :param time_horizon: total time to propagate dynamics for
    :param state_dim: size of state space
    :param data_frac: fraction of test data to use
    """
    print("\nGenerating predictions from trained model...")

    # get time step from data
    time_step = pd.read_csv(data_path).head(2)[time_col].values[1]
    print("time step: %.04f" % time_step)

    # determine batch size
    batch_size = int(np.ceil(time_horizon / time_step))
    print("batch size: %.0f" % batch_size)

    # setup data loader
    indices = np.arange(int(data_frac * len(pd.read_csv(data_path))))
    data_loader = make_test_data_loader(data_path, batch_size, state_cols, state_der_cols, ctrl_cols, indices=indices, time_col=time_col)

    # number of batches to do
    total_batches = data_loader.dataset.__len__() // batch_size

    # load model architecture
    model = setup_model(layers=nn_layers)
    # load model onto device
    model.to(device)
    # load weights + biases from pretrained model
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"))["model_state_dict"])

    # var to keep track of errors
    errors_list = []

    with torch.no_grad():
        # set model to eval mode
        model.eval()

        # keep track of current batch number
        batch_num = 0

        # generate a trajectory for each batch
        for truth_states, truth_state_ders, ctrls, time_data in data_loader:
            print('\nBatch %i/%i' % (batch_num, total_batches))
            print('-' * 10)
            num_steps = truth_states.size(0)
            # skip last batch if it is less than batch size
            if num_steps < batch_size:
                print("Skipping final batch...")
                continue

            # make a new folder to store results from this batch
            batch_folder = "test_phase/batch_" + str(batch_num) + "/"
            if not os.path.exists(model_dir + batch_folder):
                os.makedirs(model_dir + batch_folder)

            # update batch number
            batch_num += 1

            # init state variables
            nn_states = np.full((num_steps, state_dim), 0, np.float)
            # set initial conditions
            nn_states[0] = truth_states[0].cpu().numpy()
            # array to store all state derivatives
            state_ders = np.full((num_steps, truth_state_ders.size(1)), 0, np.float)

            # iterate through each step of trajectory
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
                state_der = compute_state_ders(curr_state, output, negate_yaw_der=False)

                # update states
                nn_states[idx + 1] = curr_state + state_der * time_step

                # save state derivatives
                state_ders[idx + 1] = state_der[3:]

            curr_errors = np.abs(nn_states - truth_states.numpy())
            # compute yaw errors
            curr_errors[:,2] = [e % np.pi for e in curr_errors[:,2]]
            # save errors
            errors_list.append(curr_errors)
            # print some errors on final time step
            print("abs x error (m): %.02f" % curr_errors[-1][0])
            print("abs y error (m): %.02f" % curr_errors[-1][1])
            print("abs yaw error (rad): %.02f" % (curr_errors[-1][2]))

            # convert time data to numpy
            time_data = time_data.cpu().numpy()
            # convert control data to numpy
            ctrls = ctrls.cpu().numpy()

            # create pandas DataFrames
            df_nn = pd.DataFrame(data=np.concatenate((np.reshape(time_data, (len(time_data), 1)), nn_states, state_ders, ctrls), axis=1),
                                 columns=np.concatenate(([time_col], state_cols, state_der_cols, ctrl_cols)))
            df_nn.to_csv(model_dir + batch_folder + "nn_state_variables.csv", index=False, header=True)

            # load ground truth data
            df_truth = pd.DataFrame(data=np.concatenate((np.reshape(time_data, (len(time_data), 1)), truth_states, truth_state_ders, ctrls), axis=1),
                                    columns=np.concatenate(([time_col], state_cols, state_der_cols, ctrl_cols)))
            df_truth.to_csv(model_dir + batch_folder + "truth_state_variables.csv", index=False, header=True)

            # plot trajectories and state vs. time
            state_variable_plots(df1=df_truth, df1_label="ground truth", df2=df_nn, df2_label="nn", dir_path=model_dir + batch_folder,
                                 cols_to_include=np.concatenate((state_cols, ctrl_cols)))

            # plot state der vs. time
            state_der_plots(df1=df_truth, df1_label="ground truth", df2=df_nn, df2_label="nn", dir_path=model_dir + batch_folder,
                            cols_to_include=np.concatenate((state_der_cols, ctrl_cols)))

        # calculate mean errors
        mean_errors = np.mean(errors_list, axis=0)

        # hacky way to get first set of time data
        _, _, _, time_data = iter(data_loader).next()

        # make data frame containing mean errors and time data
        df_errors = pd.DataFrame(data=np.concatenate((np.reshape(time_data, (len(time_data), 1)), mean_errors), axis=1),
                                 columns=np.concatenate(([time_col], state_cols)))
        # plot errors
        state_error_plots(df_errors, ["x_pos", "y_pos"], "yaw", dir_path=model_dir + "/test_phase")
