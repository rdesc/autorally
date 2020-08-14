"""Trains and evaluates neural network for dynamics model"""
import gc
import matplotlib.pyplot as plt
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from utils import setup_model, make_test_data_loader, torch_model_to_npz, compute_state_ders, state_variable_plots, \
    state_der_plots, state_error_plots


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
    # model = npz_to_torch_model(filename="../params/models/autorally_nnet_09_12_2018.npz", model=model) # to load pretrained model
    # load model onto device
    model.to(device)
    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # weight_decay is L2 penalty term to loss
    # set up data loaders
    data_loaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": train_loader.dataset.__len__(), "val": val_loader.dataset.__len__()}
    # dict to store train and val losses for plotting
    losses = {"train": [], "val": []}
    # label cols
    label_cols = train_loader.dataset.label_cols
    # dict to store losses of the different loss components
    split_losses = {"train": {}, "val": {}}
    for label_col in label_cols:
        split_losses['train'][label_col] = []
        split_losses['val'][label_col] = []

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
            temp_split_losses = {}
            for label_col in label_cols:
                temp_split_losses[label_col] = 0.0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # clear gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    # get output from model
                    outputs = model(inputs.float64())
                    # apply the specified loss weights and compute the loss
                    loss = criterion(torch.t(torch.mul(outputs, torch.tensor(loss_weights, dtype=torch.float64).to(device))),
                                     torch.t(torch.mul(labels.float64(), torch.tensor(loss_weights, dtype=torch.float64).to(device))))

                    # save loss splits
                    for label_col, split_loss in zip(label_cols, loss):
                        temp_split_losses[label_col] += torch.sum(split_loss).item()

                    # apply reduction to loss
                    # loss = torch.sum(loss)
                    loss = torch.mean(loss)

                    if phase == "train":
                        # calculate the gradients
                        loss.backward()
                        # update all the parameters based on the gradients calculated
                        optimizer.step()

                    # updating stats
                    # running_loss += loss.item()
                    running_loss += loss.item()*inputs.size(0)  # multiply by batch size since calculated loss was the mean

            # calculate loss for epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            losses[phase].append(epoch_loss)
            print('%s Loss: %.4f' % (phase, epoch_loss))

            # calculate split losses
            for label_col in label_cols:
                split_losses[phase][label_col].append(temp_split_losses[label_col] / dataset_sizes[phase])

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

    # convert model to npz format for mppi usage
    torch_model_to_npz(model, model_dir)

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

    # plot loss splits
    fig = plt.figure()
    for label_col in label_cols:
        plt.plot(x, split_losses['val'][label_col], label=label_col)
    plt.title("Val loss splits")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(label_cols, loc='best')
    plt.ylim(bottom=0.0)
    fig.savefig(os.path.join(model_dir, "loss_splits.pdf"), format="pdf")
    plt.close(fig)


def generate_predictions(device, model_dir, data_path, nn_layers, state_cols, state_der_cols, ctrl_cols, time_col='time',
                         time_horizon=2.5, state_dim=7, data_frac=1.0, feature_scaler=None, label_scaler=None, skip_first_batch=True):
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
    :param feature_scaler: sklearn standard scaler for features
    :param label_scaler: sklearn standard scaler for labels
    :param skip_first_batch: option to skip first batch
    """
    print("\nGenerating predictions from trained model...")

    # folder to store all test phase files
    test_phase_dir = os.path.join(model_dir, "test_phase/")

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
    state_dict = torch.load(os.path.join(model_dir, "model.pt"))
    # check if model was saved as part of a dict
    if "model_state_dict" in state_dict.keys():
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)

    # var to keep track of errors from propagating dynamics
    errors_list = []

    # var to keep track of instantaneous errors
    inst_errors = []

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

            # skip first batch if specified
            if skip_first_batch and batch_num == 0:
                print("Skipping first batch...")
                batch_num += 1
                continue

            # make a new folder to store results from this batch
            batch_folder = test_phase_dir + "batch_" + str(batch_num) + "/"
            if not os.path.exists(batch_folder):
                os.makedirs(batch_folder)

            # update batch number
            batch_num += 1

            # init state variables
            nn_states = np.full((num_steps, state_dim), 0, np.float)
            # set initial conditions
            nn_states[0] = truth_states[0].cpu().numpy()
            # array to store all state derivatives
            state_ders = np.full((num_steps, truth_state_ders.size(1)), 0, np.float)

            # FIXME: remove hard coded stuff
            # iterate through each step of trajectory
            for idx in range(num_steps - 1):
                # prep inputs to feed to neural network
                # x1 is the input from continuously feeding the predicted state back into the model
                x1 = torch.tensor(
                    [nn_states[idx][3], nn_states[idx][4], nn_states[idx][5], nn_states[idx][6], ctrls[idx][0], ctrls[idx][1]])

                # x2 is just the input from the truth state used to calculate instantaneous errors of the model
                x2 = torch.tensor(
                    [truth_states[idx][3], truth_states[idx][4], truth_states[idx][5], truth_states[idx][6], ctrls[idx][0], ctrls[idx][1]])

                # get the current state
                curr_state = nn_states[idx]

                # if data was standardized, apply transform on test data
                if feature_scaler is not None:
                    x1 = torch.tensor(feature_scaler.transform(x1.reshape(1, -1))[0])
                    x2 = torch.tensor(feature_scaler.transform(x2.reshape(1, -1))[0])

                # get outputs of neural network
                output1 = model(x1.float64().to(device)).cpu().numpy()
                output2 = model(x2.float64().to(device)).cpu().numpy()

                # apply inverse transform on output
                if label_scaler is not None:
                    output1 = label_scaler.inverse_transform(output1)
                    output2 = label_scaler.inverse_transform(output2)

                # output1 = truth_state_ders[idx + 1].cpu().numpy()  # use the truth derivatives

                # compute the state derivatives
                state_der = compute_state_ders(curr_state, output1, negate_yaw_der=True)  # NOTE: set negate_yaw_der to True if using autorally's model

                # update states
                nn_states[idx + 1] = curr_state + state_der * time_step

                # save state derivatives
                state_ders[idx + 1] = state_der[3:]

                # calculate the instantaneous signed error
                inst_errors.append(truth_state_ders[idx].cpu().numpy() - output2)

            curr_errors = np.abs(nn_states - truth_states.numpy())
            # compute yaw errors
            curr_errors[:, 2] = [e % (2 * np.pi) for e in curr_errors[:, 2]]
            curr_errors[:, 2] = [(2 * np.pi) - e if e > np.pi else e for e in curr_errors[:, 2]]
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
            df_nn.to_csv(os.path.join(batch_folder, "nn_state_variables.csv"), index=False, header=True)

            # load ground truth data
            df_truth = pd.DataFrame(data=np.concatenate((np.reshape(time_data, (len(time_data), 1)), truth_states, truth_state_ders, ctrls), axis=1),
                                    columns=np.concatenate(([time_col], state_cols, state_der_cols, ctrl_cols)))
            df_truth.to_csv(os.path.join(batch_folder, "truth_state_variables.csv"), index=False, header=True)

            # plot trajectories and state vs. time
            state_variable_plots(df1=df_truth, df1_label="ground truth", df2=df_nn, df2_label="nn", dir_path=batch_folder,
                                 cols_to_include=np.concatenate((state_cols, ctrl_cols)))

            # plot state der vs. time
            state_der_plots(df1=df_truth, df1_label="ground truth", df2=df_nn, df2_label="nn", dir_path=batch_folder,
                            cols_to_include=np.concatenate((state_der_cols, ctrl_cols)))

        # save all errors to disk
        errors_array = np.array(errors_list)
        np.save(file=os.path.join(test_phase_dir, "err.npy"), arr=errors_array)

        # hacky way to get first set of time data
        _, _, _, time_data = iter(data_loader).next()
        time_data = time_data.cpu().numpy()

        # plot mean errors and their std
        state_error_plots(errors_array, time_data, x_idx=0, y_idx=1, yaw_idx=2, dir_path=test_phase_dir, num_box_plots=3, plot_hists=True, num_hist=6)

        # calculate error std
        std_errors = np.std(errors_array, axis=0)
        # calculate mean errors
        mean_errors = np.mean(errors_array, axis=0)

        # make column names for error std
        std_error_cols = [col + "_std" for col in state_cols]
        # make data frame containing mean errors and time data
        df_errors = pd.DataFrame(data=np.concatenate((np.reshape(time_data, (len(time_data), 1)), mean_errors, std_errors), axis=1),
                                 columns=np.concatenate(([time_col], state_cols, std_error_cols)))

        # save df to disk
        df_errors.to_csv(os.path.join(test_phase_dir, "mean_errors.csv"), index=False, header=True)

        # plot instantaneous error histograms
        inst_errors = np.array(inst_errors)
        df_inst_errors = pd.DataFrame(data=inst_errors, columns=state_der_cols)
        df_inst_errors.hist()
        plt.savefig(os.path.join(test_phase_dir, "inst_error_hist.pdf"), format="pdf")
