"""Main script to start pipeline is divided into three components: data preprocessing, model training, and model testing"""
import yaml
import os
import numpy as np
import pandas as pd
import shutil
import pickle
from datetime import datetime
from shutil import copy
import torch
from sklearn.model_selection import train_test_split

from model_vehicle_dynamics import state_variable_plots, state_der_plots
from process_bag import reorder_bag, extract_bag_to_csv
from preprocess import DataClass, clip_start_end_times, convert_quaternion_to_euler, standardize_data
from train_dynamics_model import train, generate_predictions
from utils import make_data_loader


def preprocess_data(args):
    # TODO: remove hardcoded stuff
    # assumes rosbag data has already been recorded
    # e.g. rosbag record /chassisState /ground_truth/state_transformed /ground_truth/state /ground_truth/state_raw /clock /tf /imu/imu /wheelSpeeds /joy --duration=60
    print("Preprocessing data...")

    # reorder bagfile based on header timestamps
    reorder_bag(args["rosbag_filepath"])

    # specify topics to extract from bag
    topics = args['topics']

    # extract specified topics from rosbag
    topic_file_paths = extract_bag_to_csv(args["rosbag_filepath"], topics=topics)

    # init DataClass object for state data
    column_mapper = {"x": "x_pos", "y": "y_pos", "x.2": "u_x", "y.2": "u_y", "x.3": "roll_der", "z.3": "yaw_der", "secs": "time"}
    state_data = DataClass(topic_name=topics[0], column_mapper=column_mapper, df_file_path=topic_file_paths[topics[0]],
                           make_plots=args['make_preprocessing_plots'])

    # prep data: convert bag topic data to csv, load as df, and rename columns
    state_data.prep_data()

    # convert quaternion to euler angles (roll, pitch, yaw)
    state_data.df = convert_quaternion_to_euler(state_data.df, 'x.1', 'y.1', 'z.1', 'w')

    # init DataClass object for control data
    ctrl_data = DataClass(topic_name=topics[1], column_mapper={"secs": "time"}, df_file_path=topic_file_paths[topics[1]],
                          make_plots=args['make_preprocessing_plots'])

    # prep data
    ctrl_data.prep_data(cols_to_extract=["time", "steering", "throttle"])

    # call helper function to clip start and end times
    state_data.df, ctrl_data.df = clip_start_end_times("time", state_data.df, ctrl_data.df)

    # resampling args
    # resample_data assumes time data starts at 0 so need to shift the sequence by setting end_point = max_time - min_time
    end_point = int(round(state_data.df.tail(1)["time"].values[0]) - round(state_data.df.head(1)["time"].values[0]))
    up = args["upsampling_factor"]
    down = args["downsampling_factor"]
    state_cols = ["x_pos", "y_pos", "u_x", "u_y", "roll", "roll_der", "yaw", "yaw_der"]

    # resample state data
    state_data.resample_data(end_point, up, down, state_cols)

    # truncate roll and yaw
    state_data.trunc(["roll", "yaw"], maximum=np.pi, minimum=-1. * np.pi)

    # remove old data from state df
    state_cols.append("time")
    state_data.extract_cols(state_cols)

    # get derivatives from state data
    state_data.get_data_derivative(cols=["u_x", "u_y", "yaw_der"], degree=3)  # roll_der given by ground truth topic

    # make dir to store preprocessed data
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # save state data to disk
    state_data.df.to_csv(data_dir + "/df_state.csv", index=False)

    # resample control data
    ctrl_cols = ["steering", "throttle"]

    # sample rate for control data is set such that the number of sampled points is equal to the number of sampled state data
    ctrl_data.resample_data(end_point, len(state_data.df), len(ctrl_data.df), ctrl_cols)

    # truncate throttle and steering signal
    ctrl_data.trunc(ctrl_cols, maximum=1.0, minimum=-1.0)

    # save control data to disk
    ctrl_data.df.to_csv(data_dir + "/df_ctrl.csv", index=False)

    # merge control and state data
    final = pd.concat([state_data.df, ctrl_data.df[ctrl_cols]], axis=1)

    # TODO: before or after standardization
    # generate state vs. time and trajectory plot for preprocessed data
    state_variable_plots(df1=final, df1_label="preprocessed ground truth", dir_path="preprocess_plots/",
                         cols_to_include=["x_pos", "y_pos", "u_x", "u_y", "roll", "yaw", "yaw_der", "steering", "throttle"])

    # generate state der vs. time plots
    state_der_plots(df1=final, df1_label="preprocessed ground truth", dir_path="preprocess_plots/",
                    cols_to_include=["u_x_der", "u_y_der", "roll_der", "yaw_der_der", "steering", "throttle"])

    # check if standardize data option is set to true
    if args["standardize_data"]:
        # standardize features and labels
        final, scaler_list = standardize_data(final, state_data.plot_folder, args["feature_cols"], args["label_cols"])

        feature_scaler, label_scaler = scaler_list
        # add to args dict scaler objects
        args["feature_scaler"] = feature_scaler
        args["label_scaler"] = label_scaler

        # save scaler objects to disk as pickles
        pickle.dump(feature_scaler, open(data_dir + "/feature_scaler.pkl", "wb"))
        pickle.dump(label_scaler, open(data_dir + "/label_scaler.pkl", "wb"))

    # save to disk
    final.to_csv(data_dir + "/" + args["final_file_name"], index=False)

    # move files to a common folder
    folder = "pipeline_files/" + args["run_name"]
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        # prompt user if directory already exists
        answer = None
        while not(answer == "y" or answer == "n"):
            answer = raw_input("Replace already existing directory %s? (y/n): " % folder).lower().strip()
            print("")
        if answer == "y":
            shutil.rmtree(folder)
            os.makedirs(folder)
        else:
            print("Keeping old directory and leaving preprocessing files in working directory %s..." % os.getcwd())
            exit(0)

    dirs = [state_data.plot_folder, "rosbag_files", data_dir]
    for d in dirs:
        shutil.move(d, folder)

    print("Done preprocessing data")


def train_model(args):
    print("\nTraining model...")
    # get model name
    if args["model_dir_name"]:
        model_dir_path = args["results_dir"] + args["model_dir_name"]
    else:
        # make model directory name time/date if no name specified in args
        model_dir_path = args["results_dir"] + str(datetime.now().strftime('%m-%d_%H:%M/'))

    # add/update model_dir_path key in args dict
    args["model_dir_path"] = model_dir_path

    # setup directory to save models
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    # save training details
    copy("config.yml", model_dir_path)

    # get cuda device
    device = torch.device(args["device"])

    # get training data path
    if args['training_data_path']:
        training_data_path = args['training_data_path']
    else:
        training_data_path = "pipeline_files/" + args['run_name'] + "/data/train_data.csv"

    # generate indices for training and validation
    a = np.arange(len(pd.read_csv(training_data_path)))
    # get the specified fraction of test data to use
    frac = args["train_data_fraction"]
    tr_ind, val_ind = train_test_split(a, train_size=0.8*frac, test_size=0.2*frac, shuffle=True)

    # init training data loader
    train_loader = make_data_loader(training_data_path, indices=tr_ind, batch_size=args["batch_size"],
                                    feature_cols=args["feature_cols"], label_cols=args["label_cols"])
    # init validation data loader
    val_loader = make_data_loader(training_data_path, indices=val_ind, batch_size=args["batch_size"],
                                  feature_cols=args["feature_cols"], label_cols=args["label_cols"])

    # use Huber loss since don't care about outliers
    criterion = torch.nn.SmoothL1Loss()

    # start training
    train(device, model_dir_path, train_loader, val_loader, args["nn_layers"], args["epochs"], args["lr"], args["weight_decay"],
          criterion=criterion, loss_weights=args["loss_weights"])


def test_model(args):
    print("\nTesting model...")
    # get cuda device
    device = torch.device(args["device"])

    # get test data path
    if args['test_data_path']:
        test_data_path = args['test_data_path']
    else:
        test_data_path = "pipeline_files/" + args['run_name'] + "/data/test_data.csv"

    # check if args dict contains the scalers
    if "feature_scaler" not in args and args["standardize_data"]:
        if not args["scaler_paths"]:
            print("'standardize_data' arg set to True but no scaler object found in args dict and no path specified by 'scaler_paths'...")
            exit(1)
        else:
            # no scalers in dict and standardize data option is set to true so get scalers from disk
            args["feature_scaler"] = pickle.load(open(os.path.join(args["scaler_paths"], "feature_scaler.pkl"), "rb"))
            args["label_scaler"] = pickle.load(open(os.path.join(args["scaler_paths"], "label_scaler.pkl"), "rb"))
    else:
        args["feature_scaler"] = None
        args["label_scaler"] = None

    # start test phase
    generate_predictions(device, args["model_dir_path"], test_data_path, args["nn_layers"], args["state_cols"], args["state_der_cols"],
                         args["ctrl_cols"], time_horizon=args["time_horizon"], data_frac=args["test_data_fraction"],
                         feature_scaler=args["feature_scaler"], label_scaler=args["label_scaler"])


def main():
    # load config file into args
    config = "./config.yml"
    with open(config, "r") as yaml_file:
        args = yaml.load(yaml_file)

    options = ["preprocess_data", "train_model", "test_model"]
    if not any([args[i] for i in options if i in args.keys()]):
        print("No option has been selected!")
        print("One of %s needs to be set to True in config.yml" % str(options))
        exit(1)

    if args["preprocess_data"]:
        preprocess_data(args)

    if args["train_model"]:
        train_model(args)

    if args["test_model"]:
        test_model(args)


if __name__ == '__main__':
    main()
