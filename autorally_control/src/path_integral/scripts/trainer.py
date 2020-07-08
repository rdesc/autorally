import yaml
import os
import numpy as np
import pandas as pd
import shutil
from datetime import datetime
from shutil import copy
import torch
from sklearn.model_selection import train_test_split

from process_bag import reorder_bag, extract_bag_to_csv
from preprocess import DataClass, clip_start_end_times, convert_quaternion_to_euler
from train_dynamics_model import train, make_data_loader


def preprocess_data(args):
    # assumes rosbag data has already been recorded
    # e.g. rosbag record /chassisState /ground_truth/state_transformed /ground_truth/state /ground_truth/state_raw /clock /tf /imu/imu /wheelSpeeds /joy --duration=600
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

    # get derivatives from state data
    state_data.get_data_derivative(cols=["u_x", "u_y", "roll", "yaw_der"], degree=3)

    # resampling args
    end_point = round(state_data.df.tail(1)["time"].values[0])
    up = 1
    down = 2
    state_cols = ["x_pos", "y_pos", "u_x", "u_y", "u_x_der", "u_y_der", "roll", "roll_der", "yaw", "yaw_der", "yaw_der_der"]

    # resample state data
    state_data.resample_data(end_point, up, down, state_cols)

    # truncate roll and yaw
    state_data.trunc(["roll", "yaw"], max=np.pi, min=-1.*np.pi)

    # remove old data from state df
    state_cols.append("time")
    state_data.extract_cols(state_cols)

    # make dir to store preprocessed data
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # save state data to disk
    state_data.df.to_csv(data_dir + "/df_state.csv")

    # resample control data
    ctrl_cols = ["steering", "throttle"]

    # sample rate for control data is set such that the number of sampled points is equal to the number of sampled state data
    ctrl_data.resample_data(end_point, len(state_data.df), len(ctrl_data.df), ctrl_cols)

    # truncate throttle and steering signal
    ctrl_data.trunc(ctrl_cols, max=1.0, min=-1.0)

    # save control data to disk
    ctrl_data.df.to_csv(data_dir + "/df_ctrl.csv")

    # merge control and state data
    final = pd.concat([state_data.df, ctrl_data.df[["steering", "throttle"]]], axis=1)
    # save to disk
    final.to_csv(data_dir + "/data.csv")

    # move files to a common folder
    folder = "pipeline_files/" + args["run_name"]
    if not os.path.exists(folder):
        os.makedirs(folder)

    dirs = ["preprocess_plots", "rosbag_files", data_dir]
    for d in dirs:
        shutil.move(d, folder)

    print("Done preprocessing data")


def train_model(args):
    # Append time/date to directory name
    creation_time = str(datetime.now().strftime('%m-%d_%H:%M/'))
    model_dir = args["results_dir"] + creation_time  # TODO: move to pipeline_files?

    # setup directory to save models
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # save training details
    copy("config.yml", model_dir)

    device = torch.device(args["device"])

    # get data path
    data_path = args['data_path'] if args['data_path'] else "pipeline_files/" + args['run_name'] + "/data/data.csv"

    # generate indices for training and validation
    a = np.arange(len(pd.read_csv(data_path)))
    tr_ind, val_ind = train_test_split(a, train_size=0.8, test_size=0.2, shuffle=True)

    # set column names for features and labels
    feature_cols = ["roll", "u_x", "u_y", "yaw_der", "steering", "throttle"]
    label_cols = ["roll_der", "u_x_der", "u_y_der", "yaw_der_der"]

    # init training data loader
    train_loader = make_data_loader(data_path, indices=tr_ind, batch_size=args["batch_size"], feature_cols=feature_cols, label_cols=label_cols)
    # init validation data loader
    val_loader = make_data_loader(data_path, indices=val_ind, batch_size=args["batch_size"], feature_cols=feature_cols, label_cols=label_cols)

    # use Huber loss since don't care about outliers
    criterion = torch.nn.SmoothL1Loss()

    # start training
    train(device, train_loader, val_loader, args["nn_layers"], args["epochs"], args["lr"], model_dir, criterion=criterion)

    # test phase
    # generate_predictions("", args["nn_layers"], args["state_dim"])


def main():
    # load config file into args
    config = "./config.yml"
    with open(config, "r") as yaml_file:
        args = yaml.load(yaml_file)

    if args["preprocess_data"]:
        preprocess_data(args)

    if args["train_model"]:
        train_model(args)


if __name__ == '__main__':
    main()
