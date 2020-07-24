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
    # assumes rosbag data has already been recorded
    # e.g. rosbag record /chassisState /ground_truth/state_transformed /ground_truth/state /ground_truth/state_raw /clock /tf /imu/imu /wheelSpeeds /joy --duration=60
    print("Preprocessing data...")

    # make dir to store preprocessed data, plots, and rosbag files
    data_dir = 'data'
    plots_dir = 'preprocess_plots'
    rosbag_dir = 'rosbag_files'
    dirs = [data_dir, plots_dir, rosbag_dir]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    # reorder bag file based on header timestamps
    reorder_bag(args["rosbag_filepath"])

    # specify topics to extract from bag
    topics = [topic['name'] for topic in args['topics']]

    # extract specified topics from rosbag
    topic_file_paths = extract_bag_to_csv(args["rosbag_filepath"], topics=topics, folder=rosbag_dir)

    # list to keep track of all preprocessed data dfs
    data_dfs = []

    # variable for the final time step
    end_point = None
    # variable to keep track of resulting sampling rate
    sample_rate = None

    for topic_args in args["topics"]:
        print("\nPreprocessing topic %s..." % topic_args["name"])
        # init DataClass object
        data_obj = DataClass(topic_name=topic_args["name"], column_mapper=topic_args["col_mapper"], plot_folder=plots_dir,
                             df_file_path=topic_file_paths[topic_args['name']], make_plots=args['make_preprocessing_plots'])

        # prep data: load csv as df and rename columns
        print("Loading csv file and renaming columns..")
        data_obj.prep_data()

        # check if need to trim sequence to a specified time in seconds
        if args['total_data']:
            print("Trimming data to %.0f seconds..." % args['total_data'])
            data_obj.trim_sequence(args['total_data'] + round(data_obj.df.head(1)["time"].values[0]))

        # check if need to convert quaternion to euler angles (roll, pitch, yaw)
        if 'quaternion_to_euler' in topic_args:
            print("Converting quaternion to euler angle...")
            x, y, z, w = topic_args['quaternion_to_euler'].values()
            data_obj.df = convert_quaternion_to_euler(data_obj.df, x, y, z, w)

        # check if need to compute derivatives from data
        if 'compute_derivatives' in topic_args:
            print("Computing derivative data...")
            der = topic_args['compute_derivatives']
            data_obj.get_data_derivative(cols=der['cols'], degree=der['degree'])

        # init endpoint
        if end_point is None:
            # resample_data assumes time data starts at 0 so need to shift the sequence by setting end_point = max_time - min_time
            end_point = int(round(data_obj.df.tail(1)["time"].values[0]) - round(data_obj.df.head(1)["time"].values[0]))

        # check if need to resample data
        resample = topic_args['resample']
        if resample['cols']:
            print("Resampling data...")
            # if up/down sampling factor not specified resample data to match sampling rate of other data
            if not resample['upsampling_factor']:
                up = sample_rate
                down = len(data_obj.df)
            else:
                up = resample['upsampling_factor']
                down = resample['downsampling_factor']
            data_obj.resample_data(end_point, up, down, resample['cols'])
            sample_rate = len(data_obj.df)

        # check if need to truncate columns to min and max
        if 'trunc' in topic_args:
            print("Truncating data...")
            trunc = topic_args['trunc']
            data_obj.trunc(trunc['cols'], maximum=trunc['max'], minimum=trunc['min'])

        # save state data to disk
        print("Saving to disk...")
        data_obj.df.to_csv(os.path.join(data_dir, topic_args['filename']), index=False)

        data_dfs.append(data_obj.df)

    # merge control and state data
    final = pd.concat(data_dfs, axis=1)

    # generate state vs. time and trajectory plot for preprocessed data
    state_variable_plots(df1=final, df1_label="preprocessed ground truth", dir_path="preprocess_plots/",
                         cols_to_include=np.concatenate((args['state_cols'], args['ctrl_cols'])))

    # generate state der vs. time plots
    state_der_plots(df1=final, df1_label="preprocessed ground truth", dir_path="preprocess_plots/",
                    cols_to_include=np.concatenate((args['label_cols'], args['ctrl_cols'])))

    # check if standardize data option is set to true
    if args["standardize_data"]:
        print("\nStandardizing data...")
        # standardize features and labels
        final, scaler_list = standardize_data(final, plots_dir, args["feature_cols"], args["label_cols"])

        feature_scaler, label_scaler = scaler_list
        # add to args dict scaler objects
        args["feature_scaler"] = feature_scaler
        args["label_scaler"] = label_scaler

        # save scaler objects to disk as pickles
        pickle.dump(feature_scaler, open(data_dir + "/feature_scaler.pkl", "wb"))
        pickle.dump(label_scaler, open(data_dir + "/label_scaler.pkl", "wb"))

    # save to disk
    final.to_csv(os.path.join(data_dir, args["final_file_name"]), index=False)

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

    # move files
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
    device = torch.device(args["cuda_device"])

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
    criterion = torch.nn.SmoothL1Loss(reduction='none')

    # start training
    train(device, model_dir_path, train_loader, val_loader, args["nn_layers"], args["epochs"], args["lr"], args["weight_decay"],
          criterion=criterion, loss_weights=args["loss_weights"])


def test_model(args):
    print("\nTesting model...")
    # get cuda device
    device = torch.device(args["cuda_device"])

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
    generate_predictions(device, args["model_dir_path"], test_data_path, args["nn_layers"], args["state_cols"], args["label_cols"],
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
