import yaml
import os
import numpy as np
from process_bag import reorder_bag, extract_bag_to_csv
from preprocess import DataClass, clip_start_end_times, convert_quaternion_to_euler

if __name__ == '__main__':
    # TODO: improve dir structure
    # assumes rosbag data has already been recorded
    # would be cool if data collection was also part of automated pipeline maybe by sending piecewise constant controls

    # load config file into args
    config = "./config.yml"
    with open(config, "r") as yaml_file:
        args = yaml.load(yaml_file)

    # TODO:
    run_name = "run1"

    # reorder bagfile based on header timestamps
    # reorder_bag(args["rosbag_filepath"])
    #
    # specify topics to extract from bag
    topics = args['topics']
    # topic_file_paths = extract_bag_to_csv(args["rosbag_filepath"], topics=topics)
    topic_file_paths = {'/chassisState': 'rosbag_files/./sim_state_control_data_run_1/chassisState.csv', '/ground_truth/state_transformed': 'rosbag_files/./sim_state_control_data_run_1/ground_truthstate_transformed.csv'}

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
    state_data.get_data_derivative(cols=["u_x", "u_y", "roll", "yaw", "yaw_der"], degree=3)

    # resampling args
    size = len(state_data.df)
    up = 1
    down = 2
    state_cols = ["x_pos", "y_pos", "u_x", "roll", "yaw", "yaw_der", "roll_der", "u_x_der", "u_y_der"]

    # resample state data
    state_data.resample_data(size, up, down, state_cols)
    # TODO: trunc roll, and yaw at pi and -pi
    state_data.df.to_csv("data/df_state.csv")

    # resample control data
    ctrl_cols = ["steering", "throttle"]
    ctrl_data.resample_data(size, size * up / down, len(ctrl_data.df), ctrl_cols)
    # truncate throttle and steering signal
    ctrl_data.trunc(ctrl_cols)

    # TODO: get data derivative
    # TODO: resample
    # save

    # TODO: move files
    folder = "pipeline_files"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # move rosbag_files, data, preprocess_plots

# rosbag record /chassisState /ground_truth/state_transformed /ground_truth/state /ground_truth/state_raw /clock /tf /imu/imu /wheelSpeeds /joy --duration=600
