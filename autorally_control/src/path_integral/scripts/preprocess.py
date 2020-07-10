"""Data preprocessing script for state + control data (some parts are specific to rosbag data)"""
import pandas as pd
from scipy import signal, interpolate
import numpy as np
import matplotlib.pyplot as plt
import os
from tf.transformations import euler_from_quaternion

from process_bag import extract_bag_to_csv


class DataClass:
    def __init__(self, topic_name=None, column_mapper=None, bagfile_path=None, df_file_path=None, make_plots=False):
        self.topic_name = topic_name
        self.column_mapper = column_mapper
        self.bagfile_path = bagfile_path
        self.df_file_path = df_file_path
        self.make_plots = make_plots
        self.df = None

    def prep_data(self, bag_to_csv=False, cols_to_extract=None):
        """
        Convenient method that groups some steps together
        """
        if bag_to_csv:
            self.extract_bag_to_csv()
        self.load_df()
        self.rename_cols()
        if cols_to_extract is not None:
            self.extract_cols(cols_to_extract)

    def load_df(self, format_time=True):
        """
        Loads csv data as pandas DataFrame
        NOTE: pandas will append ".$int" to identical column names e.g. x, x.1, x.2...
        """
        self.df = pd.read_csv(self.df_file_path)
        if format_time:
            self.format_time_col()

    def extract_bag_to_csv(self):
        """
        Extracts topic data from rosbag
        """
        topic_path = extract_bag_to_csv(self.bagfile_path, topics=[self.topic_name])
        self.df_file_path = topic_path[0]

    def format_time_col(self):
        """
        Combines seconds and nanoseconds column
        """
        self.df["secs"] = self.df["secs"] + self.df["nsecs"] / 1e9

    def rename_cols(self):
        """
        Renames the columns based on column mapper
        """
        if self.column_mapper is not None:
            self.df = self.df.rename(columns=self.column_mapper)

    def extract_cols(self, cols):
        """
        Extracts the specified columns from the df
        :type cols: list
        """
        self.df = self.df[cols]

    def trunc(self, cols, max=1.0, min=-1.0):
        """
        Truncates the specified columns to max and min
        :type cols: list
        """
        for c in cols:
            # get the data
            x = self.df[c]
            for idx, item in enumerate(x):
                item = np.min((item, max))
                x[idx] = np.max((item, min))

            # update the data in the dataFrame
            self.df[c] = x

            if self.make_plots:
                fig = plt.figure()
                plt.plot(self.df["time"], self.df[c], 'b-')
                plt.title("%s trunacted" % c)
                plt.xlabel("time (s)")
                plot_folder = "preprocess_plots"
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)
                file_name = c + "_truncated"
                fig.savefig(plot_folder + "/" + file_name + ".pdf", format="pdf")
                plt.close(fig)

    def get_data_derivative(self, cols, degree):
        """
        Gets derivative data from the specified columns
        :type cols: list
        :type degree: int
        """
        t = self.df["time"]
        for c in cols:
            y = self.df[c]
            # construct spline
            spl = interpolate.UnivariateSpline(t, y, k=degree, s=0)

            # construct new spline representing derivative of spline
            spl_der = spl.derivative(n=1)

            # save to dataFrame
            key = c + "_der"
            self.df[key] = spl_der(t)

            if self.make_plots:
                fig = plt.figure()
                plt.plot(t, spl_der(t), 'g-')
                plt.plot(t, spl(t), 'b-', t, y, 'r-')
                plt.title("%s spline and spline der" % c)
                plt.legend(["spline_der", "spline", "data"], loc='best')
                plt.xlabel("time (s)")
                plot_folder = "preprocess_plots"
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)
                file_name = c + "_der"
                fig.savefig(plot_folder + "/" + file_name + ".pdf", format="pdf")
                plt.close(fig)

    def resample_data(self, end_point, up_factor, down_factor, cols):
        """
        Resamples data from the specified columns
        Resulting sample rate is up / down times the original sample rate
        NOTE: Assumes time data starts at 0. Can shift sequence by passing endpoint = max time - min time
        :type end_point: int
        :type up_factor: int
        :type down_factor: int
        :type cols: list
        """
        # init some vars
        t_new = None
        df_new = {}

        # iterate over each piece of data to resample
        for i, c in enumerate(cols):
            if c in self.df.columns:
                y = self.df[c]
                f_poly = signal.resample_poly(y, up_factor, down_factor)
                df_new[c] = f_poly
            else:
                print("Column '%s' not found..." % c)
                continue

            if t_new is None:
                t_new = np.linspace(0, end_point, len(f_poly))
                df_new["time"] = t_new

            if self.make_plots:
                fig = plt.figure()
                t = np.linspace(0, end_point, len(y), endpoint=False)
                plt.plot(df_new["time"], f_poly, 'b-', t, y, 'r-')
                plt.title("%s resample" % c)
                plt.legend(["resample_poly", "data"], loc='best')
                plt.xlabel("time (s)")
                plot_folder = "preprocess_plots"
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)
                file_name = c + "_resample"
                fig.savefig(plot_folder + "/" + file_name + ".pdf", format="pdf")
                plt.close(fig)

        # replace with resampled data
        self.df = pd.DataFrame(df_new)


# Helper functions
def convert_quaternion_to_euler(df, x_col, y_col, z_col, w_col):
    """
    Converts quaternion data to euler angles
    :param df: pandas data frame
    :type x_col: str
    :type y_col: str
    :type z_col: str
    :type w_col: str
    """
    # lists to store calculated roll, pitch, and yaw for each row
    roll_list = []
    pitch_list = []
    yaw_list = []
    # iterate over each data frame row
    for idx, row in df.iterrows():
        # convert quaternion to euler
        (roll, pitch, yaw) = euler_from_quaternion([row[x_col], row[y_col], row[z_col], row[w_col]])
        # update lists
        roll_list.append(roll)
        pitch_list.append(pitch)
        yaw_list.append(yaw)

    # add new columns to data frame
    df["roll"] = roll_list
    df["pitch"] = pitch_list
    df["yaw"] = yaw_list

    return df


def clip_start_end_times(col, *args):
    """
    Clips all DataFrames to a common start and end time
    :type col: str
    :param args: dfs to consider when clipping
    """
    # args are data frames
    # helper to make start and end times of data as close as possible
    start = []
    end = []
    # get all the start and end times from dfs
    for df in args:
        start.append(df.head(1)[col].values[0])
        end.append(df.tail(1)[col].values[0])

    # find the max start time
    start_max = np.ceil(max(start))
    # find the min end time
    end_min = np.floor(min(end))
    # list to store updated data frames
    new_dfs = []
    # clip all time columns at this time and update the df
    for idx, df in enumerate(args):
        df = df[df[col] >= start_max]
        new_dfs.append(df[df[col] <= end_min])

    return new_dfs
