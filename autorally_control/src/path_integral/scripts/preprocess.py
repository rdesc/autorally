import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os


class DataClass:
    """
    TODO: doc
    """

    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.format_time_col()

    def format_time_col(self):
        self.df["secs"] = self.df["secs"] + self.df["nsecs"] / 1e9

    def rename_cols(self, cols_mapper):
        self.df = self.df.rename(columns=cols_mapper)

    def extract_cols(self, cols):
        self.df = self.df[cols]

    def trunc(self, cols, max=1.0, min=-1.0, make_plots=False):
        for c in cols:
            # get the data
            x = self.df[c]
            for idx, item in enumerate(x):
                item = np.min((item, max))
                x[idx] = np.max((item, min))

            # update the data in the dataFrame
            self.df[c] = x

            if make_plots:
                fig = plt.figure()
                plt.plot(self.df["time"], self.df[c], 'b-')
                plt.title("%s trunacted" % c)
                plot_folder = "preprocess_plots"
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)
                file_name = c + "_truncated"
                fig.savefig(plot_folder + "/" + file_name + ".pdf", format="pdf")
                plt.clf()

    def resample_data(self, size, up_factor, down_factor, cols, make_plots=False):
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
                t_new = np.linspace(0, size, len(f_poly))
                df_new["time"] = t_new

            if make_plots:
                fig = plt.figure()
                t = np.linspace(0, size, len(y), endpoint=False)
                plt.plot(t_new, f_poly, 'b-', t, y, 'r-')
                plt.title("%s resample" % c)
                plt.legend(["resample_poly", "data"], loc='best')
                plot_folder = "preprocess_plots"
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)
                file_name = c + "_resample"
                fig.savefig(plot_folder + "/" + file_name + ".pdf", format="pdf")
                plt.clf()

        # replace with resampled data
        self.df = pd.DataFrame(df_new)


if __name__ == '__main__':
    up = 10
    down = 20

    # state data
    df_state = DataClass("rosbag_files/dynamics_model/ground_truthstate.csv")
    columns_mapper = {"x": "x_pos", "y": "y_pos", "x.1": "roll", "z.1": "yaw",
                      "x.2": "u_x", "y.2": "u_y", "x.3": "roll_der", "z.3": "yaw_der", "secs": "time"}
    df_state.rename_cols(columns_mapper)
    cols = columns_mapper.values()
    df_state.extract_cols(cols)
    size = len(df_state.df)
    cols.remove("time")
    df_state.resample_data(size, up, down, cols, make_plots=True)
    df_state.df.to_csv("df_state.csv")

    # control data
    df_ctrls = DataClass("rosbag_files/dynamics_model/chassisState.csv")
    columns_mapper = {"secs": "time"}
    df_ctrls.rename_cols(columns_mapper)
    cols = ["steering", "throttle", "time"]
    df_ctrls.extract_cols(cols)
    cols.remove("time")
    df_ctrls.resample_data(size, size * up / down, len(df_ctrls.df), cols, make_plots=True)
    # truncate throttle and steering signal
    df_ctrls.trunc(cols, make_plots=True)

    # merge control and state data
    final = pd.concat([df_ctrls.df, df_state.df], axis=1)
    # truncate the first few and last few columns to remove the resampling spikes
    d1 = np.arange(0, 10)
    d2 = np.arange(len(final)-10, len(final))
    final = final.drop(np.concatenate((d1, d2)))

    # produce acceleration data

    final.to_csv("final.csv", index=False)

