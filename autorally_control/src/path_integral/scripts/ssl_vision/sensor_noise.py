"""Script is written in python 3"""
import sslclient
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


# TODO: discuss values jumping around
# TODO: just connect to sslclient every time
# followed https://github.com/Juniorlimaivd/python-ssl-client for using sslclient
def get_ssl_measurements(client, measurement_count=100, robot_id=None, return_raw_values=True, **kwargs):
    """
    Gets measurements from ssl vision by using ssl client to parse messages
    NOTE: assumes robot on blue team
    :param client: instantiated ssl client object
    :param measurement_count: number of measurements to get from ssl vision
    :param robot_id: optional arg to specify the specific robot id interested in getting measurements
    :param return_raw_values: option to return the raw collected data
    """
    num_iter = 0
    hist_data = {'x': [], 'y': [], 'orientation': []}
    while num_iter < measurement_count:
        # received decoded package
        data = client.receive()

        if data.HasField('detection'):
            data = data.detection
            if data.robots_blue:
                for robot in data.robots_blue:
                    if (robot_id is not None and robot.robot_id == robot_id) or robot_id is None:
                        found_robot = robot.robot_id
                        print("found robot with robot_id %s" % str(found_robot))
                        print("measurement count {}".format(num_iter))
                        for i in ['x', 'y', 'orientation']:
                            hist_data[i].append(getattr(robot, i))
                        num_iter += 1
                if num_iter == 0 and robot_id is not None:
                    print("WARNING: no robot found with robot id %s..." % str(robot_id))

    if return_raw_values:
        return pd.DataFrame(hist_data)
    else:
        return [*kwargs.values(), np.mean(hist_data['x']), np.std(hist_data['x']), np.mean(hist_data['y']),
                np.std(hist_data['y']), np.mean(hist_data['orientation']), np.std(hist_data['orientation'])]


def plot_stationary_measurements_hist(dfs, measurement_count):
    """
    Plots a histogram of how precise each set of measurements were
    :param dfs: List of DataFrames containing the set of measurements
    :param measurement_count: Number of measurements taken for each set of measurements
    """
    centered_dfs = []
    for df in dfs:
        for c in df.columns:
            df[c] = df[c] - df[c].mean()  # centers mean at 0
            centered_dfs.append(df)
    final = pd.concat(centered_dfs)
    final.to_csv('stationary_robot_data.csv')
    fig = plt.figure()
    for idx, c in enumerate(final.columns):
        ax = fig.add_subplot(1, 3, idx + 1)
        ax.set_ylabel("frequency")
        ax.hist(final[c])
        ax.set_title('%s\n$\mu=%.06f$\n$\sigma=%0.6f$' % (c, final[c].mean(), final[c].std()))
    plt.suptitle("Histogram of robot measurements with mean centered at 0"
                 "\nx axis is the distance from mean (mm for x,y and rad for orientation)"
                 "\nMeasurement count per set %.0f, Number of sets of measurements %.0f" % (measurement_count, len(dfs)))
    plt.tight_layout()
    plt.savefig('stationary_robot_hist.png')


def compute_orientation_diffs(df, col_name='rotation'):  # TODO why not working, orientation not changing
    """
    Computes the difference in orientation between a set of measurements and the 'truth' rotation
    :param df: DataFrame containing the data
    :param col_name: name of column containing the rotations
    """
    diff = []
    for rotation, orientation in zip(df[col_name], df['orientation']):
        # check if rotation is 0 which means this is the start of a new measurement
        if rotation == 0:
            prev = orientation
        else:
            d = np.abs(orientation - prev)
            if d > np.pi:
                d -= np.pi
            diff.append(d - rotation)  # just care about the difference between ssl measurements and truth rotation
            prev = orientation

    # convert to np array
    diff = np.array(diff)
    plt.hist(diff, bins=10)
    plt.xlabel('orientation error (rad)')
    plt.ylabel('frequency')
    plt.title('Histogram of orientation difference\n$\mu=%.06f,\ \sigma=%0.6f$' % (np.mean(diff), np.std(diff)))
    plt.tight_layout()
    plt.savefig('orientation_diff.png')


def compute_euclid_dist_diffs(df, col_name='translation'):
    """
    Computes the difference in euclidean distance between a set of measurements and the 'truth' translation
    :param df: DataFrame containing the data
    :param col_name: name of column containing the translations
    """
    diff = []
    for translation, x, y in zip(df[col_name], df['x'], df['y']):
        # check if translation is 0 which means this is the start of a new measurement
        if translation == 0:
            prev = compute_euclid(x, y)
        else:
            curr = compute_euclid(x, y)
            d = np.abs(curr - prev)
            diff.append(
                d - translation)  # just care about the difference between ssl measurements and truth translation
            prev = curr

    # convert to np array
    diff = np.array(diff)
    plt.hist(diff, bins=10)
    plt.xlabel('translation error (mm)')
    plt.ylabel('frequency')
    plt.title('Histogram of translation difference\n$\mu=%.06f,\ \sigma=%0.6f$' % (np.mean(diff), np.std(diff)))
    plt.tight_layout()
    plt.savefig('translation_diff.png')


def compute_euclid(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def main():
    print("Chose mode '%s' and robot id '%s'...\n" % (str(args.mode), str(args.robot_id)))
    ssl_client = sslclient.client()
    print("ssl client trying to bind connection to port %s and IP %s..." % (ssl_client.port, ssl_client.ip))
    # Bind connection to port and IP for UDP Multicast
    ssl_client.connect()
    dfs = []
    data = []
    res = None
    num_iter = 0
    while res != "stop":
        print("\nIteration number %.0f" % num_iter)
        num_iter += 1
        if args.mode == 'stationary':
            res = input('\nPress enter for next set of measurements or type "stop" to stop taking measurements: ')
            if res == 'stop':
                break
            dfs.append(get_ssl_measurements(ssl_client, args.measurement_count, args.robot_id))
        elif args.mode == 'translation':
            res = input('\nEnter measured translation in mm, 0 if its a new set of measurements, '
                        'or type "stop" to stop taking measurements: ')
            if res == 'stop':
                break
            data.append(get_ssl_measurements(ssl_client, args.measurement_count, args.robot_id, return_raw_values=False,
                                             translation=float(res)))
        elif args.mode == 'rotation':
            res = input('\nEnter measured rotation as a fraction of pi (e.g. 0.5 if rotation is pi/2), '
                        '0 if its a new set of measurements, or type "stop" to stop taking measurements: ')
            if res == 'stop':
                break
            data.append(get_ssl_measurements(ssl_client, args.measurement_count, args.robot_id, return_raw_values=False,
                                             rotation=float(res) * np.pi))
    print("\nProcessing measurements and plotting results...")
    if args.mode == 'stationary':
        plot_stationary_measurements_hist(dfs, args.measurement_count)
    elif args.mode == 'rotation':
        df = pd.DataFrame(columns=['rotation', 'x', 'x_std', 'y', 'y_std', 'orientation', 'orientation_std'], data=data)
        df.to_csv('raw_orientation_diff_data.csv')
        compute_orientation_diffs(df)
    elif args.mode == 'translation':
        df = pd.DataFrame(columns=['translation', 'x', 'x_std', 'y', 'y_std', 'orientation', 'orientation_std'], data=data)
        df.to_csv('raw_translation_diff_data.csv')
        compute_euclid_dist_diffs(df)
    print("Done")
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes measurements from ssl vision and estimates the sensor noise.')
    parser.add_argument('-m', '--mode', type=str, nargs='?', choices=['translation', 'rotation', 'stationary'], default='stationary',
                        help='"translation" mode estimates the sensor noise from applying translations.'
                             '\n"rotation" mode estimates the sensor noise from applying rotations.'
                             '\n"stationary" mode estimates the sensor noise from taking stationary measurements.')
    parser.add_argument('-N', '--measurement_count', type=int, nargs='?', default=100, metavar='integer',
                        help='Number of measurements to get from ssl vision per iteration/set of measurements.')
    parser.add_argument('-r', '--robot-id', type=int, nargs='?', default=0, metavar='integer',
                        help='The robot id as seen in ssl vision.')
    args = parser.parse_args()
    main()
