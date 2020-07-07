"""
Partially written by Nick Speal in May 2013 at McGill University's Aerospace Mechatronics Laboratory
Modified by Rodrigue de Schaetzen in June 2020
Reorder bag code from http://wiki.ros.org/rosbag/Cookbook
"""

import csv
import math
import os
import shutil
import string
import subprocess
import sys
import time
import yaml
import argparse

import rosbag
import rospy


def status(percent, length=40):
    """
    Helper to indicate progress in stdout
    :param percent: Progress percentage
    :param length: The length of the progress bar
    """
    sys.stdout.write('\x1B[2K')  # Erase entire current line
    sys.stdout.write('\x1B[0E')  # Move to the beginning of the current line
    progress = "Progress: ["
    for i in range(0, length):
        if i < length * percent:
            progress += '='
        else:
            progress += ' '
    progress += "] " + str(round(percent * 100.0, 2)) + "%"
    sys.stdout.write(progress)
    sys.stdout.flush()


def reorder_bag(bag_file, max_offset=0):
    """
    Reorders the bagfile based on header timestamps
    :param bag_file: The path of the bag
    :param max_offset: The maximum allowed offset between header time and rosbag time
    """
    print("Reordering bag file '%s' based on header timestamps..." % bag_file)
    # Get bag duration
    info_dict = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', bag_file], stdout=subprocess.PIPE).communicate()[0])
    duration = info_dict['duration']
    start_time = info_dict['start']

    # save the orig bag file
    orig = os.path.splitext(bag_file)[0] + ".orig.bag"
    shutil.move(bag_file, orig)

    with rosbag.Bag(bag_file, 'w') as outbag:

        last_time = time.clock()
        for topic, msg, t in rosbag.Bag(orig).read_messages():

            if time.clock() - last_time > .1:
                percent = (t.to_sec() - start_time) / duration
                status(percent)
                last_time = time.clock()

            # This also replaces tf timestamps under the assumption
            # that all transforms in the message share the same timestamp
            if topic == "/tf" and msg.transforms:
                # Writing transforms to bag file 1 second ahead of time to ensure availability
                diff = math.fabs(msg.transforms[0].header.stamp.to_sec() - t.to_sec())
                outbag.write(topic, msg, msg.transforms[0].header.stamp - rospy.Duration(1) if diff < max_offset else t)
            elif msg._has_header:
                diff = math.fabs(msg.header.stamp.to_sec() - t.to_sec())
                outbag.write(topic, msg, msg.header.stamp if diff < max_offset else t)
            else:
                outbag.write(topic, msg, t)
    status(1)
    print("\ndone")


def extract_bag_to_csv(bag_file, topics='all'):
    """
    Creates a csv file for each specified topic
    :param bag_file: path to rosbag file
    :param topics: topics of interest to be extracted as csv, if 'all' then extract all topics
    :return topic_files: Returns dict {extracted_topic_name: topic_csv_file_path}
    """
    print("Reading file %s..." % bag_file)
    # access bag
    bag = rosbag.Bag(bag_file)
    bag_name = bag.filename

    # create a new directory
    folder = 'rosbag_files/' + string.rstrip(bag_name, ".bag")
    if not os.path.exists(folder):
        os.makedirs(folder)
    shutil.copyfile(bag_file, folder + '/' + bag_name)

    # get list of topics from the bag
    info_dict = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', bag_file], stdout=subprocess.PIPE).communicate()[0])
    all_topics = {}
    for item in info_dict["topics"]:
        all_topics[item["topic"]] = item["messages"]  # store topic names as keys and number of messages as values

    if topics != 'all':
        # make sure topics of interest are in all topics
        for topic_name in topics:
            if topic_name not in all_topics.keys():
                print("\nWARNING: topic '%s' is not part of the list of recorded topics in bagfile '%s'" % (topic_name, bag_file))

    # dict to store topic file paths
    topic_files = {}

    for topic_name in all_topics.keys():
        # check if current topic is part of the topics of interest
        if topics != 'all' and topic_name not in topics:
            # skip this current topic
            print("Skipping topic '%s'..." % topic_name)
            continue

        print("Extracting topic '%s'..." % topic_name)
        # Create a new CSV file for each topic
        filename = folder + '/' + string.replace(topic_name, '/', '') + '.csv'
        topic_files[topic_name] = filename

        with open(filename, 'w+') as csvfile:
            file_writer = csv.writer(csvfile, delimiter=',')
            first_iter = True  # allows header row
            # variable to keep track of number of parsed messages
            msg_count = 0.
            for subtopic, msg, t in bag.read_messages(topic_name):  # for each instant in time that has data for topic_name
                msg_count += 1.
                percent = msg_count / all_topics[topic_name]
                status(percent)
                # parse data from this instant, which is of the form of multiple lines of "Name: value\n" put it in the form of a list of 2-element lists
                msg_string = str(msg)
                msg_list = string.split(msg_string, '\n')
                instantaneous_list_of_data = []
                for nameValuePair in msg_list:
                    split_pair = string.split(nameValuePair, ':')
                    for i in range(len(split_pair)):  # should be 0 to 1
                        split_pair[i] = string.strip(split_pair[i])
                    instantaneous_list_of_data.append(split_pair)
                # write the first row from the first element of each pair
                if first_iter:  # header
                    headers = ["rosbagTimestamp"]  # first column header
                    for pair in instantaneous_list_of_data:
                        headers.append(pair[0])
                    file_writer.writerow(headers)
                    first_iter = False
                # write the value from each pair to the file
                values = [str(t)]  # first column will have rosbag timestamp
                for pair in instantaneous_list_of_data:
                    values.append(pair[1])
                file_writer.writerow(values)
            print("\n")
    bag.close()
    print("Done reading %s\n" % bag_file)
    return topic_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bagfile', nargs=1, help='input bag file')
    parser.add_argument('--max-offset', nargs=1, help='max time offset (sec) to correct.', default='0', type=float)
    args = parser.parse_args()

    # reorder bag file
    reorder_bag(args.bagfile[0], max_offset=args.max_offset)

    # extract rosbag to csv file for each topic
    extract_bag_to_csv(args.bagfile[0])
