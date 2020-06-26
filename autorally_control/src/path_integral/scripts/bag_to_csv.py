"""
Written by Nick Speal in May 2013 at McGill University's Aerospace Mechatronics Laboratory
Modified by Rodrigue de Schaetzen in June 2020
"""

import os
import shutil
import string
from tqdm import tqdm
import csv
import rosbag
import sys


def extract_bag_to_csv(bag_file):
	"""
	Create a csv file for each topic in the rosbag
	:param bag_file: path to rosbag file
	"""
	print("Reading file %s..." % bag_file)
	# access bag
	bag = rosbag.Bag(bag_file)
	bag_contents = bag.read_messages()
	bag_name = bag.filename

	# create a new directory
	folder = string.rstrip(bag_name, ".bag")
	if not os.path.exists(folder):
		os.makedirs(folder)
	shutil.copyfile(bag_name, folder + '/' + bag_name)

	# get list of topics from the bag
	list_of_topics = []
	for topic, msg, t in bag_contents:
		if topic not in list_of_topics:
			list_of_topics.append(topic)

	for topic_name in list_of_topics:
		# Create a new CSV file for each topic
		filename = folder + '/' + string.replace(topic_name, '/', '') + '.csv'
		with open(filename, 'w+') as csvfile:
			file_writer = csv.writer(csvfile, delimiter=',')
			first_iter = True  # allows header row
			for subtopic, msg, t in tqdm(bag.read_messages(topic_name)):  # for each instant in time that has data for topic_name
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
	bag.close()
	print("Done reading %s" % bag_file)


if __name__ == '__main__':
	# verify correct input arguments
	if len(sys.argv) > 2:
		print("invalid number of arguments:   " + str(len(sys.argv)))
		print("requires 'bagName'")
		sys.exit(1)
	elif len(sys.argv) == 2:
		bag_file = sys.argv[1]
		extract_bag_to_csv(bag_file)
	else:
		print("bad argument(s): " + str(sys.argv))
		sys.exit(1)
