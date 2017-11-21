import cv2
import numpy as np

import h5py
import argparse
import sys
import json

from collections import namedtuple



def load_configuration(configuration_file):
    with open(configuration_file, 'r') as content_file:
        content = content_file.read()

    return json.loads(content, object_hook=lambda d: namedtuple('Configuration', d.keys())(*d.values()))


def main(argv):
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='Output file', required=True)
    ARGS = parser.parse_args()

    CONFIG = load_configuration(ARGS.config_file)

    # Get folders
    dataset_path = CONFIG.dataset.original.path if CONFIG.dataset.parameters.dataset_in_use == 'original' else CONFIG.dataset.processed.path
    lists_path = "{}/{}".format(dataset_path, CONFIG.dataset.folders.list_folder)
    train_path = "{}/{}".format(dataset_path, CONFIG.dataset.folders.train_folder)
    test_path = "{}/{}".format(dataset_path, CONFIG.dataset.folders.test_folder)
    test_add_path = "{}/{}".format(dataset_path, CONFIG.dataset.folders.test_add_folder)



    # Creating trainning dataset
    print("Creating trainning dataset")
    train_list_filename = "{}/{}".format(lists_path, CONFIG.dataset.lists.train)

    train_filename_lists = []
    train_labels_lists = []

    with open(train_list_filename, 'r') as input_file:
        for line in input_file:
            filename, labels = line.split(",")

            filename = "{}/{}.jpg".format(train_path, filename)
            labels_list = labels.split('\n')[0].split(" ")

            train_filename_lists.append(filename)
            train_labels_lists.append(filename)

    # hdf5_train_file = "{}/{}".format(CONFIG.dataset.h5.path, CONFIG.dataset.h5.train)

    # h5f_t = h5py.File(hdf5_train_file, 'w')
    # h5f_t.create_dataset('X', (num_lines,10,Resize,Resize,3))


    print("hola")

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))