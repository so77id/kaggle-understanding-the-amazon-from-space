import cv2
import numpy as np

import h5py
import argparse
import sys
import os
import json

from collections import namedtuple

import keras
from keras.utils import np_utils


from sklearn import preprocessing

def get_h5_filenames(h5_path, h5_CONFIG, img_size, extension=".h5"):
    hdf5_train_file = "{}/{}-{}.{}".format(h5_path, h5_CONFIG.train_pattern, img_size, extension)
    hdf5_test_file = "{}/{}-{}.{}".format(h5_path, h5_CONFIG.test_pattern, img_size, extension)
    hdf5_test_add_file = "{}/{}-{}.{}".format(h5_path, h5_CONFIG.test_add_pattern, img_size, extension)

    return hdf5_train_file, hdf5_test_file, hdf5_test_add_file


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

    h5_path = "{}/{}".format(dataset_path, CONFIG.dataset.folders.h5_folder)

    print(h5_path)
    if not os.path.exists(h5_path):
        os.makedirs(h5_path, exist_ok=True)

    hdf5_train_file, hdf5_test_file, hdf5_test_add_file = get_h5_filenames(h5_path, CONFIG.dataset.h5, CONFIG.dataset.parameters.width, extension="h5")

    # Get varaibles
    width = CONFIG.dataset.parameters.width
    height = CONFIG.dataset.parameters.height
    channels = CONFIG.dataset.parameters.channels

    # Creating train dataset
    print("Creating train dataset")
    train_list_filename = "{}/{}".format(lists_path, CONFIG.dataset.lists.train)

    train_filename_lists = []
    train_labels_lists = []

    with open(train_list_filename, 'r') as input_file:
        for line in input_file:
            filename, labels = line.split(",")

            filename = "{}/{}.jpg".format(train_path, filename)
            labels_list = labels.split('\n')[0].split(" ")

            train_filename_lists.append(filename)
            train_labels_lists.append(labels_list)

    # processing labels
    list_labels = [item for sublist in train_labels_lists for item in sublist]
    encoder = preprocessing.LabelEncoder()
    encoder.fit(list_labels)
    n_labels = encoder.classes_.shape[0]

    labels = encoder.classes_
    class_enconded_file = CONFIG.prediction.class_encoded_path
    with open(class_enconded_file, "w") as writer:
        for label in labels:
            writer.write(label + ",")

    train_categorical_labels = []
    for label_per_file in train_labels_lists:
        encoded_labels = encoder.transform(label_per_file)
        categorical_labels = np_utils.to_categorical(encoded_labels, n_labels).sum(axis=0)
        train_categorical_labels.append(categorical_labels)

    # creating h5 files
    # hdf5_train_file = "{}/{}".format(h5_path, CONFIG.dataset.h5.train_file)

    h5f_t = h5py.File(hdf5_train_file, 'w')

    img_list = []
    for filename in train_filename_lists:
        print("Processing:", filename)
        img = cv2.imread(filename)
        img = cv2.resize(img, (width, height))

        img_list.append(img)

    h5f_t.create_dataset('X', data=img_list)
    h5f_t.create_dataset('Y', data=train_categorical_labels)

    h5f_t.close()

    # Creating test dataset
    print("Creating test dataset")
    test_list_filename = "{}/{}".format(lists_path, CONFIG.dataset.lists.test)

    test_filename_lists = []

    with open(test_list_filename, 'r') as input_file:
        for line in input_file:
            filename = line.split('\n')[0]

            filename = "{}/{}".format(test_path, filename)

            test_filename_lists.append(filename)

    # creating h5 files
    # hdf5_test_file = "{}/{}".format(h5_path, CONFIG.dataset.h5.test_file)

    h5f_t = h5py.File(hdf5_test_file, 'w')

    img_list = []
    for filename in test_filename_lists:
        print("Processing:", filename)
        img = cv2.imread(filename)
        img = cv2.resize(img, (width, height))

        img_list.append(img)

    img_list = np.array(img_list)

    h5f_t.create_dataset('X', data=img_list)

    h5f_t.close()

    # Creating test add dataset
    print("Creating test add dataset")
    test_add_list_filename = "{}/{}".format(lists_path, CONFIG.dataset.lists.test_add)

    test_add_filename_lists = []

    with open(test_add_list_filename, 'r') as input_file:
        for line in input_file:
            filename = line.split('\n')[0]

            filename = "{}/{}".format(test_add_path, filename)

            test_add_filename_lists.append(filename)

    # creating h5 files
    # hdf5_test_add_file = "{}/{}".format(h5_path, CONFIG.dataset.h5.test_add_file)

    h5f_t = h5py.File(hdf5_test_add_file, 'w')

    img_list = []
    for filename in test_add_filename_lists:
        print("Processing:", filename)
        img = cv2.imread(filename)
        img = cv2.resize(img, (width, height))

        img_list.append(img)

    h5f_t.create_dataset('X', data=img_list)

    h5f_t.close()



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))