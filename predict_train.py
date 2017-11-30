import h5py
import sys
import os
import argparse
import numpy as np

from utils.scores import fbeta
from utils.dataset import load_dataset
from utils.metadata import get_metadata_paths
from utils.optimizers import optimizer_factory
from utils.configuration import load_configuration

from models.factory import model_factory

from keras.callbacks import TensorBoard


def main(argv):
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='Output file', required=True)
    ARGS = parser.parse_args()

    CONFIG = load_configuration(ARGS.config_file)

    dataset = load_dataset(CONFIG)


    # Get variables
    model_name = CONFIG.network.parameters.model_name
    img_rows = int(CONFIG.dataset.parameters.height)
    img_cols = int(CONFIG.dataset.parameters.width)
    channel = int(CONFIG.dataset.parameters.channels)
    num_classes =  int(CONFIG.network.parameters.n_labels)
    dropout_keep_prob = float(CONFIG.network.parameters.dropout_keep_prob)
    label_treshold = float(CONFIG.prediction.threshold)


    # Loading labels
    print("Loading labels")
    with open(CONFIG.prediction.class_encoded_path, 'r') as reader:
        labels = np.array(reader.read().split(",")[:-1])

    train_filenames = []

    # Loading filenames
    print("Load filenames")
    lists_path  = "{}/{}".format(CONFIG.dataset.original.path, CONFIG.dataset.folders.list_folder)

    with open("{}/{}".format(lists_path, CONFIG.dataset.lists.test), 'r') as lines:
        for l in lines:
            train_filenames.append(l.split('.')[0])
    train_filenames = np.array(train_filenames)

    # Load model
    model = model_factory(model_name, img_rows, img_cols, channel, num_classes, dropout_keep_prob, checkpoint=CONFIG.prediction.checkpoint)

    test_Y = model.predict(dataset["train"]["X"])
    test_Y_ = np.where(test_Y>=label_treshold, 1, 0)

    # Writing prediction file
    with open(CONFIG.prediction.predict_file, 'w') as writer, open(CONFIG.prediction.prob_file, 'w') as prob_writer:
        writer.write("image_name,tags\n")
        for name, label, label_dec in zip(train_filenames, test_Y_, test_Y):
            # Write name
            writer.write("{},".format(name))
            prob_writer.write("{},".format(name))

            if label.sum() < 1:
                print(name, labels[np.where(label_dec > 0.1)])

            # Write label names
            file_labels = labels[np.where(label==1)]
            for i in range(file_labels.shape[0]):
                if file_labels.shape[0] - 1 == i:
                    writer.write(file_labels[i])
                else:
                    writer.write(file_labels[i] + " ")
            # write label probabilities
            for i in range(label_dec.shape[0]):
                if label_dec.shape[0] - 1 == i:
                    prob_writer.write(str(label_dec[i]))
                else:
                    prob_writer.write(str(label_dec[i]) + " ")


            writer.write("\n")
            prob_writer.write("\n")

    # print(test_Y)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))