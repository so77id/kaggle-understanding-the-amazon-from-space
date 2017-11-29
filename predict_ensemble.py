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

    dataset = load_dataset(CONFIG, train=False)


    # Get variables
    model_name_1 = CONFIG.network.parameters.model_name_1
    model_name_2 = CONFIG.network.parameters.model_name_2
    model_name_3 = CONFIG.network.parameters.model_name_3
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

    test_filenames = []

    # Loading filenames
    print("Load filenames")
    lists_path  = "{}/{}".format(CONFIG.dataset.original.path, CONFIG.dataset.folders.list_folder)

    with open("{}/{}".format(lists_path, CONFIG.dataset.lists.test), 'r') as lines:
        for l in lines:
            test_filenames.append(l.split('\n')[0])
    test_filenames = np.array(test_filenames)

    test_add_filenames = []
    with open("{}/{}".format(lists_path, CONFIG.dataset.lists.test_add), 'r') as lines:
        for l in lines:
            test_add_filenames.append(l.split('\n')[0])
    test_add_filenames = np.array(test_add_filenames)


    # Load model
    model_1 = model_factory(model_name_1, img_rows, img_cols, channel, num_classes, dropout_keep_prob, checkpoint=CONFIG.prediction.checkpoint_1)
    model_2 = model_factory(model_name_2, img_rows, img_cols, channel, num_classes, dropout_keep_prob, checkpoint=CONFIG.prediction.checkpoint_2)
    model_3 = model_factory(model_name_3, img_rows, img_cols, channel, num_classes, dropout_keep_prob, checkpoint=CONFIG.prediction.checkpoint_3)

    # Get test labels
    print("Test Shape:", dataset["test"]["X"].shape)
    print("Test add Shape:", dataset["test"]["X_add"].shape)

    test_Y_1 = model_1.predict(dataset["test"]["X"])
    test_Y_2 = model_2.predict(dataset["test"]["X"])
    test_Y_3 = model_3.predict(dataset["test"]["X"])


    test_Y_1 = np.where(test_Y_1>=label_treshold, 1, 0)
    test_Y_2 = np.where(test_Y_2>=label_treshold, 1, 0)
    test_Y_3 = np.where(test_Y_3>=label_treshold, 1, 0)

    final_Y=[]
    for idx in range(len(test_Y_1)):
        tags=[]
        for yy in range(len(test_Y_1[idx])):
            preds=[test_Y_1[idx][yy],test_Y_2[idx][yy],test_Y_3[idx][yy]]
            ensemble_soft_prediction=np.argmax(np.bincount(preds))
            tags.append(ensemble_soft_prediction)
        ensemble_soft_prediction.append(tags)

    test_Y = ensemble_soft_prediction

    # Get test_add labels
    #test_add_Y = model.predict(dataset["test"]["X_add"])
    #test_add_Y = np.where(test_add_Y>=label_treshold, 1, 0)

    with open(CONFIG.prediction.predict_file, 'w') as writer:
        j = 0
        for name, label in zip(test_filenames, test_Y):
            writer.write("{},".format(name))
            file_labels = labels[np.where(label==1)]
            j =+ 1
            for i in range(file_labels.shape[0]):
                if file_labels.shape[0] - 1 == i:
                    writer.write(file_labels[i] + "\n")
                else:
                    writer.write(file_labels[i] + " ")

        """k = 0
        for name, label in zip(test_add_filenames, test_add_Y):
            writer.write("{},".format(name))
            file_labels = labels[np.where(label==1)]
            k += 1
            for i in range(file_labels.shape[0]):
                if file_labels.shape[0] - 1 == i:
                    writer.write(file_labels[i] + "\n")
                else:
                    writer.write(file_labels[i] + " ")"""


    print("test count:", j)
    print("test add count:", k)

    # print(test_Y)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
