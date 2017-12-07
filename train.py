import h5py
import sys
import os
import argparse

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
    batch_size = int(CONFIG.network.parameters.batch_size)
    n_epoch = int(CONFIG.network.parameters.n_epoch)


    # Get paths
    metadata_path, checkpoint_path, logs_path = get_metadata_paths(CONFIG, ARGS)


    # Load model

    model = model_factory(model_name, img_rows, img_cols, channel, num_classes, dropout_keep_prob)

    # Loading optimizer
    optimizer = optimizer_factory(CONFIG.network.parameters.optimizer)

    # Creating trainner
    model.compile(optimizer=optimizer, loss=CONFIG.network.parameters.loss_function, metrics=[fbeta])


    # Fit model
    tensorboard = TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=False)

    model.fit(dataset["train"]["X"], dataset["train"]["Y"],
                  batch_size=batch_size,
                  epochs=n_epoch,
                  shuffle=True,
                  verbose=1,
                  callbacks=[tensorboard],
                  validation_data=(dataset["val"]["X"], dataset["val"]["Y"]),
              )
    # Save model
    model.save_weights(checkpoint_path)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))