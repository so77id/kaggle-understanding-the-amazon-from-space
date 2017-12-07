import h5py
import numpy as np
import math
from preprocessing.create_h5_files import get_h5_filenames


def load_dataset(CONFIG, train=True):
    dataset_path = CONFIG.dataset.original.path if CONFIG.dataset.parameters.dataset_in_use == 'original' else CONFIG.dataset.processed.path
    h5_path = "{}/{}".format(dataset_path, CONFIG.dataset.folders.h5_folder)

    hdf5_train_file, hdf5_test_file, hdf5_test_add_file = get_h5_filenames(h5_path, CONFIG.dataset.h5, CONFIG.dataset.parameters.width, extension="h5")

    # hdf5_train_file = "{}/{}".format(h5_path, CONFIG.dataset.h5.train_file)
    # hdf5_test_file = "{}/{}".format(h5_path, CONFIG.dataset.h5.test_file)
    # hdf5_test_add_file = "{}/{}".format(h5_path, CONFIG.dataset.h5.test_add_file)

    print(hdf5_train_file, hdf5_test_file, hdf5_test_add_file)

    if train:
        print("Loading train")
        h5f_train = h5py.File(hdf5_train_file, 'r')

        X = np.array(h5f_train['X'])
        Y = np.array(h5f_train['Y'], dtype=np.int)

        size = X.shape[0]
        train_x_size  = math.floor(size * 0.8)

        dataset = {"train":{"X":X[0:train_x_size], "Y":Y[0:train_x_size]}, "val":{"X":X[train_x_size:size], "Y":Y[train_x_size:size]}}

    else:
        print("Loading test")
        h5f_test = h5py.File(hdf5_test_file, 'r')
        print("Loading test add")
        h5f_test_add = h5py.File(hdf5_test_add_file, 'r')

        test_X = h5f_test['X']

        test_add_X = h5f_test_add['X']

        dataset = {"test":{"X":test_X, "X_add":test_add_X}}

    return dataset
