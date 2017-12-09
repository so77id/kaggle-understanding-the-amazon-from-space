import os
from numpy import array
from numpy import genfromtxt
from PIL import Image
from sklearn import preprocessing

def resize_images(path, new_path, size):
    imlist = os.listdir(path)
    print('\nResizing ' + str(len(imlist)) + ' images...')
    for file in imlist:
        orig = Image.open(path + '/' + file)
        small = orig.resize((size,size))
        small.save(new_path +  file, "JPEG")

def load_images(path):
    imlist = os.listdir(path)
    print('\nLoading ' + str(len(imlist)) + ' images...')

    data = array([array(Image.open(path + im)).flatten()
                 for im in imlist], dtype='float32', order='f')

    return data

def load_labels(path):
    print('\nLoading labels...')
    f = genfromtxt(path, delimiter=',', dtype=str)

    #Removing first line and column
    f = f[1:,1]

    #Taking the first word of the labels > considering just one class for each data
    for i in range(len(f)):
        f[i] = f[i].split(' ', 1)[0]

    f_min = list(set(f))
    f_min = array(f_min)
    print(str(f_min.size) + ' classes.')

    #Generating label composed by a number
    le = preprocessing.LabelEncoder()
    le.fit(f_min)
    label = le.transform(f)

    # Save a txt with this numbers?
    return label