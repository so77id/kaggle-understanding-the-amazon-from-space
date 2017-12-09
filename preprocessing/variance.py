#!/usr/bin/env python
from __future__ import print_function
from sklearn.decomposition import PCA
import data as dat
import time
import argparse
from sklearn.preprocessing import scale, normalize

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--path_train', help='Training File', required=True)
ARGS = parser.parse_args()

#Loading images
start_time = time.time()
train_data = dat.load_images(ARGS.path_train)
print('Loaded in ' + str(time.time()-start_time) + 's')

# Normalizing
train_data = train_data/255
train_data = scale(train_data)
train_data = normalize(train_data, norm = 'l2')

print('Starting PCA')
# Use PCA with all components to get variance values
start_time = time.time()
pca = PCA(n_components = 89401)
pca.fit_transform(train_data)
variances = pca.explained_variance_ratio_
print('PCA in ' + str(time.time()-start_time) + 's')

# Suming values for number of components
for i in range(1, len(variances)):
    variances[i] = variances[i] + variances[i-1]
    print(i, variances[i])
