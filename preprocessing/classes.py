#!/usr/bin/env python
from __future__ import print_function
import data as dat
import numpy as np
import argparse
from sklearn import preprocessing

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--labels_path', help='Labels File', required=True)
ARGS = parser.parse_args()

print('\nLoading labels...')
f = np.genfromtxt(ARGS.labels_path, delimiter=',', dtype=str)

#Removing first line and column
f = f[1:,1]

labels = []
for i in range(len(f)):
    labels.append(f[i].split(' '))

# Flatten list of lists
flatten = [item for sublist in labels for item in sublist]

# Get array of classes
classes = np.unique(flatten)

# Create counter to classes
counter = np.zeros(len(classes), dtype='int')

# Couting classes
for i in range(len(classes)):
    c = classes[i]
    count = 0
    for item in flatten:
        if item == c:
            count += 1
    counter[i] = count

# Create dict with indexes and counters
counter = dict(zip(np.arange(len(counter)), counter))

# Sort by counting
counter = sorted(counter.items(), reverse=True, key=lambda x: x[1])

print('Single class counter\n')

# Printing results
for item in counter:
    print(classes[item[0]], item[1])

# Conting multi classes

#Generating label composed by a number
le = preprocessing.LabelEncoder()
le.fit(labels)

# Transform labels into encode
labels_enc = le.transform(labels)

# Create counter to classes
counter = np.zeros(len(le.classes_), dtype='int')

# Counting multi classes
for c in range(len(le.classes_)):
    counter[c] = len(np.where(labels_enc == c)[0])

# Create dict with indexes and counters
counter = dict(zip(np.arange(len(counter)), counter))

# Sort by counting
counter = sorted(counter.items(), reverse=True, key=lambda x: x[1])

print('Multi class counter')

# Printing results
for item in counter:
    print(le.classes_[item[0]], item[1])