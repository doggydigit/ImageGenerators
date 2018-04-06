import os
import pickle
import warnings
import numpy as np


def unpickle(file: object) -> object:

    with open(file, 'rb') as fo:
        cif = pickle.load(fo, encoding='bytes')
    return cif


def getlevel(level):
    levels = [()] * 6
    levels[0] = (2, 4)
    levels[1] = (4, 6)
    levels[2] = (6, 8)
    levels[3] = (8, 10)
    levels[4] = (10, 12)
    levels[5] = (12, 14)
    return levels[level]


def load_cifar10():
    curdir = os.path.dirname(os.path.abspath(__file__))
    data = unpickle(curdir + '/../cifar/cifar-10-batches-py/data_batch_1')
    x = data[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    y = data[b'labels']
    # Load the rest of the images
    for i in range(2, 6):
        data = unpickle(curdir + '/../cifar/cifar-10-batches-py/data_batch_' + str(i))
        x = np.concatenate((x, data[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")))
        y = np.concatenate((y, data[b'labels']))
    # warnings.warn("You are only working with 1 of 5 batches of the CIFAR-10 dataset", FutureWarning)

    return x, y


def load_cifar100():
    raise NotImplementedError
