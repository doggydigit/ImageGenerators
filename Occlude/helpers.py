import os
import pickle
import warnings
import numpy as np


def unpickle(file: object) -> object:
    with open(file, 'rb') as fo:
        cif = pickle.load(fo, encoding='bytes')
    return cif


def getlevel(level, dataset):
    if dataset is 'cifar':
        return 2 + 2 * level, 2 + 2 * (level + 1)
    elif dataset is 'coco':
        return 10 + 50 * level, 10 + 50 * (level + 1)
    elif dataset is 'imagenet':
        # return 10 + 50 * level, 10 + 50 * (level + 1)
        return 0.05 + 0.05 * level, 0.05 + 0.05 * (level + 1)
    else:
        raise NotImplementedError


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


def load_imagenet(batch=0, small=True):
    curdir = os.path.dirname(os.path.abspath(__file__))
    if small:
        images, labels, boxes = unpickle(curdir + '/../Imagenet/Images/Batches/imagenet_batch' + str(batch) + '.pickle')
        images = np.array(images)
        return images, labels, boxes
    else:
        batch = np.load(curdir + '/../Imagenet/Images/Batches/cropped_imagenet_batch' + str(batch) + '.npz')
        return batch['images'], batch['labels'], batch['boxes']