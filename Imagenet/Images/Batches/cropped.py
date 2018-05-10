import cv2
import os
import h5py
import math
import pickle
import numpy as np
import skimage.transform as sk
import xml.etree.ElementTree as et


#
# Initialize some stuff
#
datadir = '../Data/DET/train/ILSVRC2013_train/'
annodir = '../Annotations/DET/train/ILSVRC2013_train/'
np.random.seed(1002)
wnids_dict = {}
img_dict = {}
anno_dict = {}
i = 0
nr_img = 0
imsize = 320

#
# Load wnids, which are the ids of the labels and create a better id for them
#
wnids = [wnid for wnid in os.listdir(datadir) if wnid[0] is 'n']
for wnid in wnids:
    wnids_dict[wnid] = i
    i = i + 1

#
# Load Bounding Boxes and create dictionaries with image file names according to a new general image id
#
for wnid in wnids:
    for img in os.listdir(datadir + wnid + '/'):
        if img[-1] is 'G':
            anno = annodir + wnid + '/' + img[:-4] + 'xml'
            root = et.parse(anno).getroot()
            try:
                box = root.find('object').find('bndbox')
            except AttributeError as error:
                '-.-'
            else:
                if min([int(a.text) for a in root.find('size')]) > imsize:
                    img_dict[nr_img] = [wnid, img]
                    anno_dict[nr_img] = [int(a.text) for a in box]
                    nr_img = nr_img + 1

#
# Shuffle the ids to mix up all the iamges with different labels
#
ids = np.array(range(nr_img))
np.random.shuffle(ids)

xs = [0] * nr_img
ys = [0] * nr_img

#
# Load Images and Save batches of them together with their Boudning Boxes and Labels
#
batch_size = 2000
for batch in range(math.floor(nr_img/batch_size)):

    # Load Images and gather data into lists
    images = np.ones((batch_size, imsize, imsize, 3))
    labels = [None] * batch_size
    annos = [None] * batch_size
    for i in range(batch_size):
        j = ids[batch * batch_size + i]
        handle = img_dict[j]
        im = cv2.imread(datadir + handle[0] + '/' + handle[1])

        y, x, _ = im.shape
        bisquare = min(x, y)
        startx = x // 2 - (bisquare // 2)
        starty = y // 2 - (bisquare // 2)
        images[i, :, :, :] = sk.resize(im[starty:starty + bisquare, startx:startx + bisquare],
                                       output_shape=[imsize, imsize], mode='constant')
        labels[i] = wnids_dict[handle[0]]
        annos[i] = anno_dict[j]

    # Save the lists to pickel files
    print('Saving Batch ' + str(batch))
    # with open('cropped_imagenet_batch' + str(batch) + '.pickle', 'wb') as handle:
    #     pickle.dump((images, labels, annos), handle, protocol=pickle.HIGHEST_PROTOCOL)

    # h5f = h5py.File('cropped_imagenet_batch' + str(batch) + '.h5', 'w')
    # h5f.create_dataset('images', data=images)
    # h5f.create_dataset('labels', data=labels)
    # h5f.create_dataset('boxes', data=annos)
    # h5f.close()

    np.savez_compressed('cropped_imagenet_batch' + str(batch), images=images, labels=labels, boxes=annos)
