import cv2
import os
import math
import pickle
import numpy as np
import xml.etree.ElementTree as ET

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
            root = ET.parse(anno).getroot()
            try:
                box = root.find('object').find('bndbox')
            except AttributeError as error:
                '-.-'
            else:
                sz = [int(a.text) for a in root.find('size')]
                rat = sz[0]/sz[1]
                if min(sz) > 200 and 0.5 < rat < 2.0:
                    img_dict[nr_img] = [wnid, img]
                    anno_dict[nr_img] = [int(a.text) for a in box]
                    nr_img = nr_img + 1

#
# Shuffle the ids to mix up all the iamges with different labels
#
ids = np.array(range(nr_img))
np.random.shuffle(ids)

#
# Load Images and Save batches of them together with their Boudning Boxes and Labels
#
batch_size = 2500
for batch in range(math.floor(nr_img/batch_size)):

    # Load Images and gather data into lists
    images = [None] * batch_size
    labels = [None] * batch_size
    annos = [None] * batch_size
    for i in range(batch_size):
        j = ids[batch * batch_size + i]
        handle = img_dict[j]
        images[i] = cv2.imread(datadir + handle[0] + '/' + handle[1])
        labels[i] = wnids_dict[handle[0]]
        annos[i] = anno_dict[j]

    # Save the lists to pickel files
    print('Saving Batch ' + str(batch))
    with open('imagenet_batch' + str(batch) + '.pickle', 'wb') as handle:
        pickle.dump((images, labels, annos), handle, protocol=pickle.HIGHEST_PROTOCOL)
