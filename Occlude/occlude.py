import matplotlib.patches as mp
import skimage.draw as skd
from helpers import *


def add_occluder(img, size, cx, cy, shape=None):
    if shape is None:
        i = np.random.randint(4)
    else:
        shape_dict = {'circle': 0, 'square': 1, 'triangle': 2, 'invtri': 3}
        try:
            i = shape_dict[shape]
        except KeyError:
            raise KeyError('The possible shapes are: circle, square, triangle and invtri')
    ishape = img.shape[0:2]
    oi = img
    if i is 0:
        a, b = skd.circle(cx, cy, size)
        #print('circle')
    elif i is 1:
        a, b = skd.polygon([cx-size, cx+size, cx+size, cx-size], [cy-size, cy-size, cy+size, cy+size], shape=ishape)
        #print('rectangle')
    elif i is 2:
        long = cy + int(1.1547 * size)
        short = long - int(size * 1.7320508)
        a, b = skd.polygon([short, short, long], [cx - size, cx + size, cx], shape=ishape)
        #print('triangle')
    elif i is 3:
        long = cy - int(1.1547 * size)
        short = long + int(size * 1.7320508)
        a, b = skd.polygon([short, short, long], [cx - size, cx + size, cx], shape=ishape)
    else:
        raise NotImplementedError(' There are only 4 possible occluder shapes implemented: '
                                  'Circle (0), Square (1), Triangle Down (2) and Triangle Up (3)')

    msk = np.ones(ishape, dtype=np.uint8)
    oi[a, b, :] = 0
    msk[a, b] = 0

    return oi, msk


def occlude(imgs, minsz, maxsz):
    imsizes = imgs.shape
    occ_imgs = np.copy(imgs)
    msks = np.zeros((imsizes[0], imsizes[1], imsizes[2]), dtype=np.uint8)
    for i in range(imsizes[0]):
        radius = np.random.randint(minsz, maxsz)
        cx = np.random.randint(radius, imsizes[1]-radius)
        cy = np.random.randint(radius, imsizes[2]-radius)
        occ_imgs[i, :, :, :], msks[i, :, :] = add_occluder(occ_imgs[i, :, :, :], radius, cx, cy)
    return occ_imgs, msks
