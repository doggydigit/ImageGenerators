from occlude import *


def main(level=4, dataset='cifar'):

    minsz, maxsz = getlevel(level, dataset)
    if dataset is 'cifar':
        images, labels = load_cifar10()
        occluded_imgs, masks = occlude(images, minsz, maxsz)
        with open('occluded_cifar10_level' + str(level) + '.pickle', 'wb') as handle:
            pickle.dump((images, occluded_imgs, masks), handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif dataset is 'imagenet':
        for batch in range(20):
            images, labels, boxes = load_imagenet(batch)
            occluded_imgs, masks = occlude_imagenet(images, minsz, maxsz)
            with open('occluded_imagenet_batch' + str(batch) + '_level' + str(level) + '.pickle', 'wb') as handle:
                pickle.dump((images, occluded_imgs, masks, labels, boxes), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Saved Imagenet Batch ' + str(batch))
    else:
        raise NotImplementedError


if __name__ == "__main__":
    for i in range(6):
        main(i, 'imagenet')
    # main(0, 'imagenet')