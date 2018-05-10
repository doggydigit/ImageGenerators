from occlude import *


def main(level=4, dataset='cifar', small=True):

    minsz, maxsz = getlevel(level, dataset)
    if dataset is 'cifar':
        images, labels = load_cifar10()
        occluded_imgs, masks = occlude(images, minsz, maxsz)
        with open('occluded_cifar10_level' + str(level) + '.pickle', 'wb') as handle:
            pickle.dump((images, occluded_imgs, masks), handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif dataset is 'imagenet':
        for batch in range(20,21):
            images, labels, boxes = load_imagenet(batch, small)
            occluded_imgs, masks = occlude_imagenet(images, minsz, maxsz)
            if small:
                with open('occluded_imagenet_batch' + str(batch) + '_level' + str(level) + '.pickle', 'wb') as handle:
                    pickle.dump((images, occluded_imgs, masks, labels, boxes), handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                np.savez_compressed('occluded_cropped_imagenet_batch' + str(batch) + '_level' + str(level),
                                    images=images, occluded_imgs=occluded_imgs, masks=masks,labels=labels, boxes=boxes)
            print('Saved Imagenet Batch ' + str(batch))
    else:
        raise NotImplementedError


if __name__ == "__main__":
    # for i in range(5, 9):
    #     main(i, 'imagenet', False)
    main(6, 'imagenet', False)
