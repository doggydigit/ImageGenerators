from occlude import *


def main(level=4):

    minsz, maxsz = getlevel(level)
    images, labels = load_cifar10()
    occluded_imgs, masks = occlude(images, minsz, maxsz)
    with open('occluded_cifar10_level' + str(level) + '.pickle', 'wb') as handle:
        pickle.dump((images, occluded_imgs, masks), handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    for i in range(5):
        main(i)
