import os
import pickle
import random
from generate import *


def main(level=0, loading=False, training=True, viewing=False):

    testing = True

    # generator = 'outshape'
    generator = 'deepoutshape'
    # generator = 'full'
    # generator = 'fullfull'

    train_cnst = 5
    gan = False

    mnistbool = False
    cifar10bool = False
    klab325bool = False
    imgnetbool = False
    c101bool = True

    #
    # Initialize stuff
    #
    if cifar10bool:
        nr_batches = 1
        trainprintint = 50
        datadir = 'models/CIFAR10_'
    elif imgnetbool:
        nr_batches = 20
        trainprintint = 1
        datadir = 'models/Imagenet_'
    elif c101bool:
        datadir = 'models/Imagenet_'
        if training:
            raise NotImplementedError
    else:
        raise NotImplementedError
    if training:
        loadlevel = str(level-1)
    else:
        loadlevel = str(level)

    if gan:
        raise NotImplementedError
    else:
        save_dir = datadir + generator + '_level' + str(level) + '/'
        load_dir = datadir + generator + '_level' + loadlevel + '/'
    curdir = os.path.dirname(os.path.abspath(__file__))
    np.random.seed(1002)
    tf.set_random_seed(1002)

    #
    # Load Data
    #
    if mnistbool:
        # from tensorflow.examples.tutorials.mnist import input_data
        # data = input_data.read_data_sets("MNIST_data/")
        # x_train = data.train.images[:55000, :]
        # nr_samples = 55000
        # imsize = 28
        # colors = 1
        raise NotImplementedError
    elif cifar10bool:

        with open(curdir + '/../Occlude/occluded_cifar10_level0.pickle', 'rb') as f:
            y, x, masks = pickle.load(f)
        xshape = x.shape
        nr_samples = int(xshape[0])
        imsize = int(xshape[1])
        colors = int(xshape[3])
        nr_test = int(nr_samples / 10)
        nr_train = nr_samples - nr_test
        x_train = x[:nr_train, :, :, :] / 255
        y_train = y[:nr_train, :, :, :] / 255
        msk_train = masks[:nr_train, :, :]
        x_test = x[nr_train:, :, :, :] / 255
        y_test = y[nr_train:, :, :, :] / 255
        msk_test = masks[nr_train:, :, :]
        for i in range(1, level+1):
            with open(curdir + '/../Occlude/occluded_cifar10_level' + str(level) + '.pickle', 'rb') as f:
                y, x, masks = pickle.load(f)
            y_train = np.concatenate((y_train, y[:nr_train, :, :, :] / 255), axis=0)
            x_train = np.concatenate((x_train, x[:nr_train, :, :, :] / 255), axis=0)
            msk_train = np.concatenate((msk_train, masks[:nr_train, :, :]), axis=0)
            y_test = np.concatenate((y_test, y[nr_train:, :, :, :] / 255), axis=0)
            x_test = np.concatenate((x_test, x[nr_train:, :, :, :] / 255), axis=0)
            msk_test = np.concatenate((msk_test, masks[nr_train:, :, :]), axis=0)
        nr_test = x_test.shape[0]
        nr_train = x_train.shape[0]
        nr_samples = nr_train + nr_test
        del x, y, masks
    elif klab325bool:
        # imsize = 28
        # colors = 1
        raise NotImplementedError
    elif imgnetbool:
        if training:
            y_train, x_train, msk_train = load_imagenet(0, level, training)
            nr_train = msk_train.shape[0]
            y_train = np.reshape(y_train, (nr_train, -1))
            y_test, x_test, msk_test, nr_test = None, None, None, None
        else:
            y_test, x_test, msk_test = load_imagenet(20, level, training)
            nr_test = msk_test.shape[0]
            y_train, x_train, msk_train, nr_train = None, None, None, None
        nr_samples = None
        imsize = 320
        colors = 3
    else:
        raise NotImplementedError

    # Build Network
    if cifar10bool:
        batch_size = 1000
    elif imgnetbool:
        batch_size = 100
    else:
        raise NotImplementedError
    occ_img_placeholder = tf.placeholder(tf.float32, [batch_size, imsize, imsize, colors])
    msk_placeholder = tf.placeholder(tf.float32, [batch_size, imsize, imsize])
    if mnistbool:
        raise NotImplementedError
    elif cifar10bool:
        if generator is 'outout':
            img_recons = tiny_outoutgen_network(occ_img_placeholder, msk_placeholder)
        elif generator is 'outshape':
            img_recons = tiny_outshapegen_network(occ_img_placeholder, msk_placeholder)
        elif generator is 'full':
            img_recons = tiny_full_network(occ_img_placeholder, msk_placeholder)
        elif generator is 'fullfull':
            img_recons = tiny_fullfull_network(occ_img_placeholder, msk_placeholder)
        else:
            raise NotImplementedError
    elif klab325bool:
        raise NotImplementedError
    elif imgnetbool:
        if generator is 'outshape':
            img_recons = outshapegen_network(occ_img_placeholder, msk_placeholder)
        elif generator is 'deepoutshape':
            img_recons = deepoutshapegen_network(occ_img_placeholder, msk_placeholder)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    print('Network built successfully')

    #
    # Build Optimizer
    #
    if imgnetbool:
        train_iters = 600
    else:
        train_iters = int(nr_samples * train_cnst / batch_size)
    true_img_placeholder = tf.placeholder(tf.float32, shape=[batch_size, imsize * imsize * colors])
    lx = binary_crossentropy(true_img_placeholder, tf.reshape(img_recons, [batch_size, imsize * imsize * colors]))
    lx = tf.reduce_mean(tf.reduce_sum(lx, 1))
    l2 = tf.reduce_mean(0.0001 * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
    cost = lx+l2
    optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)
    trainer = optimizer.minimize(cost)
    genfetches = []
    disfetches = []

    genfetches.extend([cost, trainer])
    if gan:
        disfetches.extend([discost, distrainer])
    lxs = [0] * train_iters
    # fetches.extend([lx, l2, cost, trainer])
    # lxs = [0] * train_iters
    # l2s = [0] * train_iters
    # costs = [0] * train_iters

    #
    # Check if your are just testing program
    #
    # if testing:
    #     print('This is only a test')
    #     train_iters = int(train_iters/100)
    #     batch_size = batch_size/100

    #
    # Train Network
    #
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()  # For later saving
    tf.global_variables_initializer().run()
    if loading:
        saver.restore(sess, os.path.join(load_dir, "drawmodel.ckpt"))
        print('Model Loaded')
    else:
        sess.run(tf.global_variables_initializer())

    if training:
        print('Training ' + generator + ' network with difficulty level ' + str(level) + ' for ' + str(train_iters) +
              ' iterations...')

        nr_iters_per_batch = int(train_iters / nr_batches)
        for b in range(nr_batches):
            indices = list(range(nr_train))
            for i in range(nr_iters_per_batch):
                isis = random.sample(indices, batch_size)
                occ_batch = x_train[isis, :, :, :]
                true_batch = y_train[isis, :]
                mask_batch = msk_train[isis, :, :]
                # occ_batch = x_train[i*batch_size:(i+1)*batch_size, :, :, :]
                # true_batch = x_true[i * batch_size:(i+1) * batch_size, :]
                # mask_batch = masks[i * batch_size:(i+1) * batch_size, :, :]
                feed_dict = {true_img_placeholder: true_batch,
                             occ_img_placeholder: occ_batch,
                             msk_placeholder: mask_batch}
                results = sess.run(genfetches, feed_dict)
                # lxs[i], l2s[i], costs[i], _ = results
                lxs[i], _ = results
                # print(lxs)
                if i % trainprintint == 0:
                    print("iter=%d : Cost: %f" % (i+b*nr_iters_per_batch, lxs[i]))
                    # print("iter=%d : Cost is %f, Lx is %f, L2 is %f" % (i, costs[i], lxs[i], l2s[i]))
            if b < nr_batches - 1:
                print('New training batch (' + str(b+1) + ')')
                y_train, x_train, msk_train = load_imagenet(b+1, level, True)
                y_train = np.reshape(y_train, (nr_train, imsize * imsize * colors))

        # Save Model
        ckpt_file = os.path.join(save_dir, "drawmodel.ckpt")
        print("Model saved in file: %s" % saver.save(sess, ckpt_file))

    #
    # Testing Model
    #
    if viewing:
        fetches = []
        fetches.extend([img_recons, cost])
        indices = list(range(nr_test))
        isis = random.sample(indices, batch_size)
        occ_batch = x_test[isis, :, :, :]
        true_batch = np.reshape(y_test[isis, :], (batch_size, imsize * imsize * colors))
        mask_batch = msk_test[isis, :, :]
        feed_dict = {true_img_placeholder: true_batch, occ_img_placeholder: occ_batch, msk_placeholder: mask_batch}
        reconstructed_imgs, testerror = sess.run(fetches, feed_dict)
        with open('reconstruction_' + generator + '.pickle', 'wb') as handle:
            pickle.dump((occ_batch, y_test[isis, :], mask_batch, reconstructed_imgs, testerror),
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

        # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trivgen'):
        #     print(i)

        # plt.imshow(reconstructed_imgs[0, :, :, :], interpolation='nearest')
        # plt.show()

    sess.close()


if __name__ == "__main__":

    # prog = 'training'
    prog = 'testing'
    # prog = 'jj'

    if prog is 'training':
        # main(1)
        # tf.reset_default_graph()
        for lev in range(5, 9):
            main(lev, True)
            tf.reset_default_graph()
    elif prog is 'testing':
        main(6, True, False, True)
    else:
        for lev in range(3, 5):
            main(lev, True)
            tf.reset_default_graph()
        # raise NotImplementedError
