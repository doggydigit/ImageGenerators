import tensorflow as tf
import numpy as np


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_relu_pool(img, filter_size, pool_sz, nr_filters, name, msk=None):
    w_conv = tf.get_variable('wconv' + name,
                             [filter_size, filter_size, img.get_shape()[3], nr_filters],
                             dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_conv = tf.get_variable('bconv1' + name, [nr_filters], dtype=tf.float32, initializer=tf.constant_initializer(0))
    if msk is None:
        h_conv = tf.nn.relu(tf.nn.conv2d(input=img, filter=w_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv)
    else:
        h_conv = tf.nn.relu(tf.nn.conv2d(input=img, filter=w_conv, strides=[1, 1, 1, 1], padding='SAME')
                            + tf.tensordot(tf.expand_dims(msk, 3), tf.expand_dims(b_conv, 0), 1))
    h_pool = tf.nn.max_pool(h_conv, ksize=[1, pool_sz, pool_sz, 1], strides=[1, pool_sz, pool_sz, 1], padding='SAME')
    return h_pool


def deconv(img, target_sz, target_dim, stri, name, fsz=6, sigbool=False):
    output_shape = [int(img.get_shape()[0]), target_sz, target_sz, target_dim]

    # print(output_shape)
    # print(img.shape)

    w_deco = tf.get_variable('wdeco' + name,
                             [fsz, fsz, target_dim, img.get_shape()[-1]],
                             dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_deco = tf.get_variable('bdeco' + name, [target_dim], dtype=tf.float32, initializer=tf.constant_initializer(.1))
    h_deco = tf.nn.conv2d_transpose(img,
                                    w_deco,
                                    output_shape=output_shape,
                                    strides=[1, stri, stri, 1],
                                    padding='SAME')
    out = tf.contrib.layers.batch_norm(inputs=h_deco+b_deco, center=True, scale=True, scope='g_bn' + name)
    if sigbool:
        return tf.nn.sigmoid(out)
    else:
        return tf.nn.relu(out)


def fullnet(img, msk, imgsz, msksz, tgtsz, name, sigbool=False):
    imgweights = tf.get_variable('imgw_' + name,
                             [imgsz, tgtsz],
                             dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
    biases = tf.get_variable('b_' + name,
                             [tgtsz],
                             dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
    out = tf.matmul(img, imgweights) + tf.matmul(img, imgweights) + biases

    if sigbool:
        return tf.nn.sigmoid(out)
    else:
        return tf.nn.relu(out)


def binary_crossentropy(t, o):
    eps = 1e-6

    return -(t * tf.log(o + eps) + (1.0 - t) * tf.log(1.0 - o + eps))


def load_imagenet(batch, level, trainingbool):
    if trainingbool:
        loaded = np.load('../Occlude/occluded_cropped_imagenet_batch' + str(batch) +
                         '_level' + str(level) + '.npz')
    else:
        loaded = np.load('../Occlude/occluded_cropped_imagenet_batch' + str(batch) +
                         '_level' + str(level) + '.npz')
    return loaded['images'], loaded['occluded_imgs'], loaded['masks']


def load_c101():
