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
        # a = tf.expand_dims(b_conv, 0)
        # b = tf.expand_dims(msk, 3)
        # print(a.shape)
        # print(b.shape)
        # tf.tensordot(a, b, 1)
        h_conv = tf.nn.relu(tf.nn.conv2d(input=img, filter=w_conv, strides=[1, 1, 1, 1], padding='SAME')
                            + tf.tensordot(tf.expand_dims(msk, 3), tf.expand_dims(b_conv, 0), 1))
    h_pool = tf.nn.max_pool(h_conv, ksize=[1, pool_sz, pool_sz, 1], strides=[1, pool_sz, pool_sz, 1], padding='SAME')
    return h_pool


def deconv(img, target_sz, target_dim, stri, name, sigbool=False):
    output_shape = [int(img.get_shape()[0]), target_sz, target_sz, target_dim]

    # print(output_shape)
    # print(img.shape)

    w_deco = tf.get_variable('wdeco' + name,
                             [6, 6, target_dim, img.get_shape()[-1]],
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


def binary_crossentropy(t, o):
    eps = 1e-6
    # o = tf.Print(o, [tf.count_nonzero(tf.is_nan(o))], 'this is o:')
    # t = tf.Print(t, [tf.count_nonzero(tf.is_nan(t))], 'this is t:')
    # j = 1.0 - o + eps
    # j = tf.Print(j, [tf.count_nonzero(tf.is_nan(j))], 'this is j:')
    # a = tf.log(j)
    # b = (1.0 - t)
    # c = tf.log(o + eps)
    # a = tf.Print(a, [tf.count_nonzero(tf.is_nan(a))], 'this is a:')
    # b = tf.Print(b, [tf.count_nonzero(tf.is_nan(b))], 'this is b:')
    # c = tf.Print(c, [tf.count_nonzero(tf.is_nan(c))], 'this is c:')
    # d = t * c
    # d = tf.Print(d, [tf.count_nonzero(tf.is_nan(d))], 'this is d:')
    # return -(d + b * a)
    return -(t * tf.log(o + eps) + (1.0 - t) * tf.log(1.0 - o + eps))

