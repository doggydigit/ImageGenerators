from helpers import *


def outoutgen_network(image, mask, reuse=False):
    with tf.variable_scope('trivgen'):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        imsize = image.get_shape()[1]

        # Build Network Layer by Layers
        l1small = conv_relu_pool(image, 5, 2, 8, '1small', mask)
        l2small = conv_relu_pool(l1small, 5, 2, 8, '2small')
        l2big = conv_relu_pool(image, 10, 4, 8, '2big', mask)
        l2 = tf.concat([l2small, l2big], 4)
        l3 = conv_relu_pool(l2, 5, 4, 4, '3')
        l4 = deconv(l3, int(imsize / 4), 9, '4')
        out = deconv(l4, imsize, 1, '5', True)

    return out


def tiny_outoutgen_network(image, mask, reuse=False):
    """For tiny images of size 32x32."""
    with tf.variable_scope('trivgen'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # Build Network Layer by Layers
        l1small = conv_relu_pool(image, 3, 2, 4, '1small', mask)
        l2small = conv_relu_pool(l1small, 4, 2, 8, '2small')
        l2big = conv_relu_pool(image, 8, 4, 8, '2big', mask)
        l2 = tf.concat([l2small, l2big], 3)

        l3 = conv_relu_pool(l2, 4, 2, 16, '3')
        # l3 = conv_relu_pool(l2, 4, 2, 16, '3')
        l4 = deconv(l3, 16, 8, 4, '4')
        out = deconv(l4, 32, 3, 2, '5', True)
    # print(l1small.shape)
    # print(l2small.shape)
    # print(l2big.shape)
    # print(l2.shape)
    # print(l3.shape)
    # print(l4.shape)
    # print(out.shape)
    return out


def tiny_outshapegen_network(image, mask, reuse=False):
    """For tiny images of size 32x32."""
    with tf.variable_scope('trivgen'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # Build Network Layer by Layers
        l1small = conv_relu_pool(image, 3, 2, 4, '1small', mask)
        l2small = conv_relu_pool(l1small, 4, 2, 8, '2small')
        l2big = conv_relu_pool(image, 8, 4, 8, '2big', mask)
        l2 = tf.concat([l2small, l2big], 3)

        l3 = conv_relu_pool(l2, 4, 2, 16, '3')
        l4 = deconv(l3, 8, 12, 2, '4')
        l5 = deconv(l4, 16, 6, 2, '5')
        l6 = deconv(l5, 32, 3, 2, '6', True)
        out = tf.multiply(image, tf.expand_dims(mask, 3)) + tf.multiply(l6, tf.expand_dims(tf.ones(mask.shape)-mask, 3))
    return out


def tiny_full_network(image, mask, reuse=False):
    """For tiny images of size 32x32."""
    with tf.variable_scope('trivgen'):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        imgshp = image.shape
        mskcnst = imgshp[1]*imgshp[2]
        imgcnst = mskcnst*imgshp[3]
        flatimage = tf.reshape(image, [imgshp[0], imgcnst])
        flatmask = tf.reshape(mask, [imgshp[0], imgshp[1]*imgshp[2]])
        lcnst = int(int(imgcnst)/4)
        l1 = fullnet(flatimage, flatmask, imgcnst, mskcnst, lcnst, '1')
        l2 = fullnet(l1, flatmask, lcnst, mskcnst, imgcnst, '2', True)
        out = tf.reshape(l2, [imgshp[0], imgshp[1], imgshp[2], imgshp[3]])

    return out


def tiny_fullfull_network(image, mask, reuse=False):
    """For tiny images of size 32x32."""
    with tf.variable_scope('trivgen'):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        imgshp = image.shape
        mskcnst = imgshp[1]*imgshp[2]
        imgcnst = mskcnst*imgshp[3]
        flatimage = tf.reshape(image, [imgshp[0], imgcnst])
        flatmask = tf.reshape(mask, [imgshp[0], imgshp[1]*imgshp[2]])
        l1 = fullnet(flatimage, flatmask, imgcnst, mskcnst, imgcnst, '1')
        l2 = fullnet(l1, flatmask, imgcnst, mskcnst, imgcnst, '2')
        l3 = fullnet(l2, flatmask, imgcnst, mskcnst, imgcnst, '3')
        l4pre = fullnet(l3, flatmask, imgcnst, mskcnst, imgcnst, '4')
        l4 = tf.concat([flatimage, l4pre], 1)
        l5 = fullnet(l4, flatmask, 2*int(imgcnst), mskcnst, imgcnst, '5', True)
        l5img = tf.reshape(l5, [imgshp[0], imgshp[1], imgshp[2], imgshp[3]])
        out = (tf.multiply(image, tf.expand_dims(mask, 3)) +
               tf.multiply(l5img, tf.expand_dims(tf.ones(mask.shape) - mask, 3)))
    return out
