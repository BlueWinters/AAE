
import tensorflow as tf
import numpy as np


def full_connect(input, in_dim, out_dim, name='fc'):
    with tf.name_scope(name) as scope:
        W = tf.get_variable(name='W', shape=[in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='b', shape=[out_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0))
        a = tf.matmul(input, W) + b
    return a

def batch_normalize(input, is_train=True, tiny=1e-6, decay=0.999, name='bn'):
    shape = input.get_shape().as_list()
    with tf.name_scope(name) as scope:
        scale = tf.get_variable('scale', [shape[-1]], dtype=tf.float32,
                                initializer=tf.constant_initializer(1))
        beta = tf.get_variable('beta', [shape[-1]], dtype=tf.float32,
                               initializer=tf.constant_initializer(0))
        non_mean = tf.get_variable('mean', shape[-1], trainable=False,
                                   initializer=tf.constant_initializer(0))
        non_var = tf.get_variable('var', shape[-1], trainable=False,
                                  initializer=tf.constant_initializer(1))
        if is_train:
            batch_mean, batch_var = tf.nn.moments(input, [0])
            train_mean = tf.assign(batch_mean, non_mean*decay + batch_mean*(1-decay))
            train_var = tf.assign(batch_var, non_var*decay + batch_var*(1-decay))
            with tf.control_dependencies([train_mean, train_var]):
                a = scale*(input-batch_mean) / tf.sqrt(tiny+batch_var) + beta
        else:
            a = scale*(input-non_mean) / tf.sqrt(tiny+non_var) + beta
    return a

def active_relu(input, name='relu'):
    with tf.name_scope(name) as scope:
        return tf.nn.relu(input)

######################################################################################
def set_fc_vars(in_dim, out_dim, stddev=1.):
    W = tf.get_variable(name='W', shape=[in_dim, out_dim], dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(float(in_dim))))
    b = tf.get_variable(name='b', shape=[out_dim], dtype=tf.float32,
                        initializer=tf.constant_initializer(0))

def set_bn_vars(shape):
    scale = tf.get_variable('scale', [shape[-1]], dtype=tf.float32,
                                initializer=tf.constant_initializer(1))
    beta = tf.get_variable('beta', [shape[-1]], dtype=tf.float32,
                           initializer=tf.constant_initializer(0))
    ave_mean = tf.get_variable('ave_mean', shape[-1], trainable=False,
                               initializer=tf.constant_initializer(0))
    ave_var = tf.get_variable('ave_var', shape[-1], trainable=False,
                              initializer=tf.constant_initializer(1))

def calc_fc(input, name='fc'):
    with tf.name_scope(name):
        W = tf.get_variable('W')
        b = tf.get_variable('b')
        return tf.matmul(input, W) + b

def calc_relu(input, name='relu'):
    with tf.name_scope(name):
        return tf.nn.relu(input)

def calc_sigmoid(input, name='sigmoid'):
    with tf.name_scope(name):
        return tf.nn.sigmoid(input)

def calc_bn(input, is_train=True, tiny=1e-6, decay=0.999, name='bn'):
    with tf.name_scope(name):
        scale = tf.get_variable('scale')
        beta = tf.get_variable('beta')
        ave_mean = tf.get_variable('ave_mean')
        ave_var = tf.get_variable('ave_var')

        if is_train:
            batch_mean, batch_var = tf.nn.moments(input, [0])
            train_mean = tf.assign(ave_mean, ave_mean*decay + batch_mean*(1-decay))
            train_var = tf.assign(ave_var, ave_var*decay + batch_var*(1-decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(input, batch_mean, batch_var, beta, scale, 0.001)
        else:
            return tf.nn.batch_normalization(input, ave_mean, ave_var, beta, scale, 0.001)

def calc_dropout(input, is_train=True, p=0.25, name='dropout'):
    with tf.name_scope(name):
        return tf.nn.dropout(input, p) if is_train is True else input