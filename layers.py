
import tensorflow as tf


def full_connect(input, in_dim, out_dim, name='fc'):
    with tf.name_scope(name) as scope:
        W = tf.get_variable(name='W', shape=[in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable(name='b', shape=[1, out_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(0))
        a = tf.matmul(input, W) + b
    return a

def batch_normalize(input, shape, tiny=1e-6, name='bn'):
    with tf.name_scope(name) as scope:
        gamma = tf.get_variable('gamma', shape, dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        beta = tf.get_variable('beta', shape, dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(0))
        mean, var = tf.nn.moments(input, [0])
        a = gamma*(input-mean) / tf.sqrt(tiny+var) + beta
    return a

def active_relu(input, name='relu'):
    with tf.name_scope(name) as scope:
        return tf.nn.relu(input)

######################################################################################
def set_fc_vars(in_dim, out_dim, stddev=1.):
    k = tf.get_variable('W', [in_dim, out_dim],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    b = tf.get_variable('b', [out_dim],
                        initializer=tf.constant_initializer(0))

def set_bn_vars(shape, stddev=1):
    gamma = tf.get_variable('gamma', shape,
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
    beta = tf.get_variable('beta', shape,
                           initializer=tf.truncated_normal_initializer(0))

def calc_fc(input, name='fc'):
    with tf.name_scope(name):
        W = tf.get_variable('W')
        b = tf.get_variable('b')
        return tf.matmul(input, W) + b

def calc_relu(input, name='relu'):
    with tf.name_scope(name):
        return tf.nn.relu(input)

def calc_bn(input, name='bn'):
    with tf.name_scope(name):
        gamma = tf.get_variable('gamma')
        beta = tf.get_variable('beta')
        mean, var = tf.nn.moments(input, [0])
        return gamma*(input-mean) / tf.sqrt(1e-6+var) + beta