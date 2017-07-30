
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
        # with tf.name_scope('summary'):
        #     tf.summary.histogram('gamma', gamma)
        #     tf.summary.histogram('beta', beta)
        a = gamma*(input-mean) / tf.sqrt(tiny+var) + beta
    return a

def active_relu(input, name='relu'):
    with tf.name_scope(name) as scope:
        return tf.nn.relu(input)