
import tensorflow as tf

def set_fc_vars(in_dim, out_dim, stddev=0.1):
    W = tf.get_variable('W', [in_dim, out_dim],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    b = tf.get_variable('b', [out_dim],
                        initializer=tf.constant_initializer(0))
    return W, b

def set_bn_vars(shape):
    gamma = tf.get_variable('gamma', shape,
                            initializer=tf.truncated_normal_initializer(stddev=1.))
    beta = tf.get_variable('beta', shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.))
    return gamma, beta

def cal_fc(input, name='fc'):
    with tf.name_scope(name) as vs:
        W = tf.get_variable('W')
        b = tf.get_variable('b')
        a = tf.matmul(input, W) + b
        # with tf.name_scope('summary'):
        #     tf.summary.histogram('W', W)
        #     tf.summary.histogram('b', b)
    return a

def calc_relu(input, name='relu'):
    with tf.name_scope(name) as vs:
        return tf.nn.relu(input)

def calc_bn(input, name='bn'):
    with tf.name_scope(name) as vs:
        gamma = tf.get_variable('gamma')
        beta = tf.get_variable('beta')
        mean, var = tf.nn.moments(input, [0])
        # with tf.name_scope('summary'):
        #     tf.summary.histogram('gamma', gamma)
        #     tf.summary.histogram('beta', beta)
        return gamma*(input-mean) / tf.sqrt(1e-6+var) + beta
