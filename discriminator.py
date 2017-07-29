
import tensorflow as tf
import layers as ly

class Discriminator(object):
    def __init__(self, z_dim, layers, name='Discriminator'):
        self.z_dim = z_dim
        self.layers = layers
        self.name = name
        self.vars = []

    def init_model(self):
        in_list = [self.z_dim]
        in_list.extend(self.layers[:-1])
        out_list = self.layers[:]
        with tf.variable_scope(self.name) as scope:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                with tf.variable_scope("layer{}".format(n)):
                    ly.set_fc_vars(in_dim=in_dim, out_dim=out_dim)
                    ly.set_bn_vars(shape=[1,out_dim])
        self.scope = scope
        self.vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

    def predict(self, input):
        h = input

        with tf.variable_scope(self.name, reuse=True) as scope:
            assert self.scope.name == scope.name
            for n in range(len(self.layers)):
                with tf.variable_scope("layer{}".format(n), reuse=True):
                    ret = ly.cal_fc(h)
                    h = ly.calc_bn(ret)
                    h = ly.calc_relu(h)
            with tf.name_scope('output'):
                return tf.nn.sigmoid(ret)

    # def _set_fc_vars(self, in_dim, out_dim, name, stddev=0.1):
    #     with tf.variable_scope(name) as vs:
    #         k = tf.get_variable('W', [in_dim, out_dim],
    #                             initializer=tf.truncated_normal_initializer(stddev=stddev))
    #         b = tf.get_variable('b', [out_dim],
    #                             initializer=tf.constant_initializer(0))
    #     return k, b
    #
    # def _set_bn_vars(self, shape, name):
    #     with tf.variable_scope(name) as vs:
    #         gamma = tf.get_variable('gamma', shape,
    #                                 initializer=tf.truncated_normal_initializer(stddev=1.))
    #         beta = tf.get_variable('beta', shape,
    #                                initializer=tf.truncated_normal_initializer(stddev=0.))
    #     return gamma, beta
    #

    #
    # def _cal_fc(self, input, name):
    #     with tf.variable_scope(name, reuse=True) as vs:
    #         W = tf.get_variable('W')
    #         b = tf.get_variable('b')
    #         a = tf.matmul(input, W) + b
    #         with tf.name_scope('summary'):
    #             tf.summary.histogram('W', W)
    #             tf.summary.histogram('b', b)
    #     return a
    #
    # def _calc_active(self, input, name='active'):
    #     with tf.name_scope(name) as vs:
    #         return tf.nn.relu(input)
    #
    # def _calc_bn(self, input, name=):
    #     with tf.name_scope(name) as vs:
    #         gamma = tf.get_variable('gamma')
    #         beta = tf.get_variable('beta')
    #         mean, var = tf.nn.moments(input, [0])
    #         with tf.name_scope('summary'):
    #             tf.summary.histogram('gamma', gamma)
    #             tf.summary.histogram('beta', beta)
    #         return gamma*(input-mean) / tf.sqrt(1e-6+var) + beta