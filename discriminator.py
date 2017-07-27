
import tensorflow as tf

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
        with tf.variable_scope(self.name):
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                self._set_fc_vars(in_dim=in_dim, out_dim=out_dim, name="layer_"+str(n))
                self._set_bn_vars(shape=[1,out_dim], name="layer_"+str(n))

        vars = tf.trainable_variables()
        for one in vars:
            if self.name in one.name:
                self.vars.append(one)

    def _set_fc_vars(self, in_dim, out_dim, name, stddev=0.1):
        with tf.variable_scope(name) as vs:
            k = tf.get_variable('W', [in_dim, out_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [out_dim],
                                initializer=tf.constant_initializer(0))
        return k, b

    def _set_bn_vars(self, shape, name):
        with tf.variable_scope(name) as vs:
            gamma = tf.get_variable('gamma', shape,
                                    initializer=tf.truncated_normal_initializer(stddev=1.))
            beta = tf.get_variable('beta', shape,
                                   initializer=tf.truncated_normal_initializer(stddev=0.))
        return gamma, beta

    def predict(self, input):
        h = input

        with tf.variable_scope(self.name, reuse=True) as vs:
            for n in range(len(self.layers)):
                ret = self._cal_fc(h, name="layer_"+str(n))
                h = self._calc_active(ret, name="layer_"+str(n))
                h = self._calc_bn(h, name="layer_"+str(n))
        # modify discriminator output distribution as sigmoid(probability)
        # return tf.nn.sigmoid(h[-1])
        return ret

    def _cal_fc(self, input, name):
        with tf.variable_scope(name, reuse=True) as vs:
            W = tf.get_variable('W')
            b = tf.get_variable('b')
            a = tf.matmul(input, W) + b
        return a

    def _calc_active(self, input, name):
        with tf.variable_scope(name, reuse=True) as vs:
            return tf.nn.relu(input)

    def _calc_bn(self, input, name):
        with tf.variable_scope(name, reuse=True) as vs:
            gamma = tf.get_variable('gamma')
            beta = tf.get_variable('beta')
            mean, var = tf.nn.moments(input, [0])
        return gamma*(input-mean) / tf.sqrt(1e-6+var) + beta