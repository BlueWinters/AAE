
import tensorflow as tf

class Discriminator(object):
    def __init__(self, z_dim, layers, name='Discriminator'):
        self.z_dim = z_dim
        self.layers = layers
        self.name = name

    def init_var(self):
        in_list = [self.z_dim, self.layers[:-1]]
        out_list = self.layers[:]
        with tf.variable_scope(self.name):
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                    self._set_dis_var(in_dim=in_dim, out_dim=out_dim, name="layer_"+str(n))

    def _set_dis_var(self, in_dim, out_dim, name, stddev=0.1):
        with tf.variable_scope(name) as vs:
            k = tf.get_variable('W', [out_dim, in_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [out_dim, 1],
                                initializer=tf.constant_initializer(0))
        return k, b