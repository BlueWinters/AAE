
import tensorflow as tf

class Discriminator(object):
    def __init__(self, z_dim, layers, name='Discriminator'):
        self.z_dim = z_dim
        self.layers = layers
        self.name = name

    def init_model(self):
        in_list = [self.z_dim]
        in_list.extend(self.layers[:-1])
        out_list = self.layers[:]
        with tf.variable_scope(self.name):
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                self._set_vars(in_dim=in_dim, out_dim=out_dim, name="layer_"+str(n))

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope=self.name)

    def _set_vars(self, in_dim, out_dim, name, stddev=0.1):
        with tf.variable_scope(name) as vs:
            k = tf.get_variable('W', [out_dim, in_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [out_dim, 1],
                                initializer=tf.constant_initializer(0))
        return k, b

    def predict(self, input):
        h = []
        h.append(input)

        in_list = [self.z_dim]
        in_list.extend(self.layers[:-1])
        out_list = self.layers[:]
        with tf.variable_scope(self.name, reuse=True) as vs:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                h.append(self._feedforward_onestep(h[-1], name="layer_"+str(n)))
        return h[-1]

    def _feedforward_onestep(self, input, name):
        with tf.variable_scope(name, reuse=True) as vs:
            W = tf.get_variable('W')
            b = tf.get_variable('b')
            a = tf.matmul(input, W) + b
        return tf.nn.sigmoid(a)

    def save(self, path):
        saver = tf.train.Saver(self.vars)
        saver.save(self.sess, path)

    def restore(self, path):
        self.saver.save(self.sess, path)