
import tensorflow as tf

class Encoder(object):
    def __init__(self, sess, encoder, z_dim, name='Encoder'):
        self.encoder = encoder
        self.z_dim = z_dim
        self.name = name
        self.sess = sess

    def init_model(self):
        in_list = self.encoder[:]
        out_list = self.encoder[1:]
        out_list.append(self.z_dim)
        with tf.variable_scope(self.name) as vs:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                self._set_vars(in_dim=in_dim, out_dim=out_dim, name="layer_"+str(n))

    def _set_vars(self, in_dim, out_dim, name, stddev=0.1):
        with tf.variable_scope(name) as vs:
            k = tf.get_variable('W', [in_dim, out_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [out_dim],
                                initializer=tf.constant_initializer(0))
        return k, b

    def feedforward(self, input):
        in_list = self.encoder[:]
        out_list = self.encoder[1:]
        out_list.append(self.z_dim)
        with tf.variable_scope(self.name, reuse=True) as vs:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                self._encoder_onestep(in_dim=in_dim, out_dim=out_dim, name="layer_"+str(n))

    def _encoder_onestep(self, input, name):
        with tf.variable_scope(name, reuse=True) as vs:
            W = tf.get_variable('W')
            b = tf.get_variable('b')
            a = tf.matmul(input, W) + b
        return tf.nn.sigmoid(a)