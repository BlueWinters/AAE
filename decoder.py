
import tensorflow as tf

class Decoder(object):
    def __init__(self, sess, decoder, z_dim, name='Decoder'):
        self.layer_set = decoder
        self.z_dim = z_dim
        self.name = name
        self.sess = sess
        self.vars = []
        
    def __call__(self, n_layer):
        return self.layer_set[n_layer]

    def init_model(self):
        in_list = [self.z_dim]
        in_list.extend(self.layer_set[:-1])
        out_list = self.layer_set[:]
        with tf.variable_scope(self.name) as vs:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                self._set_vars(in_dim=in_dim, out_dim=out_dim, name="layer_"+str(n))

        vars = tf.trainable_variables()
        for one in vars:
            if self.name in one.name:
                self.vars.append(one)

    def _set_vars(self, in_dim, out_dim, name, stddev=0.1):
        with tf.variable_scope(name) as vs:
            k = tf.get_variable('W', [in_dim, out_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [out_dim],
                                initializer=tf.constant_initializer(0))
        return k, b

    def feedforward(self, input):
        h = []
        h.append(input)

        in_list = self.layer_set[:]
        out_list = self.layer_set[1:]
        out_list.append(self.z_dim)
        with tf.variable_scope(self.name, reuse=True) as vs:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                h.append(self._decoder_onestep(h[-1], name="layer_"+str(n)))
        return h[-1]

    def _decoder_onestep(self, input, name):
        with tf.variable_scope(name, reuse=True) as vs:
            W = tf.get_variable('W')
            b = tf.get_variable('b')
            a = tf.matmul(input, W) + b
            tf.layers.dense()
        return tf.nn.relu(a)