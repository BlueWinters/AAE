
import tensorflow as tf
import layers as ly

class Encoder(object):
    def __init__(self, encoder, z_dim, name='Encoder'):
        self.layer_set = encoder
        self.z_dim = z_dim
        self.name = name
        self.vars = []
        
    def __call__(self, n_layer):
        return self.layer_set[n_layer]

    def feed_forward(self, input):
        h = input
        in_list = self.layer_set[:-1]
        out_list = self.layer_set[1:]

        with tf.variable_scope(self.name) as scope:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                with tf.variable_scope("layer{}".format(n)):
                    h = ly.full_connect(h, in_dim, out_dim)
                    h = ly.batch_normalize(h, [1,out_dim])
                    h = ly.active_relu(h)
            with tf.variable_scope("layer{}".format(len(self.layer_set))):
                ret = ly.full_connect(h, self.layer_set[-1], self.z_dim)
        self.vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        return ret

