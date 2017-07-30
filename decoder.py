
import tensorflow as tf
import layers as ly

class Decoder(object):
    def __init__(self, decoder, z_dim, name='Decoder'):
        self.layer_set = decoder
        self.z_dim = z_dim
        self.name = name

        self.vars = []
        
    def __call__(self, n_layer):
        return self.layer_set[n_layer]

    def feed_forward(self, input):
        h = input
        in_list = [self.z_dim]
        in_list.extend(self.layer_set[:-1])
        out_list = self.layer_set[:]

        with tf.variable_scope(self.name) as scope:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                with tf.variable_scope("layer{}".format(n)):
                    ret = ly.full_connect(h, in_dim, out_dim)
                    h = ly.batch_normalize(ret, [1,out_dim])
                    h = ly.active_relu(h)
        self.vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        return ret