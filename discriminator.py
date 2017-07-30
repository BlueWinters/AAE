
import tensorflow as tf
import layers as ly

class Discriminator(object):
    def __init__(self, z_dim, layers, name='Discriminator'):
        self.z_dim = z_dim
        self.layers = layers
        self.name = name
        self.vars = []

    def predict(self, input):
        h = input
        in_list = [self.z_dim]
        in_list.extend(self.layers[:-1])
        out_list = self.layers[:]

        with tf.variable_scope(self.name) as scope:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                with tf.variable_scope("layer{}".format(n)):
                    ret = ly.full_connect(h, in_dim, out_dim)
                    h = ly.batch_normalize(ret, [1,out_dim])
                    h = ly.active_relu(h)
            with tf.name_scope('output'):
                a = tf.nn.sigmoid(ret)
        self.vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        return a