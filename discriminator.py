
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

    def get_variable(self):
        vars = []
        for n in range(len(self.vars)):
            vars.append(self.vars[n])
        return vars

    def predict(self, input):
        h = input

        with tf.variable_scope(self.scope, reuse=True) as scope:
            assert self.scope.name == scope.name
            for n in range(len(self.layers)):
                with tf.variable_scope("layer{}".format(n), reuse=True):
                    ret = ly.calc_fc(h)
                    h = ly.calc_bn(ret)
                    h = ly.calc_relu(h)
            with tf.name_scope('output'):
                return tf.nn.sigmoid(ret)
