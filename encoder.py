
import tensorflow as tf
import layers as ly

class Encoder(object):
    def __init__(self, in_dim=784, h_dim=1000, out_dim=2, name='Encoder'):
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.name = name
        self.vars = []

        self._init_model()

    def _init_model(self):
        with tf.variable_scope(self.name) as scope:
            with tf.variable_scope("layer1"):
                ly.set_fc_vars(in_dim=self.in_dim, out_dim=self.h_dim)
                ly.set_bn_vars(shape=[1,self.h_dim])
            with tf.variable_scope("layer2"):
                ly.set_fc_vars(in_dim=self.h_dim, out_dim=self.h_dim)
                ly.set_bn_vars(shape=[1,self.h_dim])
            with tf.variable_scope("layer3"):
                ly.set_fc_vars(in_dim=self.h_dim, out_dim=self.out_dim)
        self.scope = scope
        self.vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

    def get_variable(self):
        vars = []
        for n in range(len(self.vars)):
            vars.append(self.vars[n])
        return vars

    def feed_forward(self, input, is_train=True):
        with tf.variable_scope(self.scope, reuse=True):
            with tf.variable_scope("layer1", reuse=True):
                h = ly.calc_fc(input)
                h = ly.calc_bn(h, is_train)
                h = ly.calc_relu(h)
            with tf.variable_scope("layer2", reuse=True):
                h = ly.calc_fc(h)
                h = ly.calc_bn(h, is_train)
                h = ly.calc_relu(h)
            with tf.variable_scope("layer3", reuse=True):
                output = ly.calc_fc(h)
        return output

