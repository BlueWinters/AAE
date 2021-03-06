
import tensorflow as tf
import layers as lr

class Decoder(object):
    def __init__(self, in_dim=2, h_dim=1024, out_dim=784, name='Decoder'):
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.name = name
        self._init_model()

    def _init_model(self):
        with tf.variable_scope(self.name) as scope:
            with tf.variable_scope("layer1"):
                lr.set_fc_vars(in_dim=self.in_dim, out_dim=self.h_dim)
                lr.set_bn_vars(shape=[1,self.h_dim])
            with tf.variable_scope("layer2"):
                lr.set_fc_vars(in_dim=self.h_dim, out_dim=self.h_dim)
                lr.set_bn_vars(shape=[1,self.h_dim])
            with tf.variable_scope("layer3"):
                lr.set_fc_vars(in_dim=self.h_dim, out_dim=self.out_dim)
        self.scope = scope

    def feed_forward(self, input, is_train=True):
        with tf.variable_scope(self.scope, reuse=True):
            with tf.variable_scope("layer1"):
                h = lr.calc_fc(input)
                h = lr.calc_bn(h, is_train)
                h = lr.calc_relu(h)
            with tf.variable_scope("layer2"):
                h = lr.calc_fc(h)
                h = lr.calc_bn(h, is_train)
                h = lr.calc_relu(h)
            with tf.variable_scope("layer3"):
                h = lr.calc_fc(h)
                output = lr.calc_sigmoid(h)
        return output

