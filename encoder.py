
import tensorflow as tf
import layers as ly

class Encoder(object):
    def __init__(self, in_dim=784, h_dim=1000, out_dim=2, name='Encoder'):
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.name = name
        self.vars = []

        self.type = 'convolution'

        self._init_model()

    def get_variable(self):
        vars = []
        for n in range(len(self.vars)):
            vars.append(self.vars[n])
        return vars

    def _init_model(self):
        if self.type == 'batch_norm':
            self._init_model_v1()
        elif self.type == 'dropout':
            self._init_model_v2()
        elif self.type == 'v3':
            self._init_model_v3()
        elif self.type == 'convolution':
            self._init_model_v4()

    def feed_forward(self, input, is_train=True):
        if self.type == 'batch_norm':
            return self.feed_forward_v1(input, is_train)
        elif self.type == 'dropout':
            return self.feed_forward_v2(input, is_train)
        elif self.type == 'v3':
            return self.feed_forward_v3(input, is_train)
        elif self.type == 'convolution':
            return self.feed_forward_v4(input, is_train)

    def _init_model_v1(self):
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
        self.vars = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

    def feed_forward_v1(self, input, is_train=True):
        with tf.variable_scope(self.scope, reuse=True):
            with tf.variable_scope("layer1"):
                h = ly.calc_fc(input)
                h = ly.calc_bn(h, is_train)
                h = ly.calc_relu(h)
            with tf.variable_scope("layer2"):
                h = ly.calc_fc(h)
                h = ly.calc_bn(h, is_train)
                h = ly.calc_relu(h)
            with tf.variable_scope("layer3"):
                output = ly.calc_fc(h)
        return output

    def _init_model_v2(self):
        with tf.variable_scope(self.name) as scope:
            with tf.variable_scope("layer1"):
                ly.set_fc_vars(in_dim=self.in_dim, out_dim=self.h_dim)
            with tf.variable_scope("layer2"):
                ly.set_fc_vars(in_dim=self.h_dim, out_dim=self.h_dim)
            with tf.variable_scope("layer3"):
                ly.set_fc_vars(in_dim=self.h_dim, out_dim=self.out_dim)
        self.scope = scope
        self.vars = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

    def feed_forward_v2(self, input, is_train=True):
        with tf.variable_scope(self.scope, reuse=True):
            with tf.variable_scope("layer1"):
                h = ly.calc_fc(input)
                h = ly.calc_dropout(h, is_train)
                h = ly.calc_relu(h)
            with tf.variable_scope("layer2"):
                h = ly.calc_fc(h)
                h = ly.calc_dropout(h, is_train)
                h = ly.calc_relu(h)
            with tf.variable_scope("layer3"):
                output = ly.calc_fc(h)
        return output

    def _init_model_v3(self):
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
        self.vars = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

    def feed_forward_v3(self, input, is_train=True):
        with tf.variable_scope(self.scope, reuse=True):
            with tf.variable_scope("layer1"):
                h = ly.calc_fc(input)
                h = ly.calc_bn(h, is_train)
                h = ly.calc_relu(h)
            with tf.variable_scope("layer2"):
                h = ly.calc_fc(h)
                h = ly.calc_bn(h, is_train)
                h = ly.calc_relu(h)
            with tf.variable_scope("layer3"):
                output = ly.calc_fc(h)
        return output

    def _init_model_v4(self):
        dim_list = [3, 16, 16, 2]
        with tf.variable_scope(self.name) as scope:
            with tf.variable_scope('layer1'):
                ly.set_conv_vars(dim_list[0], 3, 3, dim_list[1])
                ly.set_conv_bn_vars(shape=[1,dim_list[1]])
            with tf.variable_scope('layer2'):
                ly.set_conv_vars(dim_list[1], 3, 3, dim_list[2])
                ly.set_conv_bn_vars(shape=[1,dim_list[2]])
            with tf.variable_scope('layer3'):
                ly.set_conv_vars(dim_list[2], 3, 3, dim_list[3])
        self.scope = scope
        self.vars = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

    def feed_forward_v4(self, input, is_train=True):
        with tf.variable_scope(self.name, reuse=True):
            with tf.variable_scope('layer1'):
                h = ly.calc_conv(input)
                h = ly.calc_conv_bn(h)
                h = ly.calc_relu(h)
            with tf.variable_scope('layer2'):
                h = ly.calc_conv(h)
                h = ly.calc_conv_bn(h)
                h = ly.calc_relu(h)
            with tf.variable_scope('layer3'):
                output = ly.calc_conv(h)
        return output