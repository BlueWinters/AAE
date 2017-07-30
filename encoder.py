
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

    def init_model(self):
        in_list = self.layer_set[:]
        out_list = self.layer_set[1:]
        out_list.append(self.z_dim)
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

    def feed_forward(self, input):
        h = input
        with tf.variable_scope(self.scope, reuse=True) as scope:
            assert self.scope.name == scope.name
            for n in range(len(self.layer_set)):
                with tf.variable_scope("layer{}".format(n), reuse=True):
                    ret = ly.calc_fc(h)
                    h = ly.calc_bn(ret)
                    h = ly.calc_relu(h)
        return ret

