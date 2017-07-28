
import tensorflow as tf

class Encoder(object):
    def __init__(self, sess, encoder, z_dim, name='Encoder'):
        self.layer_set = encoder
        self.z_dim = z_dim
        self.name = name
        self.sess = sess
        self.vars = []
        
    def __call__(self, n_layer):
        return self.layer_set[n_layer]

    def init_model(self):
        in_list = self.layer_set[:]
        out_list = self.layer_set[1:]
        out_list.append(self.z_dim)
        with tf.variable_scope(self.name) as vs:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                self._set_fc_vars(in_dim=in_dim, out_dim=out_dim, name="layer_"+str(n))
                self._set_bn_vars(shape=[1,out_dim], name="layer_"+str(n))

        self.vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

    def _set_fc_vars(self, in_dim, out_dim, name, stddev=0.1):
        with tf.variable_scope(name) as vs:
            W = tf.get_variable('W', [in_dim, out_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [out_dim],
                                initializer=tf.constant_initializer(0))
        return W, b

    def _set_bn_vars(self, shape, name):
        with tf.variable_scope(name) as vs:
            gamma = tf.get_variable('gamma', shape,
                                    initializer=tf.truncated_normal_initializer(stddev=1.))
            beta = tf.get_variable('beta', shape,
                                   initializer=tf.truncated_normal_initializer(stddev=0.))
        return gamma, beta

    def feedforward(self, input):
        h = input

        with tf.variable_scope(self.name, reuse=True) as vs:
            for n in range(len(self.layer_set)):
                ret = self._calc_matmult(h, name="layer_"+str(n))
                h = self._calc_bn(ret, name="layer_"+str(n))
                h = self._calc_active(h, name="layer_"+str(n))
        return ret

    def _calc_matmult(self, input, name):
        with tf.variable_scope(name, reuse=True) as vs:
            W = tf.get_variable('W')
            b = tf.get_variable('b')
            a = tf.matmul(input, W) + b
            with tf.name_scope('summary'):
                tf.summary.histogram('W', W)
                tf.summary.histogram('b', b)
        return a

    def _calc_active(self, input, name):
        with tf.variable_scope(name, reuse=True) as vs:
            return tf.nn.relu(input)

    def _calc_bn(self, input, name):
        with tf.variable_scope(name, reuse=True) as vs:
            gamma = tf.get_variable('gamma')
            beta = tf.get_variable('beta')
            mean, var = tf.nn.moments(input, [0])
            with tf.name_scope('summary'):
                tf.summary.histogram('gamma', gamma)
                tf.summary.histogram('beta', beta)
        return gamma*(input-mean) / tf.sqrt(1e-6+var) + beta