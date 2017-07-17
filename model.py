
import tensorflow as tf
from sampler import Sampler


class AAE(object):
    def __init__(self, sess, encoder, z_dim, decoder, disor, sampler='Gaussiss', name='AAE'):
        self.encoder = encoder
        self.z_dim = z_dim
        self.decoder = decoder
        self.disor = disor
        self.name = name
        self.sess = sess

        self._init_vars()
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope=self.name)
        self.sess.run(tf.variables_initializer(self.vars))
        self.saver = tf.train.Saver(self.vars)

    def _init_vars(self):
        with tf.variable_scope(self.name):
            self._init_encoder_vars()
            self._init_decoder_vars()
            self._init_disor_vars()

    def _init_encoder_vars(self):
        in_list = self.encoder[:]
        out_list = self.encoder[1:]
        out_list.append(self.z_dim)
        with tf.variable_scope('encoder') as vs:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                self._set_vars(in_dim=in_dim, out_dim=out_dim, name="layer_"+str(n))

    def _init_decoder_vars(self):
        in_list = [self.z_dim]
        in_list.extend(self.decoder[:-1])
        out_list = self.decoder[:]
        with tf.variable_scope('decoder') as vs:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                self._set_vars(in_dim=in_dim, out_dim=out_dim, name="layer_"+str(n))

    def _init_disor_vars(self):
        in_list = [self.z_dim]
        in_list.extend(self.disor[:-1])
        out_list = self.disor[:]
        with tf.variable_scope('discriminator') as vs:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                self._set_vars(in_dim=in_dim, out_dim=out_dim, name="layer_"+str(n))

    def _set_vars(self, in_dim, out_dim, name, stddev=0.1):
        with tf.variable_scope(name) as vs:
            k = tf.get_variable('W', [in_dim, out_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [out_dim],
                                initializer=tf.constant_initializer(0))
        return k, b

    def reconstruct(self, input):
        with tf.variable_scope(self.name, reuse=True) as vs:
            # encoder
            f = self.encoder(input)
            # decoder
            y = self.decoder(f)
        return tf.reduce_sum(tf.square(y - input))

    def classify(self, input):

    def _reconstruct_onestep(self, input, name):
        with tf.variable_scope(name, reuse=True) as vs:
            W = tf.get_variable('W')
            b = tf.get_variable('b')
            a = tf.matmul(input, W) + b
        return tf.nn.sigmoid(a)

    def encoder(self, input):
        h = []
        h.append(input)
        # encoder
        in_list = self.encoder[:]
        out_list = self.encoder[1:]
        out_list.append(self.z_dim)
        with tf.variable_scope('encoder', reuse=True) as vs:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                h.append(self._reconstruct_onestep(h[-1], 'layer_'+str(n)))
        return h[-1]

    def decoder(self, input):
        h = []
        h.append(input)
        # decoder
        in_list = [self.z_dim]
        in_list.extend(self.decoder[:-1])
        out_list = self.decoder[:]
        with tf.variable_scope('decoder', reuse=True) as vs:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                h.append(self._reconstruct_onestep(h[-1], 'layer_'+str(n)))
        return h[-1]