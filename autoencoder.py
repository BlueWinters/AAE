
import tensorflow as tf

class Autoencoder(object):
    def __init__(self, sess, encoder, z_dim, decoder, name='Autoencoder'):
        self.encoder = encoder
        self.z_dim = z_dim
        self.decoder = decoder
        self.name = name
        self.sess = sess

        self.init_vars()
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope=self.name)
        self.sess.run(tf.variables_initializer(self.vars))
        self.saver = tf.train.Saver(self.vars)

    def init_vars(self):
        with tf.variable_scope(self.name):
            name = tf.get_variable_scope()
            self.init_encoder_var()
            self.init_decoder_var()

    def init_encoder_var(self):
        in_list = self.encoder[:]
        out_list = self.encoder[1:]
        out_list.append(self.z_dim)
        with tf.variable_scope('encoder') as vs:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                self._set_encoder_var(in_dim=in_dim, out_dim=out_dim, name="layer_"+str(n))

    def init_decoder_var(self):
        in_list = [self.z_dim]
        in_list.extend(self.decoder[:-1])
        out_list = self.decoder[:]
        with tf.variable_scope('decoder') as vs:
            for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                self._set_decoder_var(in_dim=in_dim, out_dim=out_dim, name="layer_"+str(n))

    def _set_encoder_var(self, in_dim, out_dim, name, stddev=0.1):
        with tf.variable_scope(name) as vs:
            k = tf.get_variable('W', [in_dim, out_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [out_dim],
                                initializer=tf.constant_initializer(0))
        return k, b

    def _set_decoder_var(self, in_dim, out_dim, name, stddev=0.1):
        with tf.variable_scope(name) as vs:
            k = tf.get_variable('W', [in_dim, out_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [out_dim],
                                initializer=tf.constant_initializer(0))
        return k, b

    def loss(self, input):
        h = []
        h.append(input)

        with tf.variable_scope(self.name, reuse=True) as vs:
            # encoder
            in_list = self.encoder[:]
            out_list = self.encoder[1:]
            out_list.append(self.z_dim)
            with tf.variable_scope('encoder', reuse=True) as vs:
                for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                    h.append(self._encoder_onestep(h[-1], 'layer_'+str(n)))
            # decoder
            in_list = [self.z_dim]
            in_list.extend(self.decoder[:-1])
            out_list = self.decoder[:]
            with tf.variable_scope('decoder', reuse=True) as vs:
                for n, (in_dim, out_dim) in enumerate(zip(in_list, out_list)):
                    h.append(self._decoder_onestep(h[-1], 'layer_'+str(n)))
        return tf.reduce_sum(tf.square(h[-1] - input))

    def _encoder_onestep(self, input, name):
        with tf.variable_scope(name, reuse=True) as vs:
            W = tf.get_variable('W')
            b = tf.get_variable('b')
            a = tf.matmul(input, W) + b
        return tf.nn.sigmoid(a)

    def _decoder_onestep(self, input, name):
        with tf.variable_scope(name, reuse=True) as vs:
            W = tf.get_variable('W')
            b = tf.get_variable('b')
            a = tf.matmul(input, W) + b
        return tf.nn.sigmoid(a)

    def save(self, ckpt_path):
        self.saver.save(self.sess, ckpt_path)

    def restore(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)