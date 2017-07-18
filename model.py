
import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator
from sampler import Sampler


class AAE(object):
    def __init__(self, sess, encoder, z_dim, decoder, disor,
                 batch_size=100, learn_rate=0.0001,
                 sampler='Gaussiss', name='AAE'):
        self.encoder = Encoder(sess=sess, encoder=encoder, z_dim=z_dim)
        self.z_dim = z_dim
        self.decoder = Decoder(sess=sess, decoder=decoder, z_dim=z_dim)
        self.disor = Discriminator(z_dim=z_dim, layers=disor)
        self.sampler = Sampler(type=sampler, dim=z_dim)
        self.name = name
        self.sess = sess

        self.batch_size = batch_size
        self.learn_rate = learn_rate
        #

    def init_model(self):
        # initialize variables
        with tf.variable_scope(self.name) as vs:
            self.encoder.init_model()
            self.decoder.init_model()
            self.disor.init_model()
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope=self.name)
        # initialize optimizers
        self.loss_encoder_decoder, self.opt_encoder_decoder = self.optimizer_encoder_decoder()
        self.loss_discriminator, self.opt_discriminator = self.optimizer_discriminator()
        self.loss_encoder, self.opt_encoder = self.optimizer_encoder()
        # initialize variable
        self.sess.run(tf.initialize_all_variables())

    def optimizer_encoder_decoder(self):
        self.x_encoder_decoder = tf.placeholder(tf.float32, [self.batch_size, self.encoder[0]])
        with tf.variable_scope(self.name, reuse=True) as vs:
            f = self.encoder.feedforward(self.x_encoder_decoder)
            y = self.decoder.feedforward(f)
        loss = tf.reduce_mean(tf.square(y - self.x_encoder_decoder))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
        return loss, optimizer.minimize(loss, var_list=self.encoder.vars)

    def optimizer_discriminator(self):
        self.x_discriminator = tf.placeholder(tf.float32, [self.batch_size, self.encoder[0]])
        self.z_discriminator = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
        with tf.variable_scope(self.name, reuse=True) as vs:
            z_faker = self.encoder.feedforward(self.x_discriminator)
        pred_faker = self.disor.predict(z_faker)
        pred_real = self.disor.predict(self.z_discriminator)
        #
        loss_faker = tf.reduce_mean(tf.square(pred_faker - tf.zeros_like(pred_faker)))
        loss_real = tf.reduce_mean(tf.square(pred_real - tf.ones_like(pred_real)))
        loss = loss_faker + loss_real
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
        return loss, optimizer.minimize(loss, var_list=self.disor.vars)

    def optimizer_encoder(self):
        self.x_encoder = tf.placeholder(tf.float32, [self.batch_size, self.encoder[0]])
        with tf.variable_scope(self.name, reuse=True) as vs:
            f = self.encoder.feedforward(self.x_encoder)
        pred = self.disor.predict(self.x_encoder)
        loss = tf.reduce_mean(tf.square(pred - tf.ones_like(pred)))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
        return loss, optimizer.minimize(loss, var_list=self.encoder.vars)

    def train_encoder_decoder(self, input, labels=None):
        _, loss = self.sess.run([self.opt_encoder_decoder, self.loss_encoder_decoder],
                                {self.x_encoder_decoder:input})
        return loss

    def train_discriminator(self, input, labels=None):
        z_prior = self.sampler(self.batch_size)
        _, loss = self.sess.run([self.opt_discriminator, self.loss_discriminator],
                                {self.x_discriminator:input, self.z_discriminator:z_prior})
        return loss

    def train_encoder(self, input, labels=None):
        _, loss = self.sess.run([self.opt_encoder, self.loss_encoder],
                                {self.x_encoder:input})
        return loss

    def save(self, path):
        saver = self.sess.train.Saver(self.vars)
        saver.save(self.sess, path)
