
import tensorflow as tf
import os as os

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator
from sampler import Sampler


class AAE(object):
    def __init__(self, sess, layer_encoder, z_dim, layer_decoder, layer_disor,
                 batch_size=100, learn_rate=0.0001,
                 prior_type='Gaussiss', name='AAE'):
        self.encoder = Encoder(sess=sess, encoder=layer_encoder, z_dim=z_dim)
        self.z_dim = z_dim
        self.decoder = Decoder(sess=sess, decoder=layer_decoder, z_dim=z_dim)
        self.disor = Discriminator(z_dim=z_dim, layers=layer_disor)
        self.sampler = Sampler(type=prior_type, dim=z_dim)
        self.name = name
        self.sess = sess

        self.batch_size = batch_size
        self.learn_rate = learn_rate
        # initialize model
        self.init_model()

    def init_model(self):
        # initialize variables
        with tf.variable_scope(self.name) as vs:
            self.encoder.init_model()
            self.decoder.init_model()
            self.disor.init_model()
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # initialize optimizers
        self.loss_encoder_decoder, self.opt_encoder_decoder = self.optimizer_encoder_decoder()
        self.loss_discriminator, self.opt_discriminator = self.optimizer_discriminator()
        self.loss_encoder, self.opt_encoder = self.optimizer_encoder()
        # initialize variable
        self.sess.run(tf.global_variables_initializer())

    def optimizer_encoder_decoder(self):
        self.x_encoder_decoder = tf.placeholder(tf.float32, [self.batch_size, self.encoder(0)])
        with tf.variable_scope(self.name, reuse=True) as vs:
            f = self.encoder.feedforward(self.x_encoder_decoder)
            y = self.decoder.feedforward(f)
        loss = tf.reduce_mean(tf.square(y - self.x_encoder_decoder))
        vars = self.encoder.vars
        vars.extend(self.decoder.vars)
        # name conflict, so rename the optimizer as Adam_en_de
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate, name='Adam_en_de')
        return loss, optimizer.minimize(loss, var_list=vars)

    def optimizer_discriminator(self):
        self.x_discriminator = tf.placeholder(tf.float32, [self.batch_size, self.encoder(0)])
        self.z_discriminator = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
        with tf.variable_scope(self.name, reuse=True) as vs:
            z_faker = self.encoder.feedforward(self.x_discriminator)
            pred_faker = self.disor.predict(z_faker)
            pred_real = self.disor.predict(self.z_discriminator)
        # calculate loss
        loss_faker = tf.reduce_mean(tf.square(pred_faker - tf.zeros_like(pred_faker)))
        loss_real = tf.reduce_mean(tf.square(pred_real - tf.ones_like(pred_real)))
        loss = loss_faker + loss_real
        # name conflict, so rename the optimizer as Adam_dis
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate, name='Adam_dis')
        return loss, optimizer.minimize(loss, var_list=self.disor.vars)

    def optimizer_encoder(self):
        self.x_encoder = tf.placeholder(tf.float32, [self.batch_size, self.encoder(0)])
        with tf.variable_scope(self.name, reuse=True) as vs:
            f = self.encoder.feedforward(self.x_encoder)
            pred = self.disor.predict(f)
        loss = tf.reduce_mean(tf.square(pred - tf.ones_like(pred)))
        # name conflict, so rename the optimizer as Adam_en
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate, name='Adam_en')
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

    def save(self, name):
        save_dir = 'ckpt/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        saver = tf.train.Saver(self.vars)
        saver.save(self.sess, name)

    def restore(self, path):
        saver = tf.train.Saver(self.vars)
        saver.restore(self.sess, path)

    def visual(self, input, labels):
        f = self.sess.run([self.encoder.feedforward(input)], {self.x_encoder:input})
        x, y = [], []
        for n in range(10):
            pass





if __name__ == '__main__':
    encoder_layer = [28*28, 400, 100]
    z_dim = 2
    decoder_layer = [100, 400, 28*28]
    disor_layer = [2, 16, 1]
    num_epochs = 100
    batch_size = 100
    learn_rate = 1e-3
    shape = [batch_size, 28*28]

    sess = tf.Session()
    aae = AAE(sess, encoder_layer, z_dim, decoder_layer, disor_layer)
    aae.init_model()