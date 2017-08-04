
import tensorflow as tf
import os as os

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator
from sampler import Sampler


class AAE(object):
    def __init__(self, sess, in_dim=784, z_dim=2,
                 batch_size=100, learn_rate=0.0001, name='AAE'):
        # model parameters
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.disor = Discriminator()
        self.sampler = Sampler()
        self.in_dim = in_dim
        self.z_dim = z_dim

        # training config
        self.name = name
        self.sess = sess
        self.summary_path = './summary'
        self.tiny = 1e-8
        # other
        self.batch_size = batch_size
        self.learn_rate = learn_rate

        # initialize model
        self._init_model()

    def __del__(self):
        self.sess.close()

    def _init_model(self):
        # initialize placeholder
        self.x_en_de = tf.placeholder(tf.float32, [self.batch_size, self.in_dim], 'input_en_de')
        self.x_disor = tf.placeholder(tf.float32, [self.batch_size, self.in_dim], 'input_disor')
        self.x_en = tf.placeholder(tf.float32, [self.batch_size, self.in_dim], 'input_en')
        self.z_real = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], 'z_real')


        # initialize optimizers
        self.optimizer_encoder_decoder()
        self.optimizer_discriminator()
        self.optimizer_encoder()

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # summary
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)
        # initialize variable
        self.sess.run(tf.global_variables_initializer())

    def get_variables(self):
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.vars_en_de = [vars for vars in self.vars if self.encoder.name in vars.name]
        self.vars_disor = [vars for vars in self.vars if self.encoder.name in vars.name]

    def optimizer_encoder_decoder(self):
        z = self.encoder.feed_forward(self.x_en_de)
        y = self.decoder.feed_forward(z)

        with tf.name_scope('loss_encoder_decoder'):
            # loss = self.x_en_de * tf.log(y + self.tiny) + (1. - self.x_en_de) * tf.log(1 - y + self.tiny)
            # self.loss_encoder_decoder = tf.reduce_mean(- tf.reduce_sum(loss, axis=1))
            loss = tf.reduce_sum(tf.square(y - self.x_en_de), axis=1)
            self.loss_encoder_decoder = tf.reduce_mean(loss)
            tf.summary.scalar('reconstruction', self.loss_encoder_decoder)

        # vars = self.encoder.get_variable()
        # vars.extend(self.decoder.vars)

        with tf.name_scope('trainer_encoder_decoder'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
            self.trainer_encoder_decoder = optimizer.minimize(self.loss_encoder_decoder)

    def optimizer_discriminator(self):
        z_faker = self.encoder.feed_forward(self.x_disor)
        pred_faker = self.disor.feed_forward(z_faker)

        # self.z_real = tf.random_normal([self.batch_size, self.z_dim]) * 5
        pred_real = self.disor.feed_forward(self.z_real)

        with tf.name_scope('loss_discriminator'):
            self.loss_disor_real = -tf.reduce_mean(tf.log(pred_real + self.tiny))
            self.loss_disor_faker =  -tf.reduce_mean(tf.log(1. - pred_faker + self.tiny))
            # with tf.control_dependencies([self.loss_disor_faker, self.loss_disor_real]):
            self.loss_disor = self.loss_disor_faker + self.loss_disor_real
            tf.summary.scalar('loss_faker', self.loss_disor_faker)
            tf.summary.scalar('loss_real', self.loss_disor_real)

        with tf.name_scope('trainer_disor'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
            self.trainer_disor = optimizer.minimize(self.loss_disor, var_list=self.disor.vars)
            # print(len(self.disor.vars))

    def optimizer_encoder(self):
        z = self.encoder.feed_forward(self.x_en)
        pred = self.disor.feed_forward(z)
        with tf.name_scope('loss_encoder'):
            self.loss_encoder = -tf.reduce_mean(tf.log(pred + self.tiny))
            tf.summary.scalar('loss_encoder', self.loss_encoder)

        with tf.name_scope('trainer_encoder'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
            self.trainer_encoder = optimizer.minimize(self.loss_encoder, var_list=self.encoder.vars)

    def train_encoder_decoder(self, input):
        _, loss = self.sess.run([self.trainer_encoder_decoder, self.loss_encoder_decoder],
                                {self.x_en_de:input})
        return loss

    def train_discriminator(self, input):
        z_prior = self.sampler(self.batch_size)
        _, loss_faker, loss_real = self.sess.run(
            [self.trainer_disor, self.loss_disor_faker, self.loss_disor_real],
            {self.x_disor:input, self.z_real:z_prior})
        return loss_faker, loss_real

    def train_encoder(self, input):
        _, loss = self.sess.run([self.trainer_encoder, self.loss_encoder],
                                {self.x_en:input})
        return loss

    def save(self, name):
        save_dir = 'ckpt/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        saver = tf.train.Saver(self.vars)
        saver.save(self.sess, save_dir+name)

    def restore(self, path):
        saver = tf.train.Saver(self.vars)
        saver.restore(self.sess, path)

    def image_to_latent(self, input):
        return self.sess.run(self.encoder.feed_forward(input, False))

    def image_to_image(self, input):
        z = self.sess.run(self.encoder.feed_forward(input, False))
        return self.sess.run(self.decoder.feed_forward(z, False))

    def latent_to_image(self, z):
        return self.sess.run(self.decoder.feed_forward(z, False))


if __name__ == '__main__':
    encoder_layer = [28*28, 1000, 1000]
    z_dim = 2
    decoder_layer = [1000, 1000, 28*28]
    disor_layer = [2, 1000, 1000, 1]
    num_epochs = 100
    batch_size = 100
    learn_rate = 1e-3
    shape = [batch_size, 28*28]

    sess = tf.Session()
    aae = AAE(sess)