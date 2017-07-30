
import tensorflow as tf
import numpy as np
import os as os
import matplotlib.pyplot as plt

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator
from sampler import Sampler
from tools import get_10color_list


class AAE(object):
    def __init__(self, sess, layer_encoder, z_dim, layer_decoder, layer_disor,
                 batch_size=100, learn_rate=0.0001, prior_type='Gaussiss',
                 summary_path ='./summary', name='AAE'):
        # model parameters
        self.encoder = Encoder(encoder=layer_encoder, z_dim=z_dim)
        self.z_dim = z_dim
        self.decoder = Decoder(decoder=layer_decoder, z_dim=z_dim)
        self.disor = Discriminator(z_dim=z_dim, layers=layer_disor)
        self.sampler = Sampler(type=prior_type, dim=z_dim)
        # training config
        self.name = name
        self.sess = sess
        self.summary_path = summary_path
        self.tiny = 1e-8
        # other
        self.batch_size = batch_size
        self.learn_rate = learn_rate

        # initialize model
        self.init_model()

    def __del__(self):
        # self.summary_writer.close()
        self.sess.close()

    def init_model(self):
        # initialize placeholder
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.encoder(0)], 'input')
        self.z_real = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], 'z_real')

        # initialize optimizers
        self.optimizer_encoder_decoder()
        self.optimizer_discriminator()
        self.optimizer_encoder()

        # summary
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)
        # initialize variable
        self.sess.run(tf.global_variables_initializer())

    def  init_config(self):
        # initialize summary path
        if tf.gfile.Exists(self.summary_path):
            tf.gfile.DeleteRecursively(self.summary_path)
        tf.gfile.MakeDirs(self.summary_path)

    def optimizer_encoder_decoder(self):
        self.z_faker = self.encoder.feed_forward(self.x)
        self.y = self.decoder.feed_forward(self.z_faker)

        with tf.name_scope('trainer_encoder_decoder'):
            with tf.name_scope('loss_encoder_decoder'):
                loss = tf.square(self.y - self.x)
                loss = tf.reduce_sum(loss, [1])
                self.loss_encoder_decoder = tf.reduce_mean(loss)
            vars = self.encoder.vars
            vars.extend(self.decoder.vars)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
            self.trainer_encoder_decoder = optimizer.minimize(self.loss_encoder_decoder,
                                                              var_list=vars)
            tf.summary.scalar('loss', self.loss_encoder_decoder)

    def optimizer_discriminator(self):
        with tf.name_scope('concat') as scope:
            z = tf.concat(values=[self.z_faker, self.z_real], axis=0)
        pred = self.disor.predict(z)

        with tf.name_scope('slice') as scope:
            # self.pred_faker, self.pred_real = tf.split(0, 2, pred)
            self.pred_faker = tf.slice(pred, [0,0], [0,-1])
            self.pred_real = tf.slice(pred, [0,0], [self.batch_size,-1])

        with tf.name_scope('trainer_discriminator'):
            with tf.name_scope('loss_discriminator'):
                self.loss_disor_real = -tf.reduce_mean(tf.log(self.pred_real + self.tiny))
                self.loss_disor_faker =  -tf.reduce_mean(tf.log(1. - self.pred_faker + self.tiny))
            # control dependency
            with tf.control_dependencies([self.loss_disor_faker, self.loss_disor_real]):
                self.loss_disor = self.loss_disor_faker + self.loss_disor_real
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
            self.trainer_disor = optimizer.minimize(self.loss_disor, var_list=self.disor.vars)

            # summary
            tf.summary.scalar('loss_faker', self.loss_disor_faker)
            tf.summary.scalar('loss_real', self.loss_disor_real)
            tf.summary.scalar('loss', self.loss_disor)

    def optimizer_encoder(self):
        with tf.name_scope('trainer_encoder'):
            with tf.name_scope('loss_encoder'):
                self.loss_encoder = -tf.reduce_mean(tf.log(self.pred_faker + self.tiny))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
            self.trainer_encoder = optimizer.minimize(self.loss_encoder,
                                                      var_list=self.encoder.vars)
            # summary
            tf.summary.scalar('loss', self.loss_encoder)

    def train_encoder_decoder(self, input):
        _, loss = self.sess.run([self.trainer_encoder_decoder, self.loss_encoder_decoder],
                                {self.x:input})
        return loss

    def train_discriminator(self, input):
        z_prior = self.sampler(self.batch_size)
        _, loss_faker, loss_real = self.sess.run(
            [self.trainer_disor, self.loss_disor_faker, self.loss_disor_real],
            {self.x:input, self.z_real:z_prior})
        return loss_faker, loss_real

    def train_encoder(self, input):
        _, loss = self.sess.run([self.trainer_encoder, self.loss_encoder],
                                {self.x:input})
        return loss

    def dist_to_image(self):
        with tf.name_scope('image') as vs:
            z = tf.random_normal(shape=[self.batch_size, self.z_dim])
            image = self.decoder.feed_forward(z)
            image = tf.reshape(image, [10, 28, 28, 1])
            tf.summary.image('image', image, max_outputs=10)

    def save(self, name):
        save_dir = 'ckpt/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        saver = tf.train.Saver(self.vars)
        saver.save(self.sess, save_dir+name)

    def restore(self, path):
        saver = tf.train.Saver(self.vars)
        saver.restore(self.sess, path)

    def visual(self, input, labels):
        with tf.variable_scope(self.name, reuse=True) as vs:
            f = self.sess.run(self.encoder.feed_forward(input))
        point = []
        plt.clf()
        color_list = get_10color_list()
        for n in range(10):
            index = np.where(labels[:,n] == 1)[0]
            point = f[index.tolist(),:]
            x = point[:,0]
            y = point[:,1]
            plt.scatter(x, y, color=color_list[n], edgecolors='face')
        plt.show()

    def output(self, z):
        with tf.variable_scope(self.name, reuse=True) as vs:
            image = self.sess.run(self.decoder.feed_forward(z))
        return image


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