
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
        self.encoder = Encoder(sess=sess, encoder=layer_encoder, z_dim=z_dim)
        self.z_dim = z_dim
        self.decoder = Decoder(sess=sess, decoder=layer_decoder, z_dim=z_dim)
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
        self.summary_writer.close()
        self.sess.close()

    def init_model(self):
        # initialize variables
        with tf.variable_scope(self.name) as vs:
            self.encoder.init_model()
            self.decoder.init_model()
            self.disor.init_model()
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # initialize optimizers
        self.loss_encoder_decoder, self.opt_encoder_decoder = self.optimizer_encoder_decoder()
        self.loss_disor_faker, self.loss_disor_real, self.opt_disor = self.optimizer_discriminator()
        self.loss_encoder, self.opt_encoder = self.optimizer_encoder()
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
        # initialize data

    def optimizer_encoder_decoder(self):
        self.x_encoder_decoder = tf.placeholder(tf.float32, [self.batch_size, self.encoder(0)])
        with tf.variable_scope(self.name, reuse=True) as vs:
            f = self.encoder.feedforward(self.x_encoder_decoder)
            y = self.decoder.feedforward(f)
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - self.x_encoder_decoder), [1]))

        # summary
        with tf.name_scope('encoder_decoder') as vs:
            tf.summary.scalar('loss', loss)

        # trainable variable for encoder-decoder
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

        # loss type 1
        # loss_faker = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_faker,
        #                                                                     labels=tf.zeros_like(pred_faker)))
        # loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_real,
        #                                                                    labels=tf.ones_like(pred_real)))

        # loss type 2
        loss_real = -tf.reduce_mean(tf.log(pred_real + self.tiny))
        loss_faker =  -tf.reduce_mean(tf.log(1. - pred_faker + self.tiny))

        # control dependency
        with tf.control_dependencies([loss_faker, loss_real]):
            loss = loss_faker + loss_real

        # summary
        with tf.name_scope('discriminator') as vs:
            tf.summary.scalar('loss_faker', loss_faker)
            tf.summary.scalar('loss_real', loss_real)
            tf.summary.scalar('loss', loss)

        # name conflict, so rename the optimizer as Adam_dis
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate, name='Adam_dis')
        return loss_faker, loss_real, optimizer.minimize(loss, var_list=self.disor.vars)

    def optimizer_encoder(self):
        self.x_encoder = tf.placeholder(tf.float32, [self.batch_size, self.encoder(0)])
        with tf.variable_scope(self.name, reuse=True) as vs:
            f = self.encoder.feedforward(self.x_encoder)
            pred = self.disor.predict(f)
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,
        #                                                               labels=tf.ones_like(pred)))
        loss = -tf.reduce_mean(tf.log(pred + self.tiny))

        with tf.name_scope('encoder') as vs:
            tf.summary.scalar('loss', loss)

        # name conflict, so rename the optimizer as Adam_en
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate, name='Adam_en')
        return loss, optimizer.minimize(loss, var_list=self.encoder.vars)

    def train_encoder_decoder(self, input, labels=None):
        _, loss, summary = self.sess.run([self.opt_encoder_decoder, self.loss_encoder_decoder,
                                          self.merged],
                                         {self.x_encoder_decoder:input})
        self.summary_writer.add_summary(summary)
        return loss

    def train_discriminator(self, input, labels=None):
        z_prior = self.sampler(self.batch_size)
        _, loss_faker, loss_real, summary = self.sess.run(
            [self.opt_disor, self.loss_disor_faker, self.loss_disor_real, self.merged],
            {self.x_discriminator:input, self.z_discriminator:z_prior})
        self.summary_writer.add_summary(summary)
        return loss_faker, loss_real

    def train_encoder(self, input, labels=None):
        _, loss, summary = self.sess.run([self.opt_encoder, self.loss_encoder, self.merged],
                                         {self.x_encoder:input})
        self.summary_writer.add_summary(summary)
        return loss

    def dist_to_image(self):
        with tf.name_scope('image') as vs:
            z = tf.random_normal(shape=[self.batch_size, self.z_dim])
            image = self.decoder.feedforward(z)
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
            f = self.sess.run(self.encoder.feedforward(input))
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
            image = self.sess.run(self.decoder.feedforward(z))
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