
import tensorflow as tf
import numpy as np

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator
from sampler import Sampler

import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow.examples.tutorials.mnist import input_data


def get_config_path():
    data_path = './mnist'
    summary_path = './summary'
    save_path = './save'
    return data_path, summary_path, save_path

def generate_image_grid(op):
    encoder = Encoder()
    decoder = Decoder()
    # discriminator = Discriminator()
    # sampler = Sampler()

    _, _, save_path = get_config_path()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, save_path=save_path)
        x_points = np.arange(-10, 10, 1.5).astype(np.float32)
        y_points = np.arange(-10, 10, 1.5).astype(np.float32)

        nx, ny = len(x_points), len(y_points)
        plt.subplot()
        gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

        for i, g in enumerate(gs):
            z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
            z = np.reshape(z, (1, 2))
            image = decoder.feed_forward(z)
            x = sess.run(image)
            ax = plt.subplot(g)
            img = np.array(x.tolist()).reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')
        plt.show()

def train():
    x_dim = 784
    z_dim = 2
    batch_size = 100
    n_epochs = 1000
    learn_rate = 0.001
    summary_step = 100

    data_path, summary_path, save_path = get_config_path()

    x = tf.placeholder(dtype=tf.float32, shape=[batch_size, x_dim], name='x')
    # y = tf.placeholder(dtype=tf.float32, shape=[batch_size, x_dim], name='y')
    z_real = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='z_real')
    # z_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim], name='z_input')

    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()
    sampler = Sampler()

    z = encoder.feed_forward(x)
    y = decoder.feed_forward(z)

    d_real = discriminator.feed_forward(z_real)
    d_fake = discriminator.feed_forward(z)

    # auto-encoder loss
    ae_loss = tf.reduce_mean(tf.square(x - y))

    # discriminator loss
    dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_real), logits=d_real))
    dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(d_fake), logits=d_fake))
    dc_loss = dc_loss_fake + dc_loss_real

    # generator loss
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_fake), logits=d_fake))

    all_variables = tf.trainable_variables()
    en_var = [var for var in all_variables if 'Encoder' in var.name]
    de_var = [var for var in all_variables if 'Decoder' in var.name]
    dis_var = [var for var in all_variables if 'Discriminator' in var.name]

     # optimizers
    auto_encoder_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(ae_loss)
    discriminator_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(dc_loss, var_list=dis_var)
    generator_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(gen_loss, var_list=en_var)

    # initialize variables
    init = tf.global_variables_initializer()

    # reshape images to display them
    input_images = tf.reshape(x, [-1, 28, 28, 1])
    reconstruct_images = tf.reshape(y, [-1, 28, 28, 1])

    # tensorboard visualization
    tf.summary.scalar(name='Auto-encoder Loss', tensor=ae_loss)
    tf.summary.scalar(name='Discriminator Loss', tensor=dc_loss)
    tf.summary.scalar(name='Generator Loss', tensor=gen_loss)
    tf.summary.histogram(name='Encoder Distribution', values=z)
    tf.summary.histogram(name='Real Distribution', values=z_real)
    tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
    tf.summary.image(name='Reconstructed Images', tensor=reconstruct_images, max_outputs=10)
    summary_op = tf.summary.merge_all()

    # counter
    step = 0

    # data
    mnist = input_data.read_data_sets(data_path, one_hot=True)
    n_batches = int(mnist.train.num_examples / batch_size)

    # train the model
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir=summary_path, graph=sess.graph)

        # training process
        for epochs in range(n_epochs):
            for n in range(1, n_batches + 1):
                z_real_s = sampler()
                batch_x, _ = mnist.train.next_batch(batch_size)
                sess.run(auto_encoder_optimizer, feed_dict={x: batch_x})
                sess.run(discriminator_optimizer, feed_dict={x:batch_x, z_real:z_real_s})
                sess.run(generator_optimizer, feed_dict={x:batch_x})
                # summary
                if n % summary_step == 0:
                    ae_loss, dc_r_loss, dc_f_loss, gen_loss, summary = sess.run(
                        [ae_loss, dc_loss_fake, dc_loss_real, gen_loss, summary_op],
                        feed_dict={x: batch_x, z_real:z_real_s})
                    writer.add_summary(summary, global_step=step)

                    liner = "Epoch {:3d}/{:d}, loss_en_de {:9f}, " \
                            "loss_dis_faker {:9f}, loss_dis_real {:9f}, loss_encoder {:9f}"\
                        .format(epochs, n, ae_loss, dc_f_loss, dc_r_loss, gen_loss)
                    print(liner)

                    with open(summary_path + '/log.txt', 'a') as log:
                        log.write(liner)
                step += 1

        # save model
        saver = tf.train.Saver()
        saver.save(sess, save_path=save_path)