
import tensorflow as tf
import numpy as np

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator
from model import get_config_path

import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow.examples.tutorials.mnist import input_data


def generate_image_grid():
    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    data_path, _, save_path = get_config_path()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(vars)
        saver.restore(sess, save_path=save_path)

        with tf.name_scope('latent_space'):
            z_holder = tf.placeholder(dtype=tf.float32, shape=[None,2], name='z_holder')
            image = decoder.feed_forward(z_holder, is_train=False)

        x_points = np.arange(-10, 10, 1.5).astype(np.float32)
        y_points = np.arange(-10, 10, 1.5).astype(np.float32)
        nx, ny = len(x_points), len(y_points)

        plt.subplot()
        gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)
        for i, g in enumerate(gs):
            z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
            z = np.reshape(z, (1, 2))
            x = sess.run(image, feed_dict={z_holder:z})
            ax = plt.subplot(g)
            img = np.array(x.tolist()).reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')
        plt.show()

def generate_reconstruct_image():
    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    data_path, _, save_path = get_config_path()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(vars)
        saver.restore(sess, save_path=save_path)

        mnist = input_data.read_data_sets(data_path, one_hot=True)
        x, _ = mnist.validation.next_batch(100)

        with tf.name_scope('reconstruction'):
            input = tf.placeholder(tf.float32, [100,784], 'input')
            z = encoder.feed_forward(input, is_train=False)
            output = decoder.feed_forward(z, is_train=False)

        x = x.reshape([100, 28*28])
        y = sess.run(output, feed_dict={input:x})

        figure, ax = plt.subplots(10, 10)
        for i in range(5):
            for j in range(10):
                ax[i][j].imshow(np.reshape(x[i*10+j,:], (28, 28)), cmap ='gray')
                ax[i][j].set_axis_off()
        for i in range(5):
            for j in range(10):
                ax[5+i][j].imshow(np.reshape(y[i*10+j,:], (28, 28)), cmap ='gray')
                ax[5+i][j].set_axis_off()
        plt.show()



if __name__ == '__main__':
    generate_image_grid()
    # generate_reconstruct_image()