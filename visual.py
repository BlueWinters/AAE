
import tensorflow as tf
import numpy as np

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator

import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import norm
from tensorflow.examples.tutorials.mnist import input_data


def get_config_path():
    data_path = './mnist'
    summary_path = './summary'
    save_path = 'ckpt/model'
    return data_path, summary_path, save_path

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

        x_points = np.arange(-10, 10, 0.5).astype(np.float32)
        y_points = np.arange(-10, 10, 0.5).astype(np.float32)
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

def explore_latent():
    nx, ny = 30, 30
    space_min, space_max = -40, 40
    tiny = 1e-8
    z = np.rollaxis(np.mgrid[space_max:space_min:ny*1j, space_min:space_max:nx*1j], 0, 3)
    z = np.array([norm.ppf(np.clip(one, tiny, 1 - tiny)) for one in z])
    z = np.reshape(z, [-1,2])

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
            image_holder = decoder.feed_forward(z_holder, is_train=False)
        image = sess.run(image_holder, feed_dict={z_holder:z})
        image = np.reshape(image, [ny,nx,-1])

        stack_image = np.zeros([ny*28,nx*28])
        for j in range(ny):
            for i in range(nx):
                stack_image[j*28:(j+1)*28, i*28:(i+1)*28] = np.reshape(image[j,i,:], [28,28])

        plt.imshow(stack_image, cmap='gray')
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

def visual_2d():
    def get_10color_list():
        color = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                 [0.0, 0.0, 1.0], [0.0, 1.0, 1.0],
                 [1.0, 0.0, 0.0], [1.0, 0.0, 1.0],
                 [1.0, 1.0, 0.0], [1.0, 1.0, 0.5], # [1,1,1] --> white
                 [0.5, 1.0, 1.0], [1.0, 0.5, 1.0]] # last three are chosen randomly
        return color
    #
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
        images, labels = mnist.validation.images, mnist.validation.labels

        with tf.name_scope('reconstruction'):
            input = tf.placeholder(tf.float32, [None,784], 'input')
            z = encoder.feed_forward(input, is_train=False)

        images = images.reshape([-1, 28*28])
        all_point = sess.run(z, feed_dict={input:images})

        color_list = get_10color_list()
        for n in range(10):
            index = np.where(labels[:,n] == 1)[0]
            point = all_point[index.tolist(),:]
            x = point[:,0]
            y = point[:,1]
            plt.scatter(x, y, color=color_list[n], edgecolors='face')
        plt.show()

if __name__ == '__main__':
    # generate_image_grid()
    # generate_reconstruct_image()
    visual_2d()
    # explore_latent()
    pass