
import tensorflow as tf
import numpy as np
import datafactory as df

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator
from tools import get_meshgrid
import matplotlib.pyplot as plt
import scipy.misc as misc



def generate_image_grid(save_path, x_dim):
    encoder = Encoder(in_dim=x_dim, h_dim=1024, out_dim=2)
    decoder = Decoder(in_dim=2, h_dim=1024, out_dim=x_dim)
    discriminator = Discriminator(in_dim=2, h_dim=1024, out_dim=2)

    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    vars = [var for var in all_vars if 'Decoder' in var.name]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(vars)
        saver.restore(sess, save_path='{}\model'.format(save_path))

        with tf.name_scope('latent_space'):
            z_holder = tf.placeholder(dtype=tf.float32, shape=[None,2], name='z_holder')
            image = decoder.feed_forward(z_holder, is_train=False)

        nx, ny = 21, 21
        size, chl = 28, 1
        z_sample = get_meshgrid(z_range=1, nx=nx, ny=ny)
        images = sess.run(image, feed_dict={z_holder: z_sample})


        stack_images = np.zeros([ny * size, nx * size])
        for j in range(ny):
            for i in range(nx):
                stack_images[j * size:(j + 1) * size, i * size:(i + 1) * size] = \
                    np.reshape(images[j * ny + i, :], [size, size])
        misc.imsave('{}/mainfold.png'.format(save_path), stack_images)

def visual_2d(save_path, x_dim):
    y_dim = 10
    z_dim = 2

    encoder = Encoder(in_dim=x_dim, h_dim=1024, out_dim=2)
    decoder = Decoder(in_dim=2, h_dim=1000, out_dim=x_dim)
    discriminator = Discriminator(in_dim=z_dim, h_dim=1024, out_dim=2)

    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    vars = [var for var in all_vars if 'Encoder' in var.name]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(vars)
        saver.restore(sess, save_path='{}\model'.format(save_path))

        images, labels = df.load_mnist_train('E:/dataset/mnist')

        with tf.name_scope('reconstruction'):
            input = tf.placeholder(dtype=tf.float32, shape=[None, x_dim], name='input')
            z = encoder.feed_forward(input, is_train=False)

        all_point = sess.run(z, feed_dict={input:images})

        color_list = plt.get_cmap('hsv', y_dim+1)
        for n in range(y_dim):
            index = np.where(labels[:,n] == 1)[0]
            point = all_point[index.tolist(),:]
            x = point[:,0]
            y = point[:,1]
            plt.scatter(x, y, color=color_list(n), edgecolors='face')
        plt.show()
        # plt.savefig(save_path+'/visual2d.png')

if __name__ == '__main__':
    # generate_image_grid('save/unsupervised/mnist/gaussian', 784)
    visual_2d('save/supervised/mnist/mix-gaussian', 784)
    pass
