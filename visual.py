
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import AAE
from tensorflow.examples.tutorials.mnist import input_data

encoder_layer = [28*28, 400, 100]
z_dim = 2
decoder_layer = [100, 400, 28*28]
disor_layer = [2, 16, 1]


def visual_2d():
    # session
    sess = tf.Session()
    # model
    aae = AAE(sess, encoder_layer, z_dim, decoder_layer, disor_layer)
    # data read & train
    mnist = input_data.read_data_sets("mnist/", one_hot=True)
    # visual
    aae.visual(mnist.validation.images, mnist.validation.labels)

def visual_image():
    # session
    sess = tf.Session()
    # # model
    aae = AAE(sess, encoder_layer, z_dim, decoder_layer, disor_layer)

    # random sample from distribution
    n_samples = 100
    z = np.random.normal(size=[n_samples, 2]).astype('float32')
    # get image
    images = aae.output(z)

    # images = np.random.normal(size=[100, 784]).astype('float32')

    # plot
    figure, ax = plt.subplots(10, 10)
    for i in range(10):
        for j in range(10):
            ax[i][j].imshow(np.reshape(images[i*10+j,:], (28, 28)), cmap ='gray')
            ax[i][j].set_axis_off()
    figure.show()
    plt.draw()
    plt.waitforbuttonpress()


if __name__ == '__main__':
    # visual_2d()
    visual_image()