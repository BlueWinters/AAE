
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tools import get_10color_list

from model import AAE
from tensorflow.examples.tutorials.mnist import input_data

encoder_layer = [28*28, 1000, 400, 100]
z_dim = 2
decoder_layer = [100, 400, 1000, 28*28]
disor_layer = [2, 32, 16, 1]


def visual_2d():
    # session
    sess = tf.Session()
    # model
    aae = AAE(sess)
    aae.restore('./ckpt/model.ckpt')
    # data read & train
    mnist = input_data.read_data_sets("mnist/", one_hot=True)


    images, labels = mnist.validation.images, mnist.validation.labels
    images = images.reshape([-1,28*28])
    f = aae.image_to_latent(images)
    plt.clf()
    color_list = get_10color_list()
    for n in range(10):
        index = np.where(labels[:,n] == 1)[0]
        point = f[index.tolist(),:]
        x = point[:,0]
        y = point[:,1]
        plt.scatter(x, y, color=color_list[n], edgecolors='face')
    plt.show()

def visual_image():
    sess = tf.Session()
    aae = AAE(sess)
    aae.restore('./ckpt/model.ckpt')

    # random sample from distribution
    n_samples = 100
    z = np.random.normal(size=[n_samples, 2]).astype('float32')
    # get image
    images = aae.latent_to_image(z)

    # plot
    figure, ax = plt.subplots(10, 10)
    for i in range(10):
        for j in range(10):
            ax[i][j].imshow(np.reshape(images[i*10+j,:], (28, 28)), cmap ='gray')
            ax[i][j].set_axis_off()

    figure.show()
    plt.draw()
    plt.waitforbuttonpress()

def reconstruction():
    sess = tf.Session()
    aae = AAE(sess)
    aae.restore('./ckpt/model.ckpt')

    mnist = input_data.read_data_sets("mnist/", one_hot=True)
    origin, _ = mnist.validation.next_batch(100)
    origin = origin.reshape([100, 28*28])
    output = aae.image_to_image(origin)

    figure, ax = plt.subplots(10, 10)
    for i in range(5,):
        for j in range(10):
            ax[i][j].imshow(np.reshape(output[i*10+j,:], (28, 28)), cmap ='gray')
            ax[i][j].set_axis_off()
    for i in range(5):
        for j in range(10):
            ax[5+i][j].imshow(np.reshape(origin[i*10+j,:], (28, 28)), cmap ='gray')
            ax[5+i][j].set_axis_off()
    figure.show()
    plt.draw()
    plt.waitforbuttonpress()


if __name__ == '__main__':
    visual_2d()
    # visual_image()
    # reconstruction()
    pass