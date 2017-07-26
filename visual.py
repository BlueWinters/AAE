
import tensorflow as tf
import time

from model import AAE
from tensorflow.examples.tutorials.mnist import input_data

encoder_layer = [28*28, 400, 100]
z_dim = 2
decoder_layer = [100, 400, 28*28]
disor_layer = [2, 16, 1]


# session
sess = tf.Session()
# model
aae = AAE(sess, encoder_layer, z_dim, decoder_layer, disor_layer)

# data read & train
mnist = input_data.read_data_sets("mnist/", one_hot=True)

# visual
aae.visual(mnist.validation.images, mnist.validation.labels)