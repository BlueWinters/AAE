

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist/", one_hot=True)
batch_x, batch_y = mnist.train.next_batch(50)
x = 1