
import tensorflow as tf
import os
import time

from autoencoder import Autoencoder
from tensorflow.examples.tutorials.mnist import input_data

encoder = [28*28]
z_dim = 1000
decoder = [28*28]
num_epochs = 100
batch_size = 100
learn_rate = 1e-3
shape = [batch_size, 28*28]


sess = tf.Session()
ae = Autoencoder(sess, encoder=encoder, z_dim=z_dim, decoder=decoder)
input = tf.placeholder(tf.float32, shape)
loss = ae.loss(input)

optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
train = optimizer.minimize(loss, var_list=ae.vars)
sess.run(tf.global_variables_initializer())

# data read & train
mnist = input_data.read_data_sets("mnist/", one_hot=True)

# train model
start_time = time.time()
for epoch in range(num_epochs):
    total_batch = int(mnist.train.num_examples / batch_size)
    avg_loss = 0
    for i in range(total_batch):
        batch_x, _ = mnist.train.next_batch(batch_size)

        batch_x = batch_x.reshape(shape)
        l, _ = sess.run([loss, train], {input: batch_x})
        avg_loss += l / total_batch

    print("Epoch : {:04d}, Loss : {:.9f}".format(epoch + 1, avg_loss))
print("Training time : {}".format(time.time() - start_time))

# save model
ckpt_dir = "ckpt/"
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)
ckpt_path = "ckpt/model.ckpt"
ae.save(ckpt_path)
