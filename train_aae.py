
import tensorflow as tf
import os
import time

from model import AAE
from tensorflow.examples.tutorials.mnist import input_data

encoder = [28*28, 500, 100]
z_dim = 2
decoder = [100, 500, 28*28]
disor = []
num_epochs = 100
batch_size = 100
learn_rate = 1e-3
shape = [batch_size, 28*28]


sess = tf.Session()
aae = AAE(sess, encoder=encoder, z_dim=z_dim, decoder=decoder, disor=disor)
aae.init_model()

# data read & train
mnist = input_data.read_data_sets("mnist/", one_hot=True)

# train model
start_time = time.time()
for epoch in range(num_epochs):
    total_batch = int(mnist.train.num_examples / batch_size)
    loss_encoder_decoder = 0.
    loss_discriminator = 0.
    loss_encoder = 0.
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape(shape)
        loss_encoder_decoder += aae.train_encoder_decoder(input=batch_x)
        loss_discriminator += aae.train_discriminator(input=batch_x)
        loss_encoder += aae.train_encoder(input=batch_x)
    print("Epoch {3d}: loss_en_de {:9f}, loss_dis {:9f}, loss_encoder {:9f}"
          .format(epoch+1, loss_encoder_decoder, loss_discriminator, loss_encoder))
print("Training time : {}".format(time.time() - start_time))

# save model
ckpt_dir = "ckpt/"
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)
ckpt_path = "ckpt/model.ckpt"
aae.save(ckpt_path)
