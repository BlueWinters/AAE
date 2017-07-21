
import tensorflow as tf

import time

from model import AAE
from tensorflow.examples.tutorials.mnist import input_data


encoder_layer = [28*28, 400, 100]
z_dim = 2
decoder_layer = [100, 400, 28*28]
disor_layer = [2, 16, 1]
num_epochs = 100
num_epochs_en_de = 2
num_epochs_dis = 1
num_epochs_en = 1
vis_epochs = 5
batch_size = 100
learn_rate = 1e-3
shape = [batch_size, 28*28]


sess = tf.Session()
aae = AAE(sess, encoder_layer, z_dim, decoder_layer, disor_layer)

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
        for epoch_en_de in range(num_epochs_en_de):
            loss_encoder_decoder += aae.train_encoder_decoder(input=batch_x)/num_epochs_en_de
        for epoch_dis in range(num_epochs_dis):
            loss_discriminator += aae.train_discriminator(input=batch_x)/num_epochs_dis
        for epoch_en in range(num_epochs_en):
            loss_encoder += aae.train_encoder(input=batch_x)/num_epochs_en
    if (epoch+1) % vis_epochs == 0:
        aae.visual(batch_x, batch_y)
    print("Epoch {:3d}/{:d}, loss_en_de {:9f}, loss_dis {:9f}, loss_encoder {:9f}"
          .format(num_epochs, epoch+1, loss_encoder_decoder, loss_discriminator, loss_encoder))
print("Training time : {}".format(time.time() - start_time))

#


# save model
aae.save('model.ckpt')
