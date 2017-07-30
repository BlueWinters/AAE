
import tensorflow as tf
import time

from model import AAE
from tensorflow.examples.tutorials.mnist import input_data


encoder_layer = [28*28, 1000, 1000]
z_dim = 2
decoder_layer = [1000, 1000, 28*28]
disor_layer = [2, 1000, 1000, 1]
num_epochs = 100
num_epochs_en_de = 1
num_epochs_dis = 1
num_epochs_en = 1
vis_epochs = 5
batch_size = 100
learn_rate = 1e-3
shape = [batch_size, 28*28]
summary_path = './summary'


sess = tf.Session()
aae = AAE(sess, encoder_layer, z_dim, decoder_layer, disor_layer)

# data read & train
mnist = input_data.read_data_sets("mnist/", one_hot=True)

# train model
start_time = time.time()
for epoch in range(num_epochs):
    total_batch = int(mnist.train.num_examples / batch_size)
    loss_encoder_decoder = 0.
    loss_disor_faker = 0.
    loss_disor_real = 0.
    loss_encoder = 0.
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape(shape)
        for epoch_en_de in range(num_epochs_en_de):
            tmp_en_de = aae.train_encoder_decoder(input=batch_x)
            loss_encoder_decoder += tmp_en_de/float(num_epochs_en_de*total_batch)
        for epoch_dis in range(num_epochs_dis):
            tmp_faker, tmp_real = aae.train_discriminator(input=batch_x)
            loss_disor_faker += tmp_faker/float(num_epochs_dis*total_batch)
            loss_disor_real += tmp_real/float(num_epochs_dis*total_batch)
        for epoch_en in range(num_epochs_en):
            loss_encoder += aae.train_encoder(input=batch_x)/float(num_epochs_en*total_batch)
    print("Epoch {:3d}/{:d}, loss_en_de {:9f}, loss_dis_faker {:9f}, loss_dis_real {:9f}, loss_encoder {:9f}"
          .format(epoch+1, num_epochs, loss_encoder_decoder, loss_disor_faker, loss_disor_real, loss_encoder))


print("Training time : {}".format(time.time() - start_time))

# save model
aae.save('model.ckpt')
