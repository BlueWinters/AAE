
import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator
from sampler import Sampler

from tensorflow.examples.tutorials.mnist import input_data


def get_config_path():
    data_path = './mnist'
    summary_path = 'unsupervised/summary'
    save_path = 'unsupervised/ckpt/model'
    return data_path, summary_path, save_path

def train():
    x_dim = 784
    z_dim = 2
    batch_size = 100
    n_epochs = 1000
    learn_rate = 0.001
    summary_step = 100

    data_path, summary_path, save_path = get_config_path()

    x = tf.placeholder(dtype=tf.float32, shape=[batch_size, x_dim], name='x')
    z_real = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='z_real')

    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()
    sampler = Sampler()

    z = encoder.feed_forward(x)
    y = decoder.feed_forward(z)

    d_real = discriminator.feed_forward(z_real)
    d_fake = discriminator.feed_forward(z)

    # auto-encoder loss
    ae_loss = tf.reduce_mean(tf.square(x - y))

    # discriminator loss
    dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_real), logits=d_real))
    dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(d_fake), logits=d_fake))
    with tf.control_dependencies([dc_loss_fake, dc_loss_real]):
        dc_loss = dc_loss_fake + dc_loss_real

    # generator loss
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_fake), logits=d_fake))

    all_variables = tf.trainable_variables()
    en_var = [var for var in all_variables if 'Encoder' in var.name]
    de_var = [var for var in all_variables if 'Decoder' in var.name]
    dis_var = [var for var in all_variables if 'Discriminator' in var.name]

    # optimizers
    auto_encoder_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(ae_loss)
    discriminator_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(dc_loss, var_list=dis_var)
    generator_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(gen_loss, var_list=en_var)

    # reshape images to display them
    input_images = tf.reshape(x, [-1, 28, 28, 1])
    reconstruct_images = tf.reshape(y, [-1, 28, 28, 1])

    # tensorboard visualization
    tf.summary.scalar(name='Auto-encoder Loss', tensor=ae_loss)
    tf.summary.scalar(name='Discriminator Loss', tensor=dc_loss)
    tf.summary.scalar(name='Generator Loss', tensor=gen_loss)
    tf.summary.histogram(name='Encoder Distribution', values=z)
    tf.summary.histogram(name='Real Distribution', values=z_real)
    tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
    tf.summary.image(name='Reconstructed Images', tensor=reconstruct_images, max_outputs=10)
    summary_op = tf.summary.merge_all()

    # counter
    step = 0

    # data
    mnist = input_data.read_data_sets(data_path, one_hot=True)
    n_batches = int(mnist.train.num_examples / batch_size)

    # train the model
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir=summary_path, graph=sess.graph)
        sess.run(tf.global_variables_initializer())

        # training process
        for epochs in range(n_epochs):
            for n in range(1, n_batches + 1):
                z_real_s = sampler(batch_size)
                batch_x, _ = mnist.train.next_batch(batch_size)
                sess.run(auto_encoder_optimizer, feed_dict={x:batch_x})
                sess.run(discriminator_optimizer, feed_dict={x:batch_x, z_real:z_real_s})
                sess.run(generator_optimizer, feed_dict={x:batch_x})
            # summary
            vloss_ae, vloss_dc_f, vloss_dc_r, vloss_gen, summary = sess.run(
                [ae_loss, dc_loss_fake, dc_loss_real, gen_loss, summary_op],
                feed_dict={x: batch_x, z_real:z_real_s})
            writer.add_summary(summary, global_step=epochs)

            liner = "Epoch {:3d}/{:d}, loss_en_de {:9f}, " \
                    "loss_dis_faker {:9f}, loss_dis_real {:9f}, loss_encoder {:9f}" \
                .format(epochs, n_epochs, vloss_ae, vloss_dc_f, vloss_dc_r, vloss_gen)
            print(liner)

            with open(summary_path + '/log.txt', 'a') as log:
                log.write(liner)

        # save model
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver = tf.train.Saver(var_list=vars)
        saver.save(sess, save_path=save_path)

if __name__ == '__main__':
    train()
