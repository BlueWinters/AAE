
import tensorflow as tf
import sampler as spl
import mnist_tools as mtl

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator



def get_config_path():
    data_path = 'mnist'
    summary_path = 'semi-supervised/summary'
    save_path = 'semi-supervised/ckpt/model'
    return data_path, summary_path, save_path

def ave_loss(ave_lost_list, step_loss_list, div):
    assert len(ave_lost_list) == len(step_loss_list)
    for n in range(len(ave_lost_list)):
        ave_lost_list[n] += step_loss_list[n] / div

def train():
    x_dim = 784
    y_dim = 10
    z_dim = 2
    batch_size = 100 # for both supervised and unsupervised
    n_epochs = 100
    learn_rate = 0.0001

    data_path, summary_path, save_path = get_config_path()

    # x = tf.placeholder(dtype=tf.float32, shape=[None, x_dim], name='x')
    x_l = tf.placeholder(dtype=tf.float32, shape=[None, x_dim], name='x_l')
    x_u = tf.placeholder(dtype=tf.float32, shape=[None, x_dim], name='x_u')
    z_real_l = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='z_real_l')
    z_real_u = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='z_real_u')
    y_fake = tf.placeholder(dtype=tf.float32, shape=[None, y_dim], name='y_fake')
    y_real = tf.placeholder(dtype=tf.float32, shape=[None, y_dim], name='y_real')

    tensor_one = tf.ones(shape=[batch_size, 1])
    tensor_zero = tf.zeros(shape=[batch_size, 1])

    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator(in_dim=13)

    # auto-encoder loss
    z_fake_l = encoder.feed_forward(x_l)
    z_fake_u = encoder.feed_forward(x_u)
    x_hat_l = decoder.feed_forward(z_fake_l)
    x_hat_u = decoder.feed_forward(z_fake_u)
    A_loss = tf.reduce_mean(tf.square(x_l-x_hat_l)+tf.square(x_u-x_hat_u))

    # discriminator loss
    z_fake_concat_l = tf.concat([z_fake_l, y_fake, tensor_zero], axis=1)
    d_out_fake_l = discriminator.feed_forward(z_fake_concat_l)
    z_fake_concat_u = tf.concat([z_fake_u, tf.zeros_like(y_fake), tensor_one], axis=1)
    d_out_fake_u = discriminator.feed_forward(z_fake_concat_u)

    z_real_concat_l = tf.concat([z_real_l, y_real, tensor_zero], axis=1)
    d_out_real_l = discriminator.feed_forward(z_real_concat_l)
    z_real_concat_u = tf.concat([z_real_u, tf.zeros_like(y_real), tensor_one], axis=1)
    d_out_real_u = discriminator.feed_forward(z_real_concat_u)

    # construct labels for discriminator
    labels_real = tf.concat([tensor_one, tensor_zero], axis=1)
    labels_fake = tf.concat([tensor_zero, tensor_one], axis=1)

    D_loss_real_l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_real, logits=d_out_real_l))
    D_loss_real_u = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_real, logits=d_out_real_u))
    D_loss_fake_l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_fake, logits=d_out_fake_l))
    D_loss_fake_u = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_fake, logits=d_out_fake_u))
    with tf.control_dependencies([d_out_real_l, d_out_real_u, d_out_fake_l, d_out_fake_u]):
        D_loss = D_loss_real_l + D_loss_real_u + D_loss_fake_l + D_loss_fake_u

    # generator loss
    G_loss_l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_real, logits=d_out_fake_l))
    G_loss_u = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_real, logits=d_out_fake_u))
    G_loss = G_loss_l + G_loss_u

    all_variables = tf.trainable_variables()
    en_var = [var for var in all_variables if 'Encoder' in var.name]
    de_var = [var for var in all_variables if 'Decoder' in var.name]
    dis_var = [var for var in all_variables if 'Discriminator' in var.name]

    # optimizers
    optimizer = tf.train.AdamOptimizer(learn_rate)
    A_solver = optimizer.minimize(A_loss)
    D_solver = optimizer.minimize(D_loss, var_list=dis_var)
    G_solver = optimizer.minimize(G_loss, var_list=en_var)

    # reshape images to display them
    S_input_l = tf.reshape(x_l, [-1, 28, 28, 1])
    S_rec_l = tf.reshape(x_hat_l, [-1, 28, 28, 1])
    S_input_u = tf.reshape(x_u, [-1, 28, 28, 1])
    S_rec_u = tf.reshape(x_hat_u, [-1, 28, 28, 1])

    # tensorboard visualization
    tf.summary.scalar(name='Auto-encoder Loss', tensor=A_loss)
    tf.summary.scalar(name='Discriminator Loss', tensor=D_loss)
    tf.summary.scalar(name='Generator Loss', tensor=G_loss)
    tf.summary.image(name='Input Labeled', tensor=S_input_l, max_outputs=10)
    tf.summary.image(name='R-Input Labeled', tensor=S_rec_l, max_outputs=10)
    tf.summary.image(name='Input UnLabeled', tensor=S_input_u, max_outputs=10)
    tf.summary.image(name='R-Input UnLabeled', tensor=S_rec_u, max_outputs=10)
    summary_op = tf.summary.merge_all()

    # data
    mnist_l, mnist_u = mtl.create_semi_supervised_data(data_path)
    n_batches = int(mnist_u.num_examples/batch_size)

    # train the model
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir=summary_path, graph=sess.graph)
        sess.run(tf.global_variables_initializer())

        # training process
        for epochs in range(n_epochs):
            ave_loss_list = [0, 0, 0, 0, 0, 0, 0]
            for n in range(n_batches+1):
                batch_x, batch_y = mnist_u.next_batch(batch_size)
                batch_x_u, _ = mnist_l.next_batch(batch_size)
                s_z_real_l = spl.supervised_gaussian_mixture(batch_size, batch_y)
                s_z_real_u = spl.gaussian_mixture(batch_size)
                ##
                sess.run(A_solver, feed_dict={x_l:batch_x, x_u:batch_x_u})
                sess.run(D_solver, feed_dict={x_l:batch_x, x_u:batch_x_u,
                                              y_real:batch_y, y_fake:batch_y,
                                              z_real_l:s_z_real_l, z_real_u:s_z_real_u})
                sess.run(G_solver, feed_dict={x_l:batch_x, x_u:batch_x_u,
                                              y_fake:batch_y})

                loss_list = sess.run([A_loss, D_loss,
                                      D_loss_fake_l, D_loss_fake_u, D_loss_real_l, D_loss_real_l,
                                      G_loss],
                                     feed_dict={x_l:batch_x, x_u:batch_x_u,
                                                y_real:batch_y, y_fake:batch_y,
                                                z_real_l:s_z_real_l, z_real_u:s_z_real_u})
                ave_loss(ave_loss_list, loss_list, n_batches)
            # summary
            summary = sess.run(summary_op, feed_dict={x_l:batch_x, x_u:batch_x_u,
                                                      y_real:batch_y, y_fake:batch_y,
                                                      z_real_l:s_z_real_l, z_real_u:s_z_real_u})
            writer.add_summary(summary, global_step=epochs)

            liner = "Epoch {:3d}/{:d}, loss_en_de {:9f}, loss_dis {:9f}" \
                    "(fake_l {:9f}, fake_u {:9f}, real_l {:9f}, real_u{:9f}), " \
                    "loss_gen {:9f}" \
                .format(epochs, n_epochs, ave_loss_list[0], ave_loss_list[1],
                        ave_loss_list[2], ave_loss_list[3], ave_loss_list[4],  ave_loss_list[5],
                        ave_loss_list[6])
            print(liner)

            with open(summary_path + '/log.txt', 'a') as log:
                log.write(liner)

        # save model
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver = tf.train.Saver(var_list=vars)
        saver.save(sess, save_path=save_path)

if __name__ == '__main__':
    train()
