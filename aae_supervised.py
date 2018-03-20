
import tensorflow as tf
import sampler as spl
import datafactory as df
import tools as tl

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator
from datetime import datetime


def train():
    data_set = 'mnist'
    prior = 'mix-gaussian'
    x_dim = 784
    y_dim = 10
    z_dim = 2
    batch_size = 256
    num_epochs = 500*100
    dis_epochs = 500
    save_epochs = int(num_epochs/10)
    learn_rate = 0.0001

    data_path = 'E:/dataset/{}'.format(data_set)
    save_path = tl.make_path('save/supervised/{}/{}'.format(data_set, prior), verbose=False)

    x = tf.placeholder(dtype=tf.float32, shape=[None, x_dim], name='x')
    z_real = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='z_real')
    y_fake = tf.placeholder(dtype=tf.float32, shape=[None, y_dim], name='y_fake')
    y_real = tf.placeholder(dtype=tf.float32, shape=[None, y_dim], name='y_real')

    encoder = Encoder(in_dim=x_dim, h_dim=1024, out_dim=z_dim, type='v2')
    decoder = Decoder(in_dim=z_dim, h_dim=1024, out_dim=x_dim, type='v2')
    discriminator = Discriminator(in_dim=z_dim+y_dim, h_dim=1024, out_dim=2, type='v2')

    z_fake = encoder.feed_forward(x)
    x_hat = decoder.feed_forward(z_fake)

    z_fake_concat = tf.concat([z_fake, y_fake], axis=1)
    d_fake = discriminator.feed_forward(z_fake_concat)
    z_real_concat = tf.concat([z_real, y_real], axis=1)
    d_real = discriminator.feed_forward(z_real_concat)

    # auto-encoder loss
    A_loss = tf.reduce_mean(tf.square(x - x_hat))
    # construct labels for discriminator
    tensor_one = tf.ones(shape=[batch_size, 1])
    tensor_zero = tf.zeros(shape=[batch_size, 1])
    labels_real = tf.concat([tensor_one, tensor_zero], axis=1)
    labels_fake = tf.concat([tensor_zero, tensor_one], axis=1)
    # discriminator loss
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_real, logits=d_real))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_fake, logits=d_fake))
    with tf.control_dependencies([D_loss_fake, D_loss_real]):
        D_loss = D_loss_fake + D_loss_real

    # generator loss
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_real, logits=d_fake))

    # variables
    all_variables = tf.trainable_variables()
    en_var = [var for var in all_variables if 'Encoder' in var.name]
    de_var = [var for var in all_variables if 'Decoder' in var.name]
    dc_var = [var for var in all_variables if 'Discriminator' in var.name]

    # optimizers
    A_solver = tf.train.RMSPropOptimizer(learn_rate).minimize(A_loss)
    D_solver = tf.train.RMSPropOptimizer(learn_rate).minimize(D_loss, var_list=dc_var)
    G_solver = tf.train.RMSPropOptimizer(learn_rate).minimize(G_loss, var_list=en_var)

    # tensorboard visualization
    tf.summary.histogram('z_fake', z_fake)
    tf.summary.histogram('z_real', z_real)
    tf.summary.scalar(name='auto-encoder loss', tensor=A_loss)
    tf.summary.scalar(name='discriminator Loss', tensor=D_loss)
    tf.summary.scalar(name='generator loss', tensor=G_loss)
    summary = tf.summary.merge_all()

    # data
    data = df.create_supervised_data(data_path, data_set, reshape=True)
    n_batches = int(data.num_examples/batch_size)
    z_sample = tl.get_meshgrid(z_range=10)
    # tl.make_version_info(save_path)


    file = open('{}/train.txt'.format(save_path), 'w')
    sess = tf.Session()

    # writer = tf.summary.FileWriter(logdir=summary_path, graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    ave_loss_list = [0, 0, 0, 0]
    start_time = cur_time = datetime.now()

    # training process
    for epochs in range(1,num_epochs+1):
        batch_x, batch_y = data.next_batch(batch_size)
        s_z_real = spl.supervised_gaussian_mixture(batch_size, batch_y, y_dim, z_dim)
        # solve
        sess.run(A_solver, feed_dict={x: batch_x})
        sess.run(D_solver, feed_dict={x: batch_x, y_fake: batch_y, z_real: s_z_real, y_real: batch_y})
        sess.run(G_solver, feed_dict={x: batch_x, y_fake: batch_y})

        loss_list = sess.run([A_loss, D_loss_fake, D_loss_real, G_loss],
                             feed_dict={x: batch_x, y_fake: batch_y,
                                        z_real: s_z_real, y_real: batch_y})
        tl.ave_loss(ave_loss_list, loss_list, dis_epochs)
        # summary
        # summary = sess.run(summary_op, feed_dict={x:batch_x, z_real:z_real_s})
        # writer.add_summary(summary, global_step=epochs)

        if epochs % dis_epochs == 0:
            time_use = (datetime.now() - cur_time).seconds
            liner = "Epoch {:3d}/{:d}, loss_en_de {:9f}, loss_dis_faker {:9f}, loss_dis_real {:9f}, loss_encoder {:9f} time_use {:f}" \
                .format(epochs, num_epochs, ave_loss_list[0], ave_loss_list[1], ave_loss_list[2], ave_loss_list[3], time_use)
            # step_summary = sess.run(summary, feed_dict={x:batch_x, y_fake:batch_y, z_real:s_z_real, y_real:batch_y})
            # writer.add_summary(step_summary, global_step=epochs)
            print(liner), file.writelines(liner+'\n')
            ave_loss_list = [0, 0, 0, 0] # reset to 0
            cur_time = datetime.now()

        if epochs % save_epochs == 0:
            number = int(epochs/save_epochs)
            images = sess.run(x_hat, feed_dict={z_fake:z_sample})
            tl.save_grid_images(images, '{}/{}.jpg'.format(save_path, number))


    # save model
    end_time = datetime.now()
    liner = 'time use: {}'.format((end_time - start_time).seconds)
    print(liner), file.writelines(liner)
    file.close()
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(var_list=vars)
    saver.save(sess, save_path='{}\model'.format(save_path))

if __name__ == '__main__':
    train()
