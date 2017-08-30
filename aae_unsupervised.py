
import tensorflow as tf
import sampler as spl
import datafactory as df

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator


def get_config_path():
    data_path = 'mnist'
    summary_path = 'unsupervised/mix-gaussian'
    save_path = 'unsupervised/mix-gaussian'
    return data_path, summary_path, save_path

def ave_loss(ave_lost_list, step_loss_list, div):
    assert len(ave_lost_list) == len(step_loss_list)
    for n in range(len(ave_lost_list)):
        ave_lost_list[n] += step_loss_list[n] / div

def train():
    x_dim = 784
    z_dim = 2
    batch_size = 100
    n_epochs = 100
    learn_rate = 0.0001

    data_path, summary_path, save_path = get_config_path()

    x = tf.placeholder(dtype=tf.float32, shape=[None, x_dim], name='x')
    z_real = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='z_real')
    z_grid = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='z_grid')

    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()

    z = encoder.feed_forward(x)
    y = decoder.feed_forward(z)

    d_real = discriminator.feed_forward(z_real)
    d_fake = discriminator.feed_forward(z)

    # auto-encoder loss
    A_loss = tf.reduce_mean(tf.square(x - y))
    # discriminator loss
    tensor_one = tf.ones(shape=[batch_size, 1])
    tensor_zero = tf.zeros(shape=[batch_size, 1])
    labels_real = tf.concat([tensor_one, tensor_zero], axis=1)
    labels_fake = tf.concat([tensor_zero, tensor_one], axis=1)
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_real, logits=d_real))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_fake, logits=d_fake))
    with tf.control_dependencies([D_loss_fake, D_loss_real]):
        D_loss = D_loss_fake + D_loss_real

    # generator loss
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_real, logits=d_fake))

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
    S_input = tf.reshape(x, [-1, 28, 28, 1])
    S_rec = tf.reshape(y, [-1, 28, 28, 1])
    y_grid = decoder.feed_forward(z_grid, False)
    S_space = tf.reshape(y_grid, [-1, 28, 28, 1])

    # tensorboard visualization
    tf.summary.scalar(name='Auto-encoder Loss', tensor=A_loss)
    tf.summary.scalar(name='Discriminator Loss', tensor=D_loss)
    tf.summary.scalar(name='Generator Loss', tensor=G_loss)
    tf.summary.image(name='Input', tensor=S_input, max_outputs=10)
    tf.summary.image(name='Rec-Input', tensor=S_rec, max_outputs=10)
    # tf.summary.image(name='Z-Space', tensor=S_space, max_outputs=10)
    summary_op = tf.summary.merge_all()

    # data
    mnist = df.create_supervised_data(data_path)
    n_batches = int(mnist.num_examples / batch_size)


    # train the model
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir=summary_path, graph=sess.graph)
        sess.run(tf.global_variables_initializer())

        # training process
        for epochs in range(n_epochs):
            ave_loss_list = [0, 0, 0, 0]
            for n in range(1, n_batches + 1):
                z_real_s = spl.gaussian_mixture(batch_size, z_dim)
                batch_x, _ = mnist.next_batch(batch_size)
                sess.run(A_solver, feed_dict={x:batch_x})
                sess.run(D_solver, feed_dict={x:batch_x, z_real:z_real_s})
                sess.run(G_solver, feed_dict={x:batch_x})

                loss_list = sess.run([A_loss, D_loss_fake, D_loss_real, G_loss],
                                     feed_dict={x:batch_x, z_real:z_real_s})
                ave_loss(ave_loss_list, loss_list, n_batches)
            # summary
            summary = sess.run(summary_op, feed_dict={x:batch_x, z_real:z_real_s})
            writer.add_summary(summary, global_step=epochs)

            liner = "Epoch {:3d}/{:d}, loss_en_de {:9f}, " \
                    "loss_dis_faker {:9f}, loss_dis_real {:9f}, loss_encoder {:9f}" \
                .format(epochs, n_epochs, ave_loss_list[0],
                        ave_loss_list[1], ave_loss_list[2], ave_loss_list[3])
            print(liner)

        # save model
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver = tf.train.Saver(var_list=vars)
        saver.save(sess, save_path=save_path+'/model')

if __name__ == '__main__':
    train()
