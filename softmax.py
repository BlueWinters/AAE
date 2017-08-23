
import tensorflow as tf
import datafactory as df

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator


def get_config_path():
    data_path = 'mnist'
    summary_path = 'softmax/summary'
    save_path = 'softmax/ckpt/model'
    model_path = 'semi-supervised/ckpt/model'
    return data_path, summary_path, save_path, model_path

def train():
    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator(in_dim=13)

    x_dim = 784
    z_dim = 2
    y_dim = 10
    n_epochs = 100
    learn_rate = 0.001
    batch_size = 100
    tiny = 1e-6

    x = tf.placeholder(dtype=tf.float32, shape=[None,x_dim], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None,y_dim], name='y')

    with tf.variable_scope('Softmax'):
        W = tf.get_variable(name='W', shape=[z_dim, y_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable(name='b', shape=[y_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0))

    all_variables = tf.trainable_variables()
    vars = [var for var in all_variables if 'Softmax' in var.name]
    en_var = [var for var in all_variables if 'Encoder' in var.name]
    de_var = [var for var in all_variables if 'Decoder' in var.name]
    dc_var = [var for var in all_variables if 'Discriminator' in var.name]
    model_var = en_var + de_var + dc_var

    # get features
    z = encoder.feed_forward(x)
    pred = tf.nn.softmax(tf.matmul(z, W) + b)
    loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred+tiny), reduction_indices=1))
    true_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(true_pred, tf.float32))

    optimizer = tf.train.AdamOptimizer(learn_rate)
    solver = optimizer.minimize(loss, var_list=vars)

    tf.summary.scalar(name='softmax loss', tensor=loss)
    summary_op = tf.summary.merge_all()

    data_path, summary_path, save_path, model_path = get_config_path()
    train, validation = df.create_supervised_data(data_path, validation=True)
    n_batches = int(train.num_examples/batch_size)

    # train the model
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir=summary_path, graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(model_var)
        saver.restore(sess, save_path=model_path)
        ave_loss = 0

        # training process
        for epochs in range(n_epochs):
            ave_loss = 0
            for n in range(1, n_batches+1):
                batch_x, batch_y = train.next_batch(batch_size)
                epochs_loss, _ = sess.run([loss,solver], feed_dict={x:batch_x, y:batch_y})
                ave_loss += epochs_loss/n_batches
            summary = sess.run(summary_op, feed_dict={x:batch_x, y:batch_y})
            val_acc = sess.run(accuracy, feed_dict={x:validation.images, y:validation.labels})
            train_acc = sess.run(accuracy, feed_dict={x:train.images, y:train.labels})
            writer.add_summary(summary, global_step=epochs)
            print("Epoch {:3d}/{:d}, loss {:9f}, validation {:9f}, train {:9f}"
                  .format(epochs, n_epochs, ave_loss, val_acc, train_acc))

        saver = tf.train.Saver(var_list=vars)
        saver.save(sess, save_path=save_path)



if __name__ == '__main__':
    train()