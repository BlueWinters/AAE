
import tensorflow as tf
import datafactory as df


def get_config_path():
    data_path = 'mnist'
    summary_path = 'k-softmax/src'
    save_path = 'k-softmax/src'
    return data_path, summary_path, save_path

def train():
    x_dim = 784
    y_dim = 10
    top_k = 0.15
    n_epochs = 100
    learn_rate = 0.001
    batch_size = 100
    tiny = 1e-6

    x = tf.placeholder(dtype=tf.float32, shape=[batch_size,x_dim], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[batch_size,y_dim], name='y')
    vx = tf.placeholder(dtype=tf.float32, shape=[None,x_dim], name='vx')
    vy = tf.placeholder(dtype=tf.float32, shape=[None,y_dim], name='vy')

    W = tf.get_variable(name='W', shape=[x_dim, y_dim], dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.1))
    b = tf.get_variable(name='b', shape=[y_dim], dtype=tf.float32,
                        initializer=tf.constant_initializer(0))

    # loss
    logits = tf.nn.softmax(tf.matmul(x, W) + b)
    n, c = logits.shape[0], logits.shape[1]
    tran_logits = tf.transpose(logits) # c, n
    k = tf.cast(top_k * tf.cast(n, tf.float32), tf.int32)
    keep_value, _ = tf.nn.top_k(tran_logits, k)
    shape = [c, n]
    drop = tf.where(tran_logits < keep_value[:,k-1:k], tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
    winner = logits * tf.transpose(drop)
    loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(winner+tiny), reduction_indices=1))

    # accuracy
    prediction = tf.nn.softmax(tf.matmul(vx, W) + b)
    true_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(vy,1))
    accuracy = tf.reduce_mean(tf.cast(true_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(learn_rate)
    solver = optimizer.minimize(loss)

    tf.summary.scalar(name='softmax loss', tensor=loss)
    summary_op = tf.summary.merge_all()

    data_path, summary_path, _ = get_config_path()
    train, validation = df.create_supervised_data(data_path, validation=True)
    n_batches = int(train.num_examples/batch_size)

    # train the model
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir=summary_path, graph=sess.graph)
        sess.run(tf.global_variables_initializer())

        # training process
        for epochs in range(n_epochs):
            average_loss = 0
            for n in range(1, n_batches+1):
                batch_x, batch_y = train.next_batch(batch_size)
                epochs_loss, _ = sess.run([loss,solver], feed_dict={x:batch_x, y:batch_y})
                average_loss += epochs_loss/n_batches
            summary = sess.run(summary_op, feed_dict={x:batch_x, y:batch_y})
            valid_acc = sess.run(accuracy, feed_dict={vx:validation.images, vy:validation.labels})
            train_acc = sess.run(accuracy, feed_dict={vx:train.images, vy:train.labels})
            writer.add_summary(summary, global_step=epochs)
            print("Epoch {:3d}/{:d}, loss {:9f}, validation {:9f}, train {:9f}"
                  .format(epochs, n_epochs, average_loss, valid_acc, train_acc))


if __name__ == '__main__':
    train()

