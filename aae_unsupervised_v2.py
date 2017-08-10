import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data



mb_size = 32
z_dim = 2
X_dim = 28*28
y_dim = 10
h_dim = 128

batch_size = 100
n_epochs = 100
lr = 1e-3


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig

def get_config_path():
    data_path = '/mnist'
    summary_path = 'unsupervised/summary'
    save_path = 'unsupervised/ckpt/model'
    return data_path, summary_path, save_path

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


""" Q(z|X) """
X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2 = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2 = tf.Variable(tf.zeros(shape=[z_dim]))

theta_Q = [Q_W1, Q_W2, Q_b1, Q_b2]


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z = tf.matmul(h, Q_W2) + Q_b2
    return z


""" P(X|z) """
P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_P = [P_W1, P_W2, P_b1, P_b2]


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits


""" D(z) """
D_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


def D(z):
    h = tf.nn.relu(tf.matmul(z, D_W1) + D_b1)
    logits = tf.matmul(h, D_W2) + D_b2
    prob = tf.nn.sigmoid(logits)
    return prob


""" Training """
z_sample = Q(X)
_, logits = P(z_sample)

# Sample from random z
X_samples, _ = P(z)

# E[log P(X|z)]
R_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X))
# Adversarial loss to approx. Q(z|X)
D_real = D(z)
D_fake = D(z_sample)

D_loss_real = -tf.reduce_mean(tf.log(D_real))
D_loss_fake = -tf.reduce_mean(tf.log(1. - D_fake))
D_loss = D_loss_real + D_loss_fake

G_loss = -tf.reduce_mean(tf.log(D_fake))

AE_solver = tf.train.AdamOptimizer().minimize(R_loss, var_list=theta_P + theta_Q)
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_Q)


tf.summary.scalar(name='Auto-encoder Loss', tensor=R_loss)
tf.summary.scalar(name='Discriminator Loss', tensor=D_loss)
tf.summary.scalar(name='Encoder Loss', tensor=G_loss)
tf.summary.image(name='Input Images', tensor=tf.reshape(X,[-1,28,28,1]), max_outputs=10)
tf.summary.image(name='Reconstructed Images', tensor=tf.reshape(logits,[-1,28,28,1]), max_outputs=10)
summary_op = tf.summary.merge_all()

data_path, summary_path, save_path = get_config_path()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(logdir=summary_path, graph=sess.graph)

if not os.path.exists(save_path):
    os.makedirs(save_path)

mnist = input_data.read_data_sets(data_path, one_hot=True)
n_batches = int(mnist.train.num_examples/batch_size)

for n in range(n_epochs):
    for n in range(1, n_batches+1):
        batch_x, batch_y = mnist.train.next_batch(mb_size)
        batch_z = np.random.randn(mb_size, z_dim)

        _, R_loss_curr = sess.run([AE_solver, R_loss], feed_dict={X:batch_x})
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X:batch_x, z:batch_z})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X:batch_x})

    loss_list = sess.run([R_loss, D_loss_fake, D_loss_real, G_loss], feed_dict={X:batch_x, z:batch_z})
    print('Epoch: {:}/{:}, R_loss: {:9f}, '
          'D_loss_fake: {:9f}, D_loss_fake: {:9f}, G_loss: {:9f}'
          .format(n, n_epochs, loss_list[0], loss_list[1], loss_list[2], loss_list[3]))

    if n % 10 == 0:
        samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})
        fig = plot(samples)
        plt.savefig(save_path + '/{}.png'.format(str(n).zfill(3)), bbox_inches='tight')
        plt.close(fig)


# manifold
x_points = np.arange(-10, 10, 0.5).astype(np.float32)
y_points = np.arange(-10, 10, 0.5).astype(np.float32)
nx, ny = len(x_points), len(y_points)

plt.subplot()
gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)
for i, g in enumerate(gs):
    sz = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
    sz = np.reshape(sz, (1, 2))
    x = sess.run(X_samples, feed_dict={z:sz})
    ax = plt.subplot(g)
    img = np.array(x.tolist()).reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
plt.show()