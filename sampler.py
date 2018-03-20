
import numpy as np
import random
from math import sin, cos, sqrt
import matplotlib.pyplot as plt


def onehot_categorical(batch_size, n_labels):
    y = np.zeros((batch_size, n_labels), dtype=np.float32)
    indices = np.random.randint(0, n_labels, batch_size)
    for b in range(batch_size):
        y[b, indices[b]] = 1
    return y

def uniform(batch_size, n_dim, minv=-1, maxv=1):
    return np.random.uniform(minv, maxv, (batch_size, n_dim)).astype(np.float32)

def gaussian(batch_size, n_dim, mean=0, var=1):
    return np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)

def gaussian_mixture(batch_size, n_labels=10, n_dim=2):
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    def sample(x, y, label, n_labels):
        shift = 4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 1.5
    y_var = 0.3
    div = int(n_dim/2)
    x = np.random.normal(0, x_var, (batch_size, div))
    y = np.random.normal(0, y_var, (batch_size, div))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(div):
            z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], random.randint(0, n_labels - 1), n_labels)
    return z

def supervised_gaussian_mixture(batch_size, labels, n_labels=10, n_dim=2):
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    # one-hot --> number code
    label_indices = np.argmax(labels, axis=1)

    def sample(x, y, label, n_labels):
        shift = 6#4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 1.5
    y_var = 0.1#0.3
    dim = int(n_dim/2)
    x = np.random.normal(0, x_var, [batch_size,dim])
    y = np.random.normal(0, y_var, [batch_size,dim])
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(dim):
            z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
    return z

def supervised_gaussian_mixture_2d(batch_size, labels, n_labels=10):
    def sample(x, y, label, n_labels):
        shift = 4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    # one-hot --> number code
    label_indices = np.argmax(labels, axis=1)

    x_var = 1.5
    y_var = 0.3
    x = np.random.normal(0, x_var, [batch_size,1])
    y = np.random.normal(0, y_var, [batch_size,1])
    z = np.empty((batch_size, 2), dtype=np.float32)
    for batch in range(batch_size):
        z[batch,:] = sample(x[batch, 0], y[batch, 0], label_indices[batch], n_labels)
    return z

def supervised_gaussian_mixture_3d(batch_size, labels):
    def sample(x, y, z, label, shift=4):
        n1 = int((label-1) % 10)    # 0~9
        n2 = int((label-1) / 10)    # 0~9
        r1 = 2.0 * np.pi / float(12) * float(n1+1+int(n1>4))
        r2 = 2.0 * np.pi / float(10) * float(n2)
        # roll around z
        new_x = x * cos(r1) - y * sin(r1)
        new_y = x * sin(r1) + y * cos(r1)
        new_z = z
        new_x += shift * cos(r1)
        new_y += shift * sin(r1)
        # roll around x
        new_new_x = new_x
        new_new_y = new_y * cos(r2) - new_z * sin(r2)
        new_new_z = new_y * sin(r2) + new_z * cos(r2)
        return np.array([new_new_x, new_new_y, new_new_z]).reshape((3,))

    x_var = 1
    y_var = 0.1
    z_var = 0.1
    x = np.random.normal(0, x_var, (batch_size, 1))
    y = np.random.normal(0, y_var, (batch_size, 1))
    z = np.random.normal(0, z_var, (batch_size, 1))
    point = np.empty((batch_size, 3), dtype=np.float32)
    for batch in range(batch_size):
        point[batch, :] = sample(x[batch, 0], y[batch, 0], z[batch, 0], labels[batch])
    return point

def swiss_roll(batch_size, n_dim=2, n_labels=10):
    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batch_size, n_dim), dtype=np.float32)
    dim = int(n_dim/2)
    for batch in range(batch_size):
        for zi in range(dim):
            z[batch, zi*2:zi*2+2] = sample(random.randint(0, n_labels - 1), n_labels)
    return z

def supervised_swiss_roll(batch_size, labels, n_dim=2, n_labels=10):
    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    # one-hot --> number code
    label_indices = np.argmax(labels, axis=1)

    z = np.zeros((batch_size, n_dim), dtype=np.float32)
    dim = int(n_dim/2)
    for batch in range(batch_size):
        for zi in range(dim):
            z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
    return z


if __name__ == '__main__':
    z = gaussian_mixture(50000, 10, 2)

    for n in range(10):
        x = z[:, 0]
        y = z[:, 1]
        plt.scatter(x, y)
    # plt.axis('off')
    plt.show()