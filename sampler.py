
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

def gaussian_mixture(batch_size, n_dim=2, n_labels=10):
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
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
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    dim = int(n_dim/2)
    x = np.random.normal(0, x_var, [batch_size,dim])
    y = np.random.normal(0, y_var, [batch_size,dim])
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(dim):
            z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
    return z

def swiss_roll(batch_size, n_dim, n_labels):
    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(n_dim / 2):
            z[batch, zi*2:zi*2+2] = sample(random.randint(0, n_labels - 1), n_labels)
    return z

def supervised_swiss_roll(batch_size, n_dim, label_indices, n_labels):
    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(n_dim / 2):
            z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
    return z




if __name__ == '__main__':
    pass