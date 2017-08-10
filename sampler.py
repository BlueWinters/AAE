
import numpy as np
from math import sin, cos, sqrt

import matplotlib.pyplot as plt

class Sampler(object):
    def __init__(self, type='gaussian', dim=2):
        self.type = type
        self.dim = dim

    def __call__(self, batch_size):
        if self.type == 'gaussian':
            return self._sample_2d_gaussian(batch_size)
        elif self.type == 'swiss_rolls':
            return self._sample_2d_swiss_rolls(batch_size)
        elif self.type == '10gaussian':
            return self._sample_mix_gaussian(batch_size)

    def _sample_2d_gaussian(self, batch_size, e_mean=0.0, e_std=1.0):
        return np.random.normal(loc=e_mean, scale=e_std, size=[batch_size, 2])

    def _sample_mix_gaussian(self, batch_size, n_labels=10, x_var=0.5, y_var=0.1):
        x = np.random.normal(0, x_var, [batch_size, 1])
        y = np.random.normal(0, y_var, [batch_size, 1])
        z = np.empty([batch_size, self.dim], dtype=np.float32)
        for batch in range(batch_size):
            z[batch, 0:2] = self._sample_mix_gaussian_one(
                x[batch], y[batch], np.random.randint(0, n_labels), n_labels)
        return z

    def _sample_mix_gaussian_one(self, x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    def _sample_2d_swiss_rolls(self, batch_size, length=4):
        z = np.zeros((batch_size, 2), dtype=np.float32)
        for batch in range(batch_size):
            uni = np.random.uniform(0.0, 1.0)
            r = sqrt(uni) * length
            rad = np.pi * r
            z[batch, 0] = r * cos(rad)
            z[batch, 1] = r * sin(rad)
        return z

    def sample_grid(self):
        x_points = np.reshape(np.arange(-10, 10, 1).astype(np.float32), [-1,1])
        y_points = np.reshape(np.arange(-10, 10, 1).astype(np.float32), [-1,1])
        point = np.concatenate((x_points,y_points), axis=1)
        return np.reshape(point, [-1,2])


if __name__ == '__main__':
    sampler = Sampler('10gaussian')
    point = sampler._sample_mix_gaussian(100)
    x = point[:,0]
    y = point[:,1]
    plt.scatter(x, y, edgecolors='face')
    plt.show()