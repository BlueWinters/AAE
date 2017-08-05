
import numpy as np
from math import sin, cos, sqrt

import matplotlib.pyplot as plt

class Sampler(object):
    def __init__(self, type='gaussian', dim=2):
        self.type = type
        self.dim = dim

    def __call__(self, batch_size):
        if self.type == 'gaussian':
            return self._sample_2d_Gaussian(batch_size)
        elif self.type == 'swiss_rolls':
            return self._sample_2d_swiss_rolls(batch_size)

    def _sample_2d_Gaussian(self, batch_size, e_mean=0.0, e_std=1.0):
        return np.random.normal(loc=e_mean, scale=e_std, size=[batch_size, 2])

    def _sample_10Gaussian(self, batch_size):
        pass

    def _sample_2d_swiss_rolls(self, batch_size, length=4):
        z = np.zeros((batch_size, 2), dtype=np.float32)
        for batch in range(batch_size):
            uni = np.random.uniform(0.0, 1.0)
            r = sqrt(uni) * length
            rad = np.pi * r
            z[batch, 0] = r * cos(rad)
            z[batch, 1] = r * sin(rad)
        return z


if __name__ == '__main__':
    sampler = Sampler()
    point = sampler._sample_2d_swiss_rolls(100)
    x = point[:,0]
    y = point[:,1]
    plt.scatter(x, y, edgecolors='face')
    plt.show()