
import numpy as np
import math
import matplotlib.pyplot as plt

class Sampler(object):
    def __init__(self, type='Gaussian', dim=2):
        self.type = type
        self.dim = dim

    def __call__(self, batch_size, e_mean=0.0, e_std=10.0):
        mean = np.zeros([batch_size, self.dim])
        std = np.ones([batch_size, self.dim])
        epsilon = np.random.normal(loc=e_mean, scale=e_std, size=[batch_size, self.dim])
        return mean + epsilon * std

if __name__ == '__main__':
    s = Sampler()
    p = s(1000)
    x = p[:,0]
    y = p[:,1]
    plt.scatter(x, y)
    plt.show()