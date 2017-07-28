
import numpy as np


class Sampler(object):
    def __init__(self, type='Gaussian', dim=2):
        self.type = type
        self.dim = dim

    def __call__(self, batch_size, e_mean=0.0, e_std=1.0):
        mean = np.zeros([batch_size, self.dim])
        std = np.ones([batch_size, self.dim])
        epsilon = np.random.normal(loc=e_mean, scale=e_std, size=[batch_size, self.dim])
        return mean + epsilon * std

    def _sample_2Gaussian(self, batch_size):
        pass

    def _sample_10Gaussian(self, batch_size):
        pass

    def _sample_rolls(self, batch_size):
        pass