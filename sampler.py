
import numpy as np


class Sampler(object):
    def __init__(self, type='Gaussian', dim=2):
        self.type = type
        self.dim = dim

    def __call__(self, batch_size, e_mean=0.0, e_std=1.0):
        return np.random.normal(loc=e_mean, scale=e_std, size=[batch_size, self.dim])

    def _sample_2Gaussian(self, batch_size):
        pass

    def _sample_10Gaussian(self, batch_size):
        pass

    def _sample_rolls(self, batch_size):
        pass