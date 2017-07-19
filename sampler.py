
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


def plotPoint():
    np.random.seed(2017)
    sampler = Sampler()
    p = sampler(100)
    x1 = p[0:49,0]
    y1 = p[0:49,1]
    x2 = p[50:-1,0]
    y2 = p[50:-1,1]
    plt.scatter(x1, y1, color='blue')
    plt.scatter(x2, y2, color='red')
    plt.show()



if __name__ == '__main__':
    # s = Sampler()
    # p = s(1000)
    # x = p[:,0]
    # y = p[:,1]
    # plt.scatter(x, y)
    # plt.show()

    plotPoint()