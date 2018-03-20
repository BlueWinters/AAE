
import os as os
import shutil as sh
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

from datetime import datetime


def make_path(root_dir, verbose=True):
    if verbose == True:
        time_str = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        save_path = '{}/{}'.format(root_dir, time_str)
    else:
        save_path = root_dir

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    return save_path

def ave_loss(ave_lost_list, step_loss_list, div):
    assert len(ave_lost_list) == len(step_loss_list)
    for n in range(len(ave_lost_list)):
        ave_lost_list[n] += step_loss_list[n] / div

def make_version_info(save_path):
    sh.copy('encoder.py', '{}/encoder.py'.formdfsat(save_path))
    sh.copy('decoder.py', '{}/decoder.py'.format(save_path))
    sh.copy('discriminator.py', '{}/discriminator.py'.format(save_path))
    # sh.copy('aae_supervised.py', '{}/aae_supervised.py'.format(save_path))

def get_meshgrid(z_range, nx=20, ny=20):
    sample = np.rollaxis(np.mgrid[-z_range:z_range:ny * 1j, -z_range:z_range:nx * 1j], 0, 3)
    return np.reshape(sample, newshape=[nx*nx,2])

def save_grid_images(images, save_path_and_name, nx=20, ny=20, size=28, chl=1):
    plt.cla()
    if chl == 3:
        stack_images = np.zeros([ny*size, nx*size, chl])
        for j in range(ny):
            for i in range(nx):
                stack_images[j*size:(j+1)*size, i*size:(i+1)*size, :] = np.reshape(images[j*ny+i,:], [size,size,chl])
        scipy.misc.imsave(save_path_and_name, stack_images)
        # plt.imshow(stack_images)
        # plt.savefig(save_path_and_name)
    else:
        stack_images = np.zeros([ny*size, nx*size])
        for j in range(ny):
            for i in range(nx):
                stack_images[j*size:(j+1)*size, i*size:(i+1)*size] = np.reshape(images[j*ny+i,:], [size,size])
        scipy.misc.imsave(save_path_and_name, stack_images)


if __name__ == '__main__':
    print(get_meshgrid(10))
    # t = np.mgrid[0:5,0:10]
    # print(t)
    # print(t[0,2])
    # print(t[1,3])

    # print(gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05))