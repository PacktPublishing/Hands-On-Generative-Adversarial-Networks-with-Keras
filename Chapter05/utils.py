import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
from math import ceil
import numpy as np
import argparse
from functools import partial
import os
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras.datasets import mnist
from keras import backend as K

from keras.datasets import cifar10


def get_data():
    # load cifar10 data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # convert train and test data to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # scale train and test data to [-1, 1]
    X_train = (X_train / 255) * 2 - 1
    X_test = (X_train / 255) * 2 - 1

    return X_train, X_test

def plot_images(images, filename):
    # scale images to [0.0, 1.0]
    images = (images + 1) / 2
    h, w, c = images.shape[1:]
    grid_size = ceil(np.sqrt(images.shape[0]))
    images = (images.reshape(grid_size, grid_size, h, w, c)
              .transpose(0, 2, 1, 3, 4)
              .reshape(grid_size*h, grid_size*w, c))
    plt.figure(figsize=(16, 16))
    plt.imsave(filename, images)
    plt.close('all')


def plot_losses(losses_d, losses_g, filename):
    losses_d = np.array(losses_d)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    axes[0].plot(losses_d[:, 0])
    axes[1].plot(losses_d[:, 1])
    axes[2].plot(losses_d[:, 2])
    axes[3].plot(losses_g)
    axes[0].set_title("losses_d")
    axes[1].set_title("losses_d_real")
    axes[2].set_title("losses_d_fake")
    axes[3].set_title("losses_g")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()