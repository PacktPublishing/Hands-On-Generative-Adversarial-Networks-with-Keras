"""An implementation of the improved WGAN described in https://arxiv.org/abs/1704.00028

The improved WGAN has a term in the loss function which penalizes the network if its gradient
norm moves away from 1. This is included because the Earth Mover (EM) distance used in WGANs is only easy
to calculate for 1-Lipschitz functions (i.e. functions where the gradient norm has a constant upper bound of 1).

The original WGAN paper enforced this by clipping weights to very small values [-0.01, 0.01]. However, this
drastically reduced network capacity. Penalizing the gradient norm is more natural, but this requires
second-order gradients. These are not supported for some tensorflow ops (particularly MaxPool and AveragePool)
in the current release (1.0.x), but they are supported in the current nightly builds (1.1.0-rc1 and higher).

To avoid this, this model uses strided convolutions instead of Average/Maxpooling for downsampling. If you wish to use
pooling operations in your discriminator, please ensure you update Tensorflow to 1.1.0-rc1 or higher. I haven't
tested this with Theano at all.

The model saves images using pillow. If you don't have pillow, either install it or remove the calls to generate_images.
"""
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

from resnet import (build_cifar10_resnet_discriminator,
                    build_cifar10_resnet_generator)
from dcgan import (build_cifar10_dcgan_discriminator,
                   build_cifar10_dcgan_generator)

N_CRITIC_ITERS = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.

def loss_wasserstein(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.

    The Wasserstein loss function is very simple to calculate. In a standard
    GAN, the discriminator has a sigmoid output, representing the probability
    that samples are real or generated. In Wasserstein GANs, however, the output
    is linear with no activation function! Instead of being constrained to [0,
    1], the discriminator wants to make the distance between its output for real
    and generated samples as large as possible.

    The most natural way to achieve this is to label generated samples -1 and
    real samples 1, instead of the 0 and 1 used in normal GANs, so that
    multiplying the outputs by the labels will give you the loss immediately.

    Note that the nature of this loss means that it can be (and frequently will
                                                            be) less than 0."""
    return K.mean(y_true) * y_pred


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


def train(ndf=64, ngf=64, z_dim=100, n_residual_blocks_discriminator=3,
          n_residual_blocks_generator=3, lr_d=1e-5, lr_g=1e-5,
          n_iterations=int(1e6), batch_size=128, n_checkpoint_images=36,
          out_dir='rgan_resnet'):

    X_train, _ = get_data()
    image_shape = X_train[0].shape
    print("image shape {}, min val {}, max val {}".format(
        image_shape, X_train[0].min(), X_train[0].max()))

    # plot real images for reference
    plot_images(X_train[:n_checkpoint_images], "{}/real_images.png".format(out_dir))

    # build models
    #input_shape_discriminator = image_shape
    #input_shape_generator = (z_dim, )
    n_channels = image_shape[-1]

    """
    D = build_cifar10_resnet_discriminator(
        input_shape_discriminator, ndf, n_residual_blocks_discriminator)
    G = build_cifar10_resnet_generator(
        input_shape_generator, ngf, n_residual_blocks_generator, n_channels)
    """

    D = build_cifar10_dcgan_discriminator(ndf, image_shape)
    G = build_cifar10_dcgan_generator(ngf, z_dim, n_channels)

    #img_samples = Input(shape=X_train.shape[1:])

    # define D graph and optimizer
    #D = Model(inputs=img_samples, outputs=D(img_samples))
    # D.compile(optimizer=Adam(lr_d, beta_1=0.5, beta_2=0.9), loss=loss_wasserstein)
    D.compile(optimizer=RMSprop(lr_d), loss=loss_wasserstein)

    # define D(G(z)) graph and optimizer
    D.trainable = False
    z = Input(shape=(z_dim, ))
    D_of_G = Model(inputs=z, outputs=D(G(z)))

    # define Generator's Optimizer
    # D_of_G.compile(Adam(lr=lr_g, beta_1=0.5, beta_2=0.9), loss=loss_wasserstein)
    D_of_G.compile(RMSprop(lr=lr_g), loss=loss_wasserstein)

    ones = np.ones((batch_size, 1), dtype=np.float32)
    minus_ones = -ones

    losses_d, losses_g = [], []

    # fix a z vector for training evaluation
    z_fixed = np.random.normal(0, 1, size=(n_checkpoint_images, z_dim))

    gen_iters = 0
    n_batches = int(len(X_train) / batch_size)
    cur_batch = 0
    epoch = 0
    for i in range(n_iterations):
        if gen_iters < 25 or gen_iters % 500 == 0:
            n_critic_iters = 100
        else:
            n_critic_iters = N_CRITIC_ITERS

        D.trainable = True
        G.trainable = False
        for j in range(n_critic_iters):
            for l in D.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -0.01, 0.01) for w in weights]
                l.set_weights(weights)
            if cur_batch == 0:
                ids = np.arange(X_train.shape[0])
                np.random.shuffle(ids)
                epoch += 1
            z = np.random.normal(0, 1, size=(batch_size, z_dim))
            cur_ids = ids[cur_batch * batch_size:(cur_batch+1)*batch_size]
            real_batch = X_train[cur_ids]
            fake_batch = G.predict(z)
            loss_real = D.train_on_batch(real_batch, ones)
            loss_fake = D.train_on_batch(fake_batch, minus_ones)
            losses_d.append([loss_real + loss_fake, loss_real, loss_fake])
            cur_batch = (cur_batch + 1) % n_batches

        D.trainable = False
        G.trainable = True
        z = np.random.normal(0, 1, size=(batch_size, z_dim))
        loss_g = D_of_G.train_on_batch(z, ones)
        losses_g.append(loss_g)
        gen_iters += 1

        if cur_batch == 0:
            G.trainable = False
            print("epoch={}, loss_d={:.5f}, loss_g={:.5f}".format(
                epoch, np.mean(np.array(losses_d)[:, 0]), loss_g))
            fake_images = G.predict(z_fixed)
            print("\tPlotting images and losses")
            plot_images(fake_images, "{}/fake_images_epoch{}.png".format(out_dir, epoch))
            plot_losses(losses_d, losses_g, "{}/losses.png".format(out_dir))
            epoch += 1


train()
