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


def rel_disc_loss(y_true, y_pred, disc_r=None, disc_f=None):
    epsilon = 0.000001
    return -(K.mean(K.log(K.sigmoid(disc_r - K.mean(disc_f, axis=0))+epsilon), axis=0)\
           +K.mean(K.log(1-K.sigmoid(disc_f - K.mean(disc_r, axis=0))+epsilon), axis=0))


def rel_gen_loss(y_true, y_pred, disc_r=None, disc_f=None):
    epsilon=0.000001
    return -(K.mean(K.log(K.sigmoid(disc_f - K.mean(disc_r, axis=0))+epsilon), axis=0)\
           +K.mean(K.log(1-K.sigmoid(disc_r - K.mean(disc_f, axis=0))+epsilon), axis=0))


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
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes = axes.flatten()
    axes[0].plot(losses_d)
    axes[1].plot(losses_g)
    axes[0].set_title("losses_d")
    axes[1].set_title("losses_g")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def train(ndf=128, ngf=128, z_dim=100, n_residual_blocks_discriminator=3,
          n_residual_blocks_generator=3, lr_d=1e-4, lr_g=1e-4,
          n_iterations=int(1e6), batch_size=128, n_checkpoint_images=36,
          out_dir='rgan_resnet'):

    X_train, _ = get_data()
    image_shape = X_train[0].shape
    print("image shape {}, min val {}, max val {}".format(
        image_shape, X_train[0].min(), X_train[0].max()))

    # plot real images for reference
    plot_images(X_train[:n_checkpoint_images], "{}/real_images.png".format(out_dir))

    # build models
    input_shape_discriminator = image_shape
    input_shape_generator = (z_dim, )
    n_channels = image_shape[-1]

    D = build_cifar10_resnet_discriminator(
        input_shape_discriminator, ndf, n_residual_blocks_discriminator)
    G = build_cifar10_resnet_generator(
        input_shape_generator, ngf, n_residual_blocks_generator, n_channels)

    #D = build_cifar10_dcgan_discriminator(ndf, image_shape)
    #G = build_cifar10_dcgan_generator(ngf, z_dim, n_channels)

    real_samples = Input(shape=X_train.shape[1:])
    z = Input(shape=(z_dim, ))
    fake_samples = G(z)
    D_real = D(real_samples)
    D_fake = D(fake_samples)

    loss_d_fn = partial(rel_disc_loss, disc_r=D_real, disc_f=D_fake)
    loss_g_fn = partial(rel_gen_loss, disc_r=D_real, disc_f=D_fake)

    # define D graph and optimizer
    G.trainable = False
    D.trainable = True
    D_model = Model(inputs=[real_samples, z], outputs=[D_real, D_fake])
    D_model.compile(optimizer=Adam(lr_d, beta_1=0.5, beta_2=0.9),
                    loss=[loss_d_fn, None])

    # define G graph and optimizer
    G.trainable = True
    D.trainable = False
    G_model = Model(inputs=[real_samples, z], outputs=[D_real, D_fake])

    # define Generator's Optimizer
    G_model.compile(Adam(lr=lr_g, beta_1=0.5, beta_2=0.9),
                    loss=[loss_g_fn, None])

    # vars for keeping track of losses
    losses_d, losses_g = [], []
    dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

    # fix a z vector for training evaluation
    z_fixed = np.random.normal(0, 1, size=(n_checkpoint_images, z_dim))

    gen_iters = 0
    n_batches = int(len(X_train) / batch_size)
    cur_batch = 0
    epoch = 0
    for i in range(n_iterations):
        D.trainable = True
        G.trainable = False
        if cur_batch == 0:
            ids = np.arange(X_train.shape[0])
            np.random.shuffle(ids)
            epoch += 1
        z = np.random.normal(0, 1, size=(batch_size, z_dim))
        cur_ids = ids[cur_batch * batch_size:(cur_batch+1)*batch_size]
        real_batch = X_train[cur_ids]
        loss_d = D_model.train_on_batch([real_batch, z], dummy_y)[0]
        losses_d.append([loss_d])
        cur_batch = (cur_batch + 1) % n_batches

        for _ in range(3):
            if cur_batch == 0:
                ids = np.arange(X_train.shape[0])
                np.random.shuffle(ids)
                epoch += 1
            D.trainable = False
            G.trainable = True
            z = np.random.normal(0, 1, size=(batch_size, z_dim))
            loss_g = G_model.train_on_batch([real_batch, z], dummy_y)[0]
            losses_g.append(loss_g)
            gen_iters += 1
            cur_batch = (cur_batch + 1) % n_batches

        if cur_batch == 0:
            G.trainable = False
            print("epoch={}, loss_d={:.5f}, loss_g={:.5f}".format(
                epoch, np.mean(losses_d), np.mean(loss_g)))
            fake_images = G.predict(z_fixed)
            print("\tPlotting images and losses")
            plot_images(fake_images, "{}/fake_images_epoch{}.png".format(out_dir, epoch))
            plot_losses(losses_d, losses_g, "{}/losses.png".format(out_dir))
            epoch += 1


train()
