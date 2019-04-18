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
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K

from keras.datasets import cifar10

from resnet import build_cifar10_resnet_discriminator, build_cifar10_resnet_generator

N_CRITIC_ITERS = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper


def loss_wasserstein(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.

    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.

    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.

    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)


def loss_gradient_penalty(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.

    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!

    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.

    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.

    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


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
    fig, axes = plt.subplots(3, 2, figsize=(8, 8))
    axes = axes.flatten()
    axes[0].plot(losses_d[:, 0])
    axes[1].plot(losses_d[:, 1])
    axes[2].plot(losses_d[:, 2])
    axes[3].plot(losses_d[:, 3])
    axes[4].plot(losses_g)
    axes[0].set_title("losses_d")
    axes[1].set_title("losses_d_real")
    axes[2].set_title("losses_d_fake")
    axes[3].set_title("losses_d_gp")
    axes[4].set_title("losses_g")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def train(ndf=128, ngf=128, z_dim=128, n_residual_blocks_discriminator=3,
          n_residual_blocks_generator=3, lr_d=2e-4, lr_g=2e-4,
          n_iterations=int(1e6), batch_size=128, n_checkpoint_images=36,
          out_dir='wgan-gp_resnet_nobn'):
    global BATCH_SIZE
    BATCH_SIZE = batch_size
    X_train, _ = get_data()
    image_shape = X_train[0].shape
    print("image shape {}, min val {}, max val {}".format(
        image_shape, X_train[0].min(), X_train[0].max()))

    # plot real images for reference
    plot_images(X_train[:n_checkpoint_images], "{}/real_images.png".format(out_dir))

    # build models
    input_shape_generator = (z_dim, )
    n_channels = image_shape[-1]
    input_shape_discriminator = image_shape

    D = build_cifar10_resnet_discriminator(
        input_shape_discriminator, ndf, n_residual_blocks_discriminator)
    G = build_cifar10_resnet_generator(
        input_shape_generator, ngf, n_residual_blocks_generator, n_channels)

    real_samples = Input(shape=X_train.shape[1:])
    z = Input(shape=(z_dim, ))
    fake_samples = G(z)
    averaged_samples = RandomWeightedAverage()([real_samples, fake_samples])
    D_real = D(real_samples)
    D_fake = D(fake_samples)
    D_averaged = D(averaged_samples)

    # The gradient penalty loss function requires the input averaged samples to
    # get gradients. However, Keras loss functions can only have two arguments,
    # y_true and y_pred. We get around this by making a partial() of the
    # function with the averaged samples here.
    loss_gp = partial(loss_gradient_penalty,
                      averaged_samples=averaged_samples,
                      gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    # Functions need names or Keras will throw an error
    loss_gp.__name__ = 'loss_gradient_penalty'

    # define D graph and optimizer
    # define D graph and optimizer
    G.trainable = False
    D.trainable = True
    D_model = Model(inputs=[real_samples, z],
                    outputs=[D_real, D_fake, D_averaged])
    D_model.compile(optimizer=Adam(lr_d, beta_1=0.5, beta_2=0.9),
                    loss=[loss_wasserstein, loss_wasserstein, loss_gp])

    # define D(G(z)) graph for training the Generator
    G.trainable = True
    D.trainable = False
    G_model = Model(inputs=z, outputs=D_fake)

    # define Generator's Optimizer
    G_model.compile(Adam(lr=lr_g, beta_1=0.5, beta_2=0.9),
                   loss=loss_wasserstein)

    ones = np.ones((batch_size, 1), dtype=np.float32)
    minus_ones = -ones
    dummy = np.zeros((batch_size, 1), dtype=np.float32)

    losses_d, losses_g = [], []

    # fix a z vector for training evaluation
    z_fixed = np.random.uniform(-1, 1, size=(n_checkpoint_images, z_dim))

    gen_iters = 0
    n_batches = int(len(X_train) / batch_size)
    cur_batch = 0
    epoch = 0
    for i in range(n_iterations):
        D.trainable = True
        G.trainable = False
        for j in range(N_CRITIC_ITERS):
            if cur_batch == 0:
                ids = np.arange(X_train.shape[0])
                np.random.shuffle(ids)
                epoch += 1
            z = np.random.normal(0, 1, size=(batch_size, z_dim))
            cur_ids = ids[cur_batch * batch_size:(cur_batch+1)*batch_size]
            real_batch = X_train[cur_ids]
            loss_real, loss_fake, loss_gp, _ = D_model.train_on_batch([real_batch, z], [ones, minus_ones, dummy])
            losses_d.append([loss_real + loss_fake, loss_real, loss_fake, loss_gp])
            cur_batch = (cur_batch + 1) % n_batches

        if cur_batch == 0:
            ids = np.arange(X_train.shape[0])
            np.random.shuffle(ids)
            epoch += 1

        D.trainable = False
        G.trainable = True
        z = np.random.normal(0, 1, size=(batch_size, z_dim))
        loss_g = G_model.train_on_batch(z, ones)
        losses_g.append(loss_g)
        gen_iters += 1
        cur_batch = (cur_batch + 1) % n_batches

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
