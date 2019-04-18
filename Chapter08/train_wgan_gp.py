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

The model saves text using pillow. If you don't have pillow, either install it or remove the calls to generate_text.
"""
import numpy as np
from functools import partial
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.layers.merge import _Merge
from keras import backend as K

from models import (build_resnet_discriminator, build_resnet_generator)

from tensorboardX import SummaryWriter
from data_utils import get_data, iterate_minibatches, sample, log_text

N_CRITIC_ITERS = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
GLOB_STR = "data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/*"
BATCH_SIZE = 0


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


def loss_gradient_penalty(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.

    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term
    to the loss function that penalizes the network if the gradient norm moves
    away from 1. However, it is impossible to evaluate this function at all
    points in the input space. The compromise used in the paper is to choose
    random points on the lines between real and generated samples, and check the
    gradients at these points. Note that it is the gradient w.r.t. the input
    averaged samples, not the weights of the discriminator, that we're
    penalizing!

    In order to evaluate the gradients, we must first run samples through the
    generator and evaluate the loss.  Then we get the gradients of the
    discriminator w.r.t. the input averaged samples.  The l2 norm and penalty
    can then be calculated for this gradient.

    Note that this loss function requires the original averaged samples as
    input, but Keras only supports passing y_true and y_pred to loss functions.
    To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # get gradients of D(averaged) wrt averaged
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm:
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (||grad|| - 1)^2
    gradient_penalty = gradient_penalty_weight * K.square(gradient_l2_norm - 1)
    # compute mean
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms,
    this outputs a random point on the line between each pair of input points.

    Inheriting from _Merge is a little messy but it was the quickest solution I
    could think of.  Improvements appreciated."""
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def log_losses(loss_d, loss_g, iteration, logger):
    names = ['loss_d', 'loss_d_real', 'loss_d_fake', 'loss_d_gp']
    for i in range(len(loss_d)):
        logger.add_scalar(names[i], loss_d[i], iteration)
    logger.add_scalar("losses_g", loss_g, iteration)


def train(ndf=512, ngf=512, z_dim=128, n_residual_blocks_discriminator=5,
          n_residual_blocks_generator=5, lr_d=1e-4, lr_g=1e-4,
          n_iterations=int(1e6), batch_size=64, iters_per_checkpoint=2000,
          n_checkpoint_text=36, vocabulary_size=2048, max_sentence_length=32,
          sos='<s>', eos='</s>', unk_token='<unk>', symbol_based=False,
          word_join=' ', out_dir='wgan_gp_word'):
    global BATCH_SIZE

    BATCH_SIZE = batch_size
    logger = SummaryWriter(out_dir)
    logger.add_scalar('d_lr', lr_d, 0)
    logger.add_scalar('g_lr', lr_g, 0)
    data, token_to_id, id_to_token = get_data(
        GLOB_STR, vocabulary_size, sos, eos, unk_token, symbol_based)
    data_iterator = iterate_minibatches(
        data, token_to_id, vocabulary_size, max_sentence_length, batch_size)

    text_shape = next(data_iterator)[0].shape
    print("text shape {}".format(text_shape))

    # plot real text for reference
    real_sample = next(data_iterator)
    log_text(sample(real_sample, id_to_token, word_join), 'real', '0', logger)

    # build models
    input_shape_discriminator = text_shape
    input_shape_generator = (z_dim, )
    D = build_resnet_discriminator(
        input_shape_discriminator, ndf, n_residual_blocks_discriminator)
    G = build_resnet_generator(
        input_shape_generator, ngf, n_residual_blocks_generator,
        max_sentence_length, vocabulary_size)

    # build model outputs
    real_inputs = Input(shape=real_sample.shape[1:])
    z = Input(shape=(z_dim, ))
    fake_samples = G(z)
    D_real = D(real_inputs)
    D_fake = D(fake_samples)
    averaged_samples = RandomWeightedAverage()([real_inputs, fake_samples])
    D_real = D(real_inputs)
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
    G.trainable = False
    D.trainable = True
    D_model = Model(inputs=[real_inputs, z],
                    outputs=[D_real, D_fake, D_averaged])
    D_model.compile(optimizer=Adam(lr_d, beta_1=0.5, beta_2=0.9),
                    loss=[loss_wasserstein, loss_wasserstein, loss_gp])

    # define D(G(z)) graph and optimizer
    G.trainable = True
    D.trainable = False
    G_model = Model(inputs=z, outputs=D_fake)
    G_model.compile(Adam(lr=lr_g, beta_1=0.5, beta_2=0.9),
                    loss=loss_wasserstein)

    ones = np.ones((batch_size, 1), dtype=np.float32)
    minus_ones = -ones
    dummy = np.zeros((batch_size, 1), dtype=np.float32)

    # fix a z vector for training evaluation
    z_fixed = np.random.uniform(-1, 1, size=(n_checkpoint_text, z_dim))

    for i in range(n_iterations):
        D.trainable = True
        G.trainable = False
        for j in range(N_CRITIC_ITERS):
            z = np.random.normal(0, 1, size=(batch_size, z_dim))
            real_batch = next(data_iterator)
            losses_d = D_model.train_on_batch(
                [real_batch, z], [ones, minus_ones, dummy])

        D.trainable = False
        G.trainable = True
        z = np.random.normal(0, 1, size=(batch_size, z_dim))
        loss_g = G_model.train_on_batch(z, ones)

        if (i % iters_per_checkpoint) == 0:
            G.trainable = False
            fake_text = G.predict(z_fixed)
            log_text(sample(fake_text, id_to_token, word_join), 'fake', i,
                     logger)

        log_losses(losses_d, loss_g, i, logger)

train()
