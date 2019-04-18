import numpy as np
from functools import partial
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras import backend as K

from models import (build_resnet_discriminator,
                    build_resnet_generator)
from dcgan import (build_dcgan_discriminator,
                   build_dcgan_generator)

from tensorboardX import SummaryWriter
from data_utils import get_data, iterate_minibatches, sample, print_data
from data_utils import log_text
GLOB_STR = "data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/*"

def rel_disc_loss(y_true, y_pred, disc_r=None, disc_f=None):
    epsilon = 0.000001
    return -(K.mean(K.log(K.sigmoid(disc_r - K.mean(disc_f, axis=0))+epsilon), axis=0)\
           +K.mean(K.log(1-K.sigmoid(disc_f - K.mean(disc_r, axis=0))+epsilon), axis=0))


def rel_gen_loss(y_true, y_pred, disc_r=None, disc_f=None):
    epsilon=0.000001
    return -(K.mean(K.log(K.sigmoid(disc_f - K.mean(disc_r, axis=0))+epsilon), axis=0)\
           +K.mean(K.log(1-K.sigmoid(disc_r - K.mean(disc_f, axis=0))+epsilon), axis=0))


def log_losses(loss_d, loss_g, iteration, logger):
    logger.add_scalar("loss_d", loss_d, iteration)
    logger.add_scalar("loss_g", loss_g, iteration)


def train(ndf=128, ngf=128, z_dim=128, n_residual_blocks_discriminator=5,
          n_residual_blocks_generator=5, lr_d=1e-5, lr_g=1e-5,
          n_iterations=int(1e6), batch_size=64, n_checkpoint_text=36,
          vocabulary_size=96, max_sentence_length=32, eos_token='</s>',
          out_dir='rgan'):

    logger = SummaryWriter(out_dir)
    data, token_to_id, id_to_token = get_data(GLOB_STR, vocabulary_size)
    data_iterator = iterate_minibatches(
        data, token_to_id, vocabulary_size, max_sentence_length, eos_token,
        batch_size)

    text_shape = next(data_iterator)[0].shape
    print("text shape {}".format(text_shape))

    # plot real text for reference
    real_sample = next(data_iterator)
    print_data(sample(real_sample, id_to_token)[:4])

    # build models
    input_shape_discriminator = text_shape
    input_shape_generator = (z_dim, )
    D = build_resnet_discriminator(
        input_shape_discriminator, ndf, n_residual_blocks_discriminator)
    G = build_resnet_generator(
        input_shape_generator, ngf, n_residual_blocks_generator,
        max_sentence_length, vocabulary_size)

    #D = build_dcgan_discriminator(ndf, text_shape)
    #G = build_dcgan_generator(ngf, z_dim, n_channels)

    # build model outputs
    real_inputs = Input(shape=real_sample.shape[1:])
    z = Input(shape=(z_dim, ))
    fake_samples = G(z)
    D_real = D(real_inputs)
    D_fake = D(fake_samples)

    # build losses
    loss_d_fn = partial(rel_disc_loss, disc_r=D_real, disc_f=D_fake)
    loss_g_fn = partial(rel_gen_loss, disc_r=D_real, disc_f=D_fake)

    # define D graph and optimizer
    G.trainable = False
    D.trainable = True
    D_model = Model(inputs=[real_inputs, z], outputs=[D_real, D_fake])
    D_model.compile(optimizer=Adam(lr_d, beta_1=0.5, beta_2=0.999),
                    loss=[loss_d_fn, None])

    # define G graph and optimizer
    G.trainable = True
    D.trainable = False
    G_model = Model(inputs=[real_inputs, z], outputs=[D_real, D_fake])

    # define G Optimizer
    G_model.compile(Adam(lr=lr_g, beta_1=0.5, beta_2=0.999),
                    loss=[loss_g_fn, None])

    # dummy loss
    dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

    # fix a z vector for training evaluation
    z_fixed = np.random.normal(0, 1, size=(n_checkpoint_text, z_dim))

    n_batches = int(len(data) / batch_size)
    cur_batch = 0
    epoch = 0
    for i in range(n_iterations):
        D.trainable = True
        G.trainable = False
        z = np.random.normal(0, 1, size=(batch_size, z_dim))
        real_batch = next(data_iterator)
        loss_d = D_model.train_on_batch([real_batch, z], dummy_y)[0]
        cur_batch = (cur_batch + 1) % n_batches

        if cur_batch == 0:
            G.trainable = False
            fake_text = G.predict(z_fixed)
            log_text(dist_to_sentence(fake_text, id_to_token)[: 4], i, 'fake',
                     logger)
            epoch += 1

        D.trainable = False
        G.trainable = True
        z = np.random.normal(0, 1, size=(batch_size, z_dim))
        loss_g = G_model.train_on_batch([real_batch, z], dummy_y)[0]
        cur_batch = (cur_batch + 1) % n_batches

        if cur_batch == 0:
            G.trainable = False
            fake_text = G.predict(z_fixed)
            log_text(dist_to_sentence(fake_text, id_to_token)[: 4], i, 'fake',
                     logger)
            epoch += 1

        log_losses(loss_d, loss_g, i, logger)


train()
