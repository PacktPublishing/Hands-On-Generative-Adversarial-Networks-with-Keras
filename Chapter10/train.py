import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
from functools import partial
import os
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam, RMSprop
from keras import backend as K

from models import (build_segan_discriminator, build_segan_generator)
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from glob import glob

from tensorboardX import SummaryWriter


CLEAN_DATA_PATH = "data/clean_trainset_wav_16k/"
NOISY_DATA_PATH = "data/noisy_trainset_wav_16k/"
MAX_WAV_VALUE = 32768.0


def mean_absolute_error(y_true, y_pred, denoised_audio, clean_audio):
    return K.mean(K.abs(clean_audio - denoised_audio))


def iterate_minibatches(clean_data_path, noisy_data_path, segment_length=16384,
                        batch_size=128):
    clean_filepaths = sorted(glob(clean_data_path+"*.wav"))
    noisy_filepaths = sorted(glob(noisy_data_path+"*.wav"))
    n_files = len(clean_filepaths)
    cur_batch = 0
    while True:
        # drop last?
        if (n_files - cur_batch*batch_size) < batch_size or cur_batch == 0:
            ids = np.random.randint(0, n_files, n_files)
            np.random.shuffle(ids)
            cur_batch = 0

        train_data = []
        for i in range(batch_size):
            sr_clean, clean_data = read(
                clean_filepaths[ids[cur_batch*batch_size+i]])
            sr_noisy, noisy_data = read(
                noisy_filepaths[ids[cur_batch*batch_size+i]])
            start_id = np.random.randint(0, len(clean_data) - segment_length)
            clean_data = clean_data[start_id:start_id+segment_length]
            noisy_data = noisy_data[start_id:start_id+segment_length]
            clean_data = clean_data[None, :, None]
            noisy_data = noisy_data[None, :, None]
            train_data.append([clean_data, noisy_data])

        cur_batch = (cur_batch + 1) % int(len(clean_filepaths)/batch_size)

        train_data = np.array(train_data) / MAX_WAV_VALUE
        yield train_data, cur_batch


def log_audio(audios, logger, name="synthesis", sr=16000):
    for i in range(len(audios)):
        logger.add_audio('{}_{}'.format(name, i), audios[i], 0, sample_rate=sr)


def log_losses(loss_d, loss_g, iteration, logger):
    logger.add_scalar("losses_d", loss_d[0], iteration)
    logger.add_scalar("losses_d_real", loss_d[1], iteration)
    logger.add_scalar("losses_d_fake", loss_d[2], iteration)
    logger.add_scalar("losses_g", loss_g[0], iteration)
    logger.add_scalar("losses_g_fake", loss_g[1], iteration)
    logger.add_scalar("losses_g_reconstruction", loss_g[2], iteration)


def train(ndf=64, ngf=64, use_upsampling=False, lr_d=5e-5,
          lr_g=5e-5, n_iterations=int(1e6), batch_size=256,
          n_checkpoint_audio=8, segment_length=16384, reconstruction_weight=100,
          out_dir='lsgan_segan_transpose'):

    logger = SummaryWriter(out_dir)
    data_iterator = iterate_minibatches(
        CLEAN_DATA_PATH, NOISY_DATA_PATH, segment_length, batch_size)

    # build models
    clean_input_shape = (1, 16384, 1)
    noisy_input_shape = (1, 16384, 1)
    z_input_shape = (1, 16, 1024)

    D = build_segan_discriminator(noisy_input_shape, clean_input_shape)
    G = build_segan_generator(noisy_input_shape, z_input_shape,
                              use_upsampling=use_upsampling)

    # define D graph and optimizer
    D.compile(optimizer=RMSprop(lr_d), loss='mse')

    # define D(G(z)) graph and optimizer
    D.trainable = False
    z_input = Input(shape=z_input_shape)
    noisy_input = Input(shape=noisy_input_shape)
    clean_input = Input(shape=clean_input_shape)
    denoised_output = G([noisy_input, z_input])
    D_fake = D([noisy_input, denoised_output])
    D_of_G = Model(inputs=[noisy_input, z_input, clean_input],
                   outputs=[D_fake, denoised_output])

    # reconstruction loss
    loss_reconstruction = partial(mean_absolute_error,
                                  denoised_audio=denoised_output,
                                  clean_audio=clean_input)

    # define Generator's Optimizer
    D_of_G.compile(RMSprop(lr=lr_g), loss=['mse', loss_reconstruction],
                   loss_weights=[1, reconstruction_weight])

    ones = np.ones((batch_size, 1), dtype=np.float32)
    zeros = np.zeros((batch_size, 1), dtype=np.float32)
    dummy = np.zeros((batch_size, 1), dtype=np.float32)

    # fix a z vector for training evaluation
    z_fixed = np.random.normal(0, 1, size=(n_checkpoint_audio,) + z_input_shape)
    data_batch, cur_batch = next(data_iterator)
    clean_fixed = data_batch[:n_checkpoint_audio, 0]
    noisy_fixed = data_batch[:n_checkpoint_audio, 1]
    log_audio(clean_fixed[:, 0, :, 0], logger, 'clean')
    log_audio(noisy_fixed[:, 0, :, 0], logger, 'noisy')

    epoch = 0
    for i in range(n_iterations):
        if cur_batch == 1:
            G.trainable = False
            fake_audio = G.predict([noisy_fixed, z_fixed])
            log_audio(fake_audio[:n_checkpoint_audio, 0, :, 0], logger,
                      'denoised')
            epoch += 1
        D.trainable = True
        G.trainable = False

        z = np.random.normal(0, 1, size=(batch_size, ) + z_input_shape)
        data_batch, cur_batch = next(data_iterator)
        clean_batch = data_batch[:, 0]
        noisy_batch = data_batch[:, 1]
        fake_batch = G.predict([noisy_batch, z])
        loss_real = D.train_on_batch([noisy_batch, clean_batch], ones)
        loss_fake = D.train_on_batch([noisy_batch, fake_batch], zeros)
        loss_d = [loss_real + loss_fake, loss_real, loss_fake]

        D.trainable = False
        G.trainable = True
        data_batch, cur_batch = next(data_iterator)
        clean_batch = data_batch[:, 0]
        noisy_batch = data_batch[:, 1]
        z = np.random.normal(0, 1, size=(batch_size, ) + z_input_shape)
        loss_g = D_of_G.train_on_batch([noisy_batch, z, clean_batch],
                                       [ones, dummy])
        fake_audio = G.predict([noisy_fixed, z_fixed])
        log_losses(loss_d, loss_g, i, logger)
        print("nxt_batch", cur_batch, "min", fake_audio.min(), "max",
              fake_audio.max())

train()
