# adapted from https://github.com/MSC-BUAA/Keras-progressive_growing_of_gans

import numpy as np
from keras.optimizers import Adam
from keras import backend as K

from models import Generator, Discriminator, GAN

from tensorboardX import SummaryWriter
from utils import iterate_minibatches, log_images, log_losses
from utils import save_model

N_CRITIC_ITERS = 1
GP_WEIGHT = 10
MINIBATCH_OVERWRITES = {
    0: 16,
    1: 16,
    2: 16,
    3: 16,
    4: 16,
    5: 16,
}


def get_interpolated_images(real_samples, fake_samples):
    p = np.random.uniform(0, 1, size=(real_samples.shape[0], 1, 1, 1))
    return p * real_samples + (1-p) * fake_samples


def loss_gradient_penalty(gamma, y_pred):
    return K.mean(K.square(y_pred - gamma) / K.square(gamma))


def loss_wasserstein(y_true, y_pred):
    return K.mean(y_true*y_pred)


def loss_mean(y_true, y_pred):
    return K.mean(y_pred)


def train(n_channels=3, resolution=32, z_dim=128, n_labels=0, lr=1e-3,
          e_drift=1e-3, wgp_target=750, initial_resolution=4, total_kimg=25000,
          training_kimg=500, transition_kimg=500, iters_per_checkpoint=500,
          n_checkpoint_images=16, glob_str='cifar10', out_dir='cifar10'):

    # instantiate logger
    logger = SummaryWriter(out_dir)

    # load data
    batch_size = MINIBATCH_OVERWRITES[0]
    train_iterator = iterate_minibatches(glob_str, batch_size, resolution)

    # build models
    G = Generator(n_channels, resolution, z_dim, n_labels)
    D = Discriminator(n_channels, resolution, n_labels)

    G_train, D_train = GAN(G, D, z_dim, n_labels, resolution, n_channels)

    D_opt = Adam(lr=lr, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
    G_opt = Adam(lr=lr, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

    # define loss functions
    D_loss = [loss_mean, loss_gradient_penalty, 'mse']
    G_loss = [loss_wasserstein]

    # compile graphs used during training
    G.compile(G_opt, loss=loss_wasserstein)
    D.trainable = False
    G_train.compile(G_opt, loss=G_loss)
    D.trainable = True
    D_train.compile(D_opt, loss=D_loss, loss_weights=[1, GP_WEIGHT, e_drift])

    # for computing the loss
    ones = np.ones((batch_size, 1), dtype=np.float32)
    zeros = ones * 0.0

    # fix a z vector for training evaluation
    z_fixed = np.random.normal(0, 1, size=(n_checkpoint_images, z_dim))

    # vars
    resolution_log2 = int(np.log2(resolution))
    starting_block = resolution_log2
    starting_block -= np.floor(np.log2(initial_resolution))
    cur_block = starting_block
    cur_nimg = 0

    # compute duration of each phase and use proxy to update minibatch size
    phase_kdur = training_kimg + transition_kimg
    phase_idx_prev = 0

    # offset variable for transitioning between blocks
    offset = 0
    i = 0
    while cur_nimg < total_kimg * 1000:
        # block processing
        kimg = cur_nimg / 1000.0
        phase_idx = int(np.floor((kimg + transition_kimg) / phase_kdur))
        phase_idx = max(phase_idx, 0.0)
        phase_kimg = phase_idx * phase_kdur

        # update batch size and ones vector if we switched phases
        if phase_idx_prev < phase_idx:
            batch_size = MINIBATCH_OVERWRITES[phase_idx]
            train_iterator = iterate_minibatches(glob_str, batch_size)
            ones = np.ones((batch_size, 1), dtype=np.float32)
            zeros = ones * 0.0
            phase_idx_prev = phase_idx

        # possibly gradually update current level of detail
        if transition_kimg > 0 and phase_idx > 0:
            offset = (kimg + transition_kimg - phase_kimg) / transition_kimg
            offset = min(offset, 1.0)
            offset = offset + phase_idx - 1
            cur_block = max(starting_block - offset, 0.0)

        # update level of detail
        K.set_value(G_train.cur_block, np.float32(cur_block))
        K.set_value(D_train.cur_block, np.float32(cur_block))

        # train D
        for j in range(N_CRITIC_ITERS):
            z = np.random.normal(0, 1, size=(batch_size, z_dim))
            real_batch = next(train_iterator)
            fake_batch = G.predict_on_batch([z])
            interpolated_batch = get_interpolated_images(real_batch, fake_batch)
            losses_d = D_train.train_on_batch(
                [real_batch, fake_batch, interpolated_batch],
                [ones, ones*wgp_target, zeros])
            cur_nimg += batch_size

        # train G
        z = np.random.normal(0, 1, size=(batch_size, z_dim))
        loss_g = G_train.train_on_batch(z, -1*ones)

        logger.add_scalar("cur_block", cur_block, i)
        logger.add_scalar("learning_rate", lr, i)
        logger.add_scalar("batch_size", z.shape[0], i)
        print("iter", i, "cur_block", cur_block, "lr", lr, "kimg", kimg,
              "losses_d", losses_d, "loss_g", loss_g)
        if (i % iters_per_checkpoint) == 0:
            G.trainable = False
            fake_images = G.predict(z_fixed)
            # log fake images
            log_images(fake_images, 'fake', i, logger,
                       fake_images.shape[1], fake_images.shape[2],
                       int(np.sqrt(n_checkpoint_images)))

            # plot real images for reference
            log_images(real_batch[:n_checkpoint_images], 'real', i, logger,
                       real_batch.shape[1], real_batch.shape[2],
                       int(np.sqrt(n_checkpoint_images)))

            # save the model to eventually resume training or do inference
            save_model(G, out_dir+"/model.json", out_dir+"/model.h5")

        log_losses(losses_d, loss_g, i, logger)
        i += 1


train()
