import numpy as np
from functools import partial
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras import backend as K
#K.set_floatx('float16')
#K.set_epsilon(1e-4)

from tensorboardX import SummaryWriter
from models import build_discriminator, build_generator
from utils import iterate_minibatches, log_images, log_losses, save_model

def mean_absolute_error(y_true, y_pred, real_samples, fake_samples):
    return K.mean(K.abs(real_samples - fake_samples))


def train(data_folderpath='data/edges2shoes', image_size=256, ndf=64, ngf=64,
          lr_d=2e-4, lr_g=2e-4, n_iterations=int(1e6),
          batch_size=64, iters_per_checkpoint=100, n_checkpoint_samples=16,
          reconstruction_weight=100, out_dir='gan'):

    logger = SummaryWriter(out_dir)
    logger.add_scalar('d_lr', lr_d, 0)
    logger.add_scalar('g_lr', lr_g, 0)

    data_iterator = iterate_minibatches(
        data_folderpath + "/train/*.jpg", batch_size, image_size)
    val_data_iterator = iterate_minibatches(
        data_folderpath + "/val/*.jpg", n_checkpoint_samples, image_size)
    img_ab_fixed, _ = next(val_data_iterator)
    img_a_fixed, img_b_fixed = img_ab_fixed[:, 0], img_ab_fixed[:, 1]

    img_a_shape = img_a_fixed.shape[1:]
    img_b_shape = img_b_fixed.shape[1:]
    patch = int(img_a_shape[0] / 2**4)  # n_layers
    disc_patch = (patch, patch, 1)
    print("img a shape ", img_a_shape)
    print("img b shape ", img_b_shape)
    print("disc_patch ", disc_patch)

    # plot real text for reference
    log_images(img_a_fixed, 'real_a', '0', logger)
    log_images(img_b_fixed, 'real_b', '0', logger)

    # build models
    D = build_discriminator(
        img_a_shape, img_b_shape, ndf, activation='sigmoid')
    G = build_generator(img_a_shape, ngf)

    # build model outputs
    img_a_input  = Input(shape=img_a_shape)
    img_b_input  = Input(shape=img_b_shape)

    fake_samples = G(img_a_input)
    D_real = D([img_a_input, img_b_input])
    D_fake = D([img_a_input, fake_samples])

    loss_reconstruction = partial(mean_absolute_error,
                                  real_samples=img_b_input,
                                  fake_samples=fake_samples)
    loss_reconstruction.__name__ = 'loss_reconstruction'

    # define D graph and optimizer
    G.trainable = False
    D.trainable = True
    D_model = Model(inputs=[img_a_input, img_b_input],
                    outputs=[D_real, D_fake])
    D_model.compile(optimizer=Adam(lr_d, beta_1=0.5, beta_2=0.9),
                    loss='binary_crossentropy')

    # define D(G(z)) graph and optimizer
    G.trainable = True
    D.trainable = False
    G_model = Model(inputs=[img_a_input, img_b_input],
                    outputs=[D_fake, fake_samples])
    G_model.compile(Adam(lr=lr_g, beta_1=0.5, beta_2=0.9),
                    loss=['binary_crossentropy', loss_reconstruction],
                    loss_weights=[1, reconstruction_weight])

    ones = np.ones((batch_size, ) + disc_patch, dtype=np.float32)
    zeros = np.zeros((batch_size, ) + disc_patch, dtype=np.float32)
    dummy = zeros

    for i in range(n_iterations):
        D.trainable = True
        G.trainable = False

        image_ab_batch, _ = next(data_iterator)
        loss_d = D_model.train_on_batch(
            [image_ab_batch[:, 0], image_ab_batch[:, 1]],
            [ones, zeros])

        D.trainable = False
        G.trainable = True
        image_ab_batch, _ = next(data_iterator)
        loss_g = G_model.train_on_batch(
            [image_ab_batch[:, 0], image_ab_batch[:, 1]],
            [ones, dummy])

        print("iter", i)
        if (i % iters_per_checkpoint) == 0:
            G.trainable = False
            fake_image = G.predict(img_a_fixed)
            log_images(fake_image, 'val_fake', i, logger)
            save_model(G, out_dir)

        log_losses(loss_d, loss_g, i, logger)

train()
