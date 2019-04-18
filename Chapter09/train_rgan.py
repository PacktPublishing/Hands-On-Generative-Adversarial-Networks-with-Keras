import numpy as np
from functools import partial
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras import backend as K

from models import build_discriminator, build_generator

from tensorboardX import SummaryWriter
from data_utils import get_data, iterate_minibatches, images_from_bytes
from data_utils import log_text, log_images

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


def train(data_filepath='data/flowers.hdf5', ndf=64, ngf=128, z_dim=128,
          emb_dim=128, lr_d=1e-4, lr_g=1e-4, n_iterations=int(1e6),
          batch_size=64, iters_per_checkpoint=100, n_checkpoint_samples=16,
          out_dir='rgan'):

    logger = SummaryWriter(out_dir)
    logger.add_scalar('d_lr', lr_d, 0)
    logger.add_scalar('g_lr', lr_g, 0)
    train_data = get_data(data_filepath, 'train')
    val_data = get_data(data_filepath, 'valid')
    data_iterator = iterate_minibatches(train_data, batch_size)
    val_data_iterator = iterate_minibatches(val_data, n_checkpoint_samples)
    val_data = next(val_data_iterator)
    img_fixed = images_from_bytes(val_data[0])
    emb_fixed = val_data[1]
    txt_fixed = val_data[2]

    img_shape = img_fixed[0].shape
    emb_shape = emb_fixed[0].shape
    print("emb shape {}".format(img_shape))
    print("img shape {}".format(emb_shape))
    z_shape = (z_dim, )

    # plot real text for reference
    log_images(img_fixed, 'real', '0', logger)
    log_text(txt_fixed, 'real', '0', logger)

    # build models
    D = build_discriminator(img_shape, emb_shape, emb_dim, ndf)
    G = build_generator(z_shape, emb_shape, emb_dim, ngf)

    # build model outputs
    real_inputs = Input(shape=img_shape)
    txt_inputs = Input(shape=emb_shape)
    z_inputs = Input(shape=(z_dim, ))
    fake_samples = G([z_inputs, txt_inputs])
    D_real = D([real_inputs, txt_inputs])
    D_fake = D([fake_samples, txt_inputs])

    # build losses
    loss_d_fn = partial(rel_disc_loss, disc_r=D_real, disc_f=D_fake)
    loss_g_fn = partial(rel_gen_loss, disc_r=D_real, disc_f=D_fake)

    # define D graph and optimizer
    G.trainable = False
    D.trainable = True
    D_model = Model(inputs=[real_inputs, txt_inputs, z_inputs],
                    outputs=[D_real, D_fake])
    D_model.compile(optimizer=Adam(lr_d, beta_1=0.5, beta_2=0.999),
                    loss=[loss_d_fn, None])

    # define G graph and optimizer
    G.trainable = True
    D.trainable = False
    G_model = Model(inputs=[real_inputs, z_inputs, txt_inputs],
                    outputs=[D_real, D_fake])
    G_model.compile(Adam(lr=lr_g, beta_1=0.5, beta_2=0.999),
                    loss=[loss_g_fn, None])

    # dummy loss
    dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

    # fix a z vector for training evaluation
    z_fixed = np.random.uniform(-1, 1, size=(n_checkpoint_samples, z_dim))

    for i in range(n_iterations):
        D.trainable = True
        G.trainable = False
        z = np.random.normal(0, 1, size=(batch_size, z_dim))
        real_batch = next(data_iterator)
        loss_d = D_model.train_on_batch(
                [images_from_bytes(real_batch[0]), real_batch[1], z],
                dummy_y)[0]

        D.trainable = False
        G.trainable = True
        z = np.random.normal(0, 1, size=(batch_size, z_dim))
        real_batch = next(data_iterator)
        loss_g = G_model.train_on_batch(
            [images_from_bytes(real_batch[0]), z, real_batch[1]],
            dummy_y)[0]

        print("iter", i)
        if (i % iters_per_checkpoint) == 0:
            G.trainable = False
            fake_image = G.predict([z_fixed, emb_fixed])
            log_images(fake_image, 'val_fake', i, logger)
            log_images(img_fixed, 'val_real', i, logger)
            log_text(txt_fixed, 'val_fake', i, logger)

        log_losses(loss_d, loss_g, i, logger)


train()
