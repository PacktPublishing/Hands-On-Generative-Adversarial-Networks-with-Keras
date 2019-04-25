from time import clock
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
# from keras import backend as K
# K.set_floatx('float16')
# K.set_epsilon(1e-4)

from models import build_discriminator, build_generator

from tensorboardX import SummaryWriter
from utils import get_data, iterate_minibatches, images_from_bytes
from utils import log_text, log_images, log_losses, save_model


def train(data_filepath='data/flowers.hdf5', ndf=64, ngf=128, z_dim=128,
          emb_dim=128, lr_d=2e-4, lr_g=2e-4, n_iterations=int(1e6),
          batch_size=64, iters_per_checkpoint=500, n_checkpoint_samples=16,
          out_dir='gan'):

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
    D = build_discriminator(img_shape, emb_shape, emb_dim, ndf,
                            activation='sigmoid')
    G = build_generator(z_shape, emb_shape, emb_dim, ngf)

    # build model outputs
    real_inputs = Input(shape=img_shape)
    txt_inputs = Input(shape=emb_shape)
    txt_shuf_inputs = Input(shape=emb_shape)
    z_inputs = Input(shape=(z_dim, ))

    fake_samples = G([z_inputs, txt_inputs])
    D_real = D([real_inputs, txt_inputs])
    D_wrong = D([real_inputs, txt_shuf_inputs])
    D_fake = D([fake_samples, txt_inputs])

    # define D graph and optimizer
    G.trainable = False
    D.trainable = True
    D_model = Model(inputs=[real_inputs, txt_inputs, txt_shuf_inputs, z_inputs],
                    outputs=[D_real, D_wrong, D_fake])
    D_model.compile(optimizer=Adam(lr_d, beta_1=0.5, beta_2=0.9),
                    loss='binary_crossentropy', loss_weights=[1, 0.5, 0.5])

    # define D(G(z)) graph and optimizer
    G.trainable = True
    D.trainable = False
    G_model = Model(inputs=[z_inputs, txt_inputs], outputs=D_fake)
    G_model.compile(Adam(lr=lr_g, beta_1=0.5, beta_2=0.9),
                    loss='binary_crossentropy')

    ones = np.ones((batch_size, 1, 1, 1), dtype=np.float32)
    zeros = np.zeros((batch_size, 1, 1, 1), dtype=np.float32)

    # fix a z vector for training evaluation
    z_fixed = np.random.uniform(-1, 1, size=(n_checkpoint_samples, z_dim))

    for i in range(n_iterations):
        start = clock()
        D.trainable = True
        G.trainable = False
        z = np.random.normal(0, 1, size=(batch_size, z_dim))
        real_batch = next(data_iterator)
        images_batch = images_from_bytes(real_batch[0])
        emb_text_batch = real_batch[1]
        ids = np.arange(len(emb_text_batch))
        np.random.shuffle(ids)
        emb_text_batch_shuffle = emb_text_batch[ids]
        loss_d = D_model.train_on_batch(
            [images_batch, emb_text_batch, emb_text_batch_shuffle, z],
            [ones, zeros, zeros])

        D.trainable = False
        G.trainable = True
        z = np.random.normal(0, 1, size=(batch_size, z_dim))
        real_batch = next(data_iterator)
        loss_g = G_model.train_on_batch([z, real_batch[1]], ones)

        print("iter", i, "time", clock() - start)
        if (i % iters_per_checkpoint) == 0:
            G.trainable = False
            fake_image = G.predict([z_fixed, emb_fixed])
            log_images(fake_image, 'val_fake', i, logger)
            save_model(G, 'gan')

        log_losses(loss_d, loss_g, i, logger)


train()
