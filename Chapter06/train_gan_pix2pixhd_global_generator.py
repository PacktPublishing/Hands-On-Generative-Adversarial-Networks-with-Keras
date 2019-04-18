import numpy as np
from functools import partial
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras import backend as K

from tensorboardX import SummaryWriter
from models_pix2pixhd import build_discriminator, build_global_generator
from utils import iterate_minibatches, log_images, log_losses, save_model


def get_d_outputs(d, input_a, input_b, input_b_fake):
    outs_real = d([input_a, input_b])
    outs_fake = d([input_a, input_b_fake])
    return outs_real[0], outs_real[1], outs_fake[0], outs_fake[1]


def compile_d(inputs, outputs, lr):
    D_model = Model(inputs=inputs, outputs=outputs)
    D_model.compile(optimizer=Adam(lr, beta_1=0.5, beta_2=0.999), loss='mse')
    return D_model


def loss_feature_matching(y_true, fake_samples, image_input, real_samples, D,
                          feature_matching_weight=10):
    f_fake = D([image_input, fake_samples])[1:]
    f_real = D([image_input, real_samples])[1:]

    loss_feat_match = 0
    for i in range(len(f_fake)):
        loss_feat_match += K.mean(K.abs(f_fake[i] - f_real[i]))
    loss_feat_match *= feature_matching_weight
    return loss_feat_match


def loss_dummy(y_true, y_pred):
    return 0


def train(data_folderpath='data/edges2shoes', image_size=256, ndf=64, ngf=64,
          lr_d=2e-4, lr_g=2e-4, n_iterations=int(1e6),
          batch_size=64, iters_per_checkpoint=100, n_checkpoint_samples=16,
          feature_matching_weight=10, out_dir='lsgan'):

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
    patch = int(img_a_shape[0] / 2**3)  # n_layers
    disc_patch_0 = (patch, patch, 1)
    disc_patch_1 = (int(patch/2), int(patch/2), 1)
    disc_patch_2 = (int(patch/4), int(patch/4), 1)
    print("img a shape ", img_a_shape)
    print("img b shape ", img_b_shape)
    print("disc_patch ", disc_patch_0)
    print("disc_patch ", disc_patch_1)
    print("disc_patch ", disc_patch_2)

    # plot real text for reference
    log_images(img_a_fixed, 'real_a', '0', logger)
    log_images(img_b_fixed, 'real_b', '0', logger)

    # build models
    D0 = build_discriminator(
        img_a_shape, img_b_shape, ndf, activation='linear', n_downsampling=0,
        name='Discriminator0')
    D1 = build_discriminator(
        img_a_shape, img_b_shape, ndf, activation='linear', n_downsampling=1,
        name='Discriminator1')
    D2 = build_discriminator(
        img_a_shape, img_b_shape, ndf, activation='linear', n_downsampling=2,
        name='Discriminator2')
    G = build_global_generator(img_a_shape, ngf)

    # build model outputs
    img_a_input = Input(shape=img_a_shape)
    img_b_input = Input(shape=img_b_shape)

    fake_samples = G(img_a_input)[0]
    D0_real = D0([img_a_input, img_b_input])[0]
    D0_fake = D0([img_a_input, fake_samples])[0]

    D1_real = D1([img_a_input, img_b_input])[0]
    D1_fake = D1([img_a_input, fake_samples])[0]

    D2_real = D2([img_a_input, img_b_input])[0]
    D2_fake = D2([img_a_input, fake_samples])[0]

    # define D graph and optimizer
    G.trainable = False
    D0.trainable = True
    D1.trainable = True
    D2.trainable = True
    D0_model = Model([img_a_input, img_b_input], [D0_real, D0_fake],
                     name='Discriminator0_model')
    D1_model = Model([img_a_input, img_b_input], [D1_real, D1_fake],
                     name='Discriminator1_model')
    D2_model = Model([img_a_input, img_b_input], [D2_real, D2_fake],
                     name='Discriminator2_model')
    D0_model.compile(optimizer=Adam(lr_d, beta_1=0.5, beta_2=0.999),
                     loss=['mse', 'mse'])
    D1_model.compile(optimizer=Adam(lr_d, beta_1=0.5, beta_2=0.999),
                     loss=['mse', 'mse'])
    D2_model.compile(optimizer=Adam(lr_d, beta_1=0.5, beta_2=0.999),
                     loss=['mse', 'mse'])

    # define D(G(z)) graph and optimizer
    G.trainable = True
    D0.trainable = False
    D1.trainable = False
    D2.trainable = False

    loss_fm0 = partial(loss_feature_matching, image_input=img_a_input,
                       real_samples=img_b_input, D=D0,
                       feature_matching_weight=feature_matching_weight)
    loss_fm1 = partial(loss_feature_matching, image_input=img_a_input,
                       real_samples=img_b_input, D=D1,
                       feature_matching_weight=feature_matching_weight)
    loss_fm2 = partial(loss_feature_matching, image_input=img_a_input,
                       real_samples=img_b_input, D=D2,
                       feature_matching_weight=feature_matching_weight)

    G_model = Model(
        inputs=[img_a_input, img_b_input],
        outputs=[D0_fake, D1_fake, D2_fake, fake_samples, fake_samples, fake_samples])
    G_model.compile(Adam(lr=lr_g, beta_1=0.5, beta_2=0.999),
                    loss=['mse', 'mse', 'mse', loss_fm0, loss_fm1, loss_fm2])

    ones_0 = np.ones((batch_size, ) + disc_patch_0, dtype=np.float32)
    ones_1 = np.ones((batch_size, ) + disc_patch_1, dtype=np.float32)
    ones_2 = np.ones((batch_size, ) + disc_patch_2, dtype=np.float32)
    zeros_0 = np.zeros((batch_size, ) + disc_patch_0, dtype=np.float32)
    zeros_1 = np.zeros((batch_size, ) + disc_patch_1, dtype=np.float32)
    zeros_2 = np.zeros((batch_size, ) + disc_patch_2, dtype=np.float32)
    dummy = np.ones((batch_size, ), dtype=np.float32)

    for i in range(n_iterations):
        D0.trainable = True
        D1.trainable = True
        D2.trainable = True
        G.trainable = False

        image_ab_batch, _ = next(data_iterator)
        fake_image = G.predict(image_ab_batch[:, 0])[0]
        loss_d0 = D0_model.train_on_batch(
            [image_ab_batch[:, 0], image_ab_batch[:, 1]], [ones_0, zeros_0])
        loss_d1 = D0_model.train_on_batch(
            [image_ab_batch[:, 0], image_ab_batch[:, 1]], [ones_1, zeros_1])
        loss_d2 = D0_model.train_on_batch(
            [image_ab_batch[:, 0], image_ab_batch[:, 1]], [ones_2, zeros_2])

        D0.trainable = False
        D1.trainable = False
        D2.trainable = False
        G.trainable = True
        image_ab_batch, _ = next(data_iterator)
        loss_g = G_model.train_on_batch(
            [image_ab_batch[:, 0], image_ab_batch[:, 1]],
            [ones, ones, ones, dummy, dummy, dummy])

        print("iter", i)
        if (i % iters_per_checkpoint) == 0:
            G.trainable = False
            fake_image = G.predict(img_a_fixed)
            log_images(fake_image, 'val_fake', i, logger)
            save_model(G, out_dir)

        log_losses(loss_d, loss_g, i, logger)

train()
