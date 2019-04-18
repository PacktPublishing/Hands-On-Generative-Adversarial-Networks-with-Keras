import numpy as np
from functools import partial
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras import backend as K

from tensorboardX import SummaryWriter
from models_pix2pixhd import build_discriminator, build_global_generator
from models_pix2pixhd import build_encoder
from utils import iterate_minibatches_cityscapes, log_images, log_losses_pix2pixhd
from utils import save_model


def loss_feature_matching(y_true, fake_samples, lbl_map, real_samples,
                          edges_map, D, feature_matching_weight=10):
    f_fake = D([lbl_map, fake_samples, edges_map])[1:]
    f_real = D([lbl_map, real_samples, edges_map])[1:]

    loss_feat_match = 0
    for i in range(len(f_fake)):
        loss_feat_match += K.mean(K.abs(f_fake[i] - f_real[i]))
    loss_feat_match *= feature_matching_weight
    return loss_feat_match


def train(data_folderpath='data/cityscapes', image_size=256, ndf=32, ngf=64,
          lr_d=2e-4, lr_g=2e-4, n_feat_channels=3, use_edges=True,  n_iterations=int(1e6),
          batch_size=1, iters_per_checkpoint=100, n_checkpoint_samples=4,
          feature_matching_weight=10, out_dir='temp'):

    logger = SummaryWriter(out_dir)
    logger.add_scalar('d_lr', lr_d, 0)
    logger.add_scalar('g_lr', lr_g, 0)

    data_iterator = iterate_minibatches_cityscapes(
        data_folderpath, "train", use_edges, batch_size)
    val_data_iterator = iterate_minibatches_cityscapes(
        data_folderpath, "train", use_edges, n_checkpoint_samples)
    input_val, _ = next(val_data_iterator)
    img_fixed = input_val[..., :3]
    lbl_fixed = input_val[..., 3][..., None]
    edges_fixed = input_val[..., 4][..., None]

    img_shape = img_fixed.shape[1:]
    lbl_shape = lbl_fixed.shape[1:]
    edges_shape = edges_fixed.shape[1:]
    feats_shape = tuple(img_fixed.shape[1:-1]) + (n_feat_channels, )
    h_patch = int(lbl_shape[0] / 2**3)  # n_layers
    w_patch = int(lbl_shape[1] / 2**3)  # n_layers
    disc_patch_0 = (h_patch, w_patch, 1)
    disc_patch_1 = (int(h_patch/2), int(w_patch/2), 1)
    disc_patch_2 = (int(h_patch/4), int(w_patch/4), 1)
    print("img shape ", img_shape)
    print("lbl shape ", lbl_shape)
    print("edges shape ", edges_shape)
    print("disc_patch ", disc_patch_0)
    print("disc_patch ", disc_patch_1)
    print("disc_patch ", disc_patch_2)


    # plot real data for reference
    plot_dims = int(np.sqrt(img_fixed.shape[0]))
    log_images(img_fixed, 'real_img', '0', logger, img_fixed.shape[1],
               img_fixed.shape[2], plot_dims)
    log_images(lbl_fixed, 'real_lbls', '0', logger, img_fixed.shape[1],
               img_fixed.shape[2], plot_dims)
    log_images(edges_fixed, 'real_edges', '0', logger, img_fixed.shape[1],
               img_fixed.shape[2], plot_dims)

    # build models
    D0 = build_discriminator(lbl_shape, img_shape, edges_shape, ndf,
                             activation='linear', n_downsampling=0,
                             name='Discriminator0')
    D1 = build_discriminator(lbl_shape, img_shape, edges_shape, ndf,
                             activation='linear', n_downsampling=1,
                             name='Discriminator1')
    D2 = build_discriminator(lbl_shape, img_shape, edges_shape, ndf,
                             activation='linear', n_downsampling=2,
                             name='Discriminator2')
    G = build_global_generator(lbl_shape, edges_shape, feats_shape, ngf)
    E = build_encoder(img_shape, edges_shape, n_feat_channels)

    # build model outputs
    lbl_input = Input(shape=lbl_shape)
    img_input = Input(shape=img_shape)
    edges_input = Input(shape=edges_shape)

    fake_samples = G([lbl_input, edges_input, E([img_input, edges_input])])[0]
    D0_real = D0([lbl_input, img_input, edges_input])[0]
    D0_fake = D0([lbl_input, fake_samples, edges_input])[0]

    D1_real = D1([lbl_input, img_input, edges_input])[0]
    D1_fake = D1([lbl_input, fake_samples, edges_input])[0]

    D2_real = D2([lbl_input, img_input, edges_input])[0]
    D2_fake = D2([lbl_input, fake_samples, edges_input])[0]

    # define D graph and optimizer
    G.trainable = False
    D0.trainable = True
    D1.trainable = True
    D2.trainable = True
    D0_model = Model([lbl_input, img_input, edges_input], [D0_real, D0_fake],
                     name='Discriminator0_model')
    D1_model = Model([lbl_input, img_input, edges_input], [D1_real, D1_fake],
                     name='Discriminator1_model')
    D2_model = Model([lbl_input, img_input, edges_input], [D2_real, D2_fake],
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

    loss_fm0 = partial(loss_feature_matching, lbl_map=lbl_input,
                       real_samples=img_input, edges_map=edges_input, D=D0,
                       feature_matching_weight=feature_matching_weight)
    loss_fm1 = partial(loss_feature_matching, lbl_map=lbl_input,
                       real_samples=img_input, edges_map=edges_input, D=D1,
                       feature_matching_weight=feature_matching_weight)
    loss_fm2 = partial(loss_feature_matching, lbl_map=lbl_input,
                       real_samples=img_input, edges_map=edges_input, D=D2,
                       feature_matching_weight=feature_matching_weight)

    G_model = Model(
        inputs=[lbl_input, img_input, edges_input],
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
        E.trainable = False

        batch, _ = next(data_iterator)
        img = batch[..., :3]
        segmap = batch[..., 3][..., None]
        edges = batch[..., 4][..., None]
        feats = E.predict([img, edges])

        fake_image = G.predict([segmap, edges, feats])[0]
        loss_d0 = D0_model.train_on_batch([segmap, img, edges],
            [ones_0, zeros_0])
        loss_d1 = D1_model.train_on_batch([segmap, img, edges],
            [ones_1, zeros_1])
        loss_d2 = D2_model.train_on_batch([segmap, img, edges],
            [ones_2, zeros_2])

        D0.trainable = False
        D1.trainable = False
        D2.trainable = False
        G.trainable = True
        E.trainable = True

        batch, _ = next(data_iterator)
        img = batch[..., :3]
        segmap = batch[..., 3][..., None]
        edges = batch[..., 4][..., None]
        feats = E.predict([img, edges])

        loss_g = G_model.train_on_batch(
            [segmap, img, edges], [ones_0, ones_1, ones_2, dummy, dummy, dummy])

        print("iter", i)
        if (i % iters_per_checkpoint) == 0:
            G.trainable = False
            fake_image = G.predict([lbl_fixed, edges_fixed, feats])[0]
            log_images(fake_image, 'val_fake', i, logger,
                       fake_image.shape[1], fake_image.shape[2],
                       int(np.sqrt(fake_image.shape[0])))
            save_model(G, out_dir)

        log_losses_pix2pixhd([loss_d0, loss_d1, loss_d2], loss_g, i, logger)

train()
