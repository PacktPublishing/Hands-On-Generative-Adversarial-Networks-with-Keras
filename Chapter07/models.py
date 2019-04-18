# adapted from https://github.com/MSC-BUAA/Keras-progressive_growing_of_gans

import keras.backend as K
from keras.layers import *
from keras.models import Model, Sequential
import numpy as np
from layers import *


def n_filters(stage, fmap_base, fmap_max):
    return min(int(fmap_base / (2.0 ** stage)), fmap_max)


def Generator(n_channels=1, resolution=32, z_dim=512, n_labels=0,
              fmap_base=4096, fmap_max=128, normalize_z=True):
    # model setup
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2 ** resolution_log2 and resolution >= 4
    cur_block = K.variable(0.0, dtype='float32')

    # initialize z input
    z = Input(shape=[z_dim])

    # normalize z's such that they lay on a unit hypersphere
    x = PixelNormLayer()(z) if normalize_z else z

    # possibly concatenate z and labels
    inputs = z
    if n_labels:
        labels = Input(shape=[n_labels])
        inputs = [z, labels]
        x = Concatenate()([x, labels])

    # first block 4x4, sqrt(2)/4 as in github
    x = DenseBlock(x, n_filters(1, fmap_base, fmap_max)*4*4, gain=np.sqrt(2)/4,
                   use_pixelnorm=True, use_activation=True,
                   reshape=(4, 4, n_filters(1, fmap_base, fmap_max)))
    x = ConvBlock(x, n_filters(1, fmap_base, fmap_max), 3, 'same',
                  use_pixelnorm=True)

    # subsequent blocks or stages 8 x 8 and larger...
    block_activations = [x]
    for block in range(2, resolution_log2):
        x = UpsamplingLayer(factor=2)(x)
        x = ConvBlock(x, n_filters(block, fmap_base, fmap_max), 3, 'same',
                      use_pixelnorm=True)
        x = ConvBlock(x, n_filters(block, fmap_base, fmap_max), 3, 'same',
                      use_pixelnorm=True)
        block_activations.append(x)

    # compute the final output for each block
    block_outputs = [Conv1x1(l, n_channels, gain=1.0, use_activation=False)
                     for l in reversed(block_activations)]

    # select the output block
    output = BlockSelectionLayer(cur_block)(block_outputs)

    # instantiate the model and current block
    model = Model(inputs=inputs, outputs=[output])
    model.cur_block = cur_block
    print('Generator')
    model.summary()
    return model


def Discriminator(n_channels=1, resolution=32, n_labels=0, fmap_base=4096,
                  fmap_max=128):

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2 ** resolution_log2 and resolution >= 4
    cur_block = K.variable(np.float(0.0), dtype='float32')

    images_in = Input(shape=[resolution, resolution, n_channels])
    images = images_in

    # the first block has a 1x1 conv
    x = Conv1x1(images, n_filters(resolution_log2-1, fmap_base, fmap_max),
                use_activation=True)

    # subsequent blocks or stages
    block_inputs = []
    for i in range(resolution_log2-1, 1, -1):
        x = ConvBlock(x, n_filters(i, fmap_base, fmap_max), 3, 'same',
                      use_pixelnorm=False)
        x = ConvBlock(x, n_filters(i - 1, fmap_base, fmap_max), 3, 'same',
                      use_pixelnorm=False)
        x = DownsamplingLayer(factor=2)(x)
        block = DownsamplingLayer(factor=2 ** (resolution_log2-i))(images)
        block = Conv1x1(block, n_filters(i - 1, fmap_base, fmap_max),
                        use_activation=True)
        x = BlockSelectionLayer(
            cur_block, first_incoming_block=resolution_log2 - i - 1)([x, block])

    # compute and concatenate minibatch statistics
    x = MinibatchStatConcatLayer()(x)

    # last conv
    x = ConvBlock(x, n_filters(1, fmap_base, fmap_max), 3, 'same', False)
    x = ConvBlock(x, n_filters(0, fmap_base, fmap_max), 4, 'valid', False)

    # last dense block is linear
    outputs = DenseBlock(x, 1+n_labels, gain=1.0, use_pixelnorm=False,
                         use_activation=False)

    # instantiate the model and current block
    model = Model(inputs=[images_in], outputs=outputs)
    model.cur_block = cur_block
    print('Discriminator')
    model.summary()
    return model


def GAN(G, D, z_dim, n_labels, resolution, n_channels):
    G_train = Sequential([G, D])
    G_train.cur_block = G.cur_block

    shape = D.get_input_shape_at(0)[1:]
    gen_input, real_input = Input(shape), Input(shape)
    interpolation = Input(shape)

    sub = Subtract()([D(gen_input), D(real_input)])
    norm = GradNorm()([D(interpolation), interpolation])
    D_train = Model([real_input, gen_input, interpolation],
                    [sub, norm, Reshape((1, ))(D(real_input))])
    D_train.cur_block = D.cur_block

    return G_train, D_train
