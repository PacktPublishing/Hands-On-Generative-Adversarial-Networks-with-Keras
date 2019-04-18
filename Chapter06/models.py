from keras.layers import Input, Concatenate
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import Activation
from keras.models import Model
# from BN16 import BatchNormalizationFP16 as BatchNormalization


def encoding_block(x, n_filters, kernel_size=4, strides=2):
    x = Conv2D(n_filters, kernel_size=kernel_size, strides=strides,
               padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def decoding_block(x, skip_input, n_filters, kernel_size=4):
    x = UpSampling2D(size=2)(x)
    x = Conv2D(n_filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Concatenate()([x, skip_input])
    return x


def build_generator(input_shape=(256, 256, 3), ngf=64, kernel_size=4,
                    strides=2):
    """U-Net Generator"""

    image_input = Input(shape=input_shape)
    n_channels = input_shape[-1]

    # encoding blocks
    e1 = Conv2D(ngf, kernel_size=kernel_size, strides=2, padding='same')(
        image_input)
    e1 = LeakyReLU(alpha=0.2)(e1)

    e2 = encoding_block(e1, ngf*2)
    e3 = encoding_block(e2, ngf*4)
    e4 = encoding_block(e3, ngf*8)
    e5 = encoding_block(e4, ngf*8)
    e6 = encoding_block(e5, ngf*8)
    x = encoding_block(e6, ngf*8)

    # decoding blocks
    x = decoding_block(x, e6, ngf*8)
    x = decoding_block(x, e5, ngf*8)
    x = decoding_block(x, e4, ngf*8)
    x = decoding_block(x, e3, ngf*4)
    x = decoding_block(x, e2, ngf*2)
    x = decoding_block(x, e1, ngf)

    x = UpSampling2D(size=2)(x)
    x = Conv2D(n_channels, kernel_size=4, strides=1, padding='same')(x)
    x = Activation('tanh')(x)

    # create model graph
    model = Model(inputs=image_input, outputs=x, name='Generator')

    print("\nGenerator")
    model.summary()
    return model


def build_discriminator(input_shape_a=(256, 256, 3), input_shape_b=(256, 256, 3),
                        ndf=64, n_layers=3, kernel_size=4, strides=2,
                        activation='linear'):
    """patchGAN discriminator"""
    input_a = Input(shape=input_shape_a)
    input_b = Input(shape=input_shape_b)

    x = Concatenate(axis=-1)([input_a, input_b])

    x = Conv2D(ndf, kernel_size=kernel_size, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    for i in range(1, n_layers):
        x = encoding_block(x, ndf * (2**i), kernel_size)

    ndf_mult = min(2 ** n_layers, 8)
    x = Conv2D(ndf * ndf_mult, kernel_size=kernel_size, strides=strides,
               padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(1, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = Activation(activation)(x)

    # create model graph
    model = Model(inputs=[input_a, input_b], outputs=x, name='Discriminator')

    print("\nDiscriminator")
    model.summary()
    return model
