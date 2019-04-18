from keras.layers import Input, Conv2D, Activation, BatchNormalization
from keras.layers import GlobalAveragePooling2D, UpSampling2D
from keras.layers import Add
from keras.initializers import RandomNormal

weight_init = RandomNormal(mean=0., stddev=0.05)


def MeanPoolConv(x, n_filters):
    x = Conv2D(filters=n_filters, kernel_size=(1, 1), strides=1, padding='same',
               kernel_initializer=weight_init)(x)
    x = Lambda(lambda v: (v[:, ::2, ::2, :] +
                          v[:, 1::2, ::2, :] +
                          v[:, ::2, 1::2, :] +
                          v[:, 1::2, 1::2, :]) / 4.)(x)
    return x


def ConvMeanPool(x, n_filters, kernel_size):
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=1,
               padding='same', kernel_initializer=weight_init)(x)

    x = Lambda(lambda v: (v[:, ::2, ::2, :] +
                          v[:, 1::2, ::2, :] +
                          v[:, ::2, 1::2, :] +
                          v[:, 1::2, 1::2, :]) / 4.)(x)
    return x


def UpsampleConv(x, n_filters):
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=n_filters, kernel_size=1, strides=1, padding='same',
               kernel_initializer=weight_init, bias=False)(x)
    return x


def resnet_block_generator(input, n_blocks, n_filters, kernel_size=(3, 3), stride=2):
    output = input
    for i in range(n_blocks):
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Conv2DTranspose(filters=n_filters, kernel_size=kernel_size,
                                 strides=stride, padding='same',
                                 kernel_initializer=weight_init)(output)

        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=1,
                        padding='same', kernel_initializer=weight_init)(output)

        if input.shape[1:] != output.shape[1:]:
            # Upsample input to match output dimension
            input = UpsampleConv(input, n_filters)
            print("resnet: adding layer to match residual input to output")

        # Residual Connection
        output = Add()([input, output])

    return output

from keras.layers import Input, Conv2D, Activation, BatchNormalization, AveragePooling2D
from keras.layers import Add

def resnet_block_discriminator(input, n_blocks, n_filters, kernel_size=(3, 3), stride=1):
    output = input
    for i in range(n_blocks):
        output = BatchNormalization()(output)
        output = LeakyReLU(0.2)(output)
        output = Conv2D(filters=n_filters, kernel_size=kernel_size,
                        strides=stride, padding='same', bias=False,
                        kernel_initializer=weight_init)(output)

        output = BatchNormalization()(output)
        output = LeakyReLU(0.2)(output)
        output = ConvMeanPool(output, n_filters, kernel_size)

        if input.shape[1:] != output.shape[1:]:
            # Downsample input to match output dimension
            """
            input = AveragePooling2D(pool_size=(1, 1),
                strides=stride, padding='same')(input)
            """
            input  = MeanPoolConv(input, n_filters)
            print("resnet: adding layer to match residual input to output")

        # Residual Connection
        output = Add()([input, output])
    return output


from keras.layers import Input, Activation, ZeroPadding2D, Dense
from keras.layers import Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Lambda, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def build_cifar10_resnet_generator(input_shape, n_filters, n_residual_blocks,
                                   n_channels):
    """ adapted from wgan-gp """

    inputs = Input(shape=input_shape)

    # Dense 1: 4 x 4 x n_filters
    x = Dense(4 * 4 * n_filters, input_shape=input_shape)(inputs)
    x = Reshape((4, 4, n_filters))(x)

    # ResNet blocks
    x = resnet_block_generator(x, n_residual_blocks, n_filters)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # Output layer
    x = Conv2D(filters=n_channels, kernel_size=(3, 3), padding='same')(x)
    x = Activation('tanh')(x)

    # create model graph
    model = Model(inputs=inputs, outputs=x, name='Generator')

    print("\nGenerator ResNet")
    model.summary()
    return model


"""
ngf = 128
n_channels = 3
n_residual_blocks_generator = 3
z_dim = 128
input_shape_generator = (z_dim, )
g = G(input_shape_generator, ngf, n_residual_blocks_generator, n_channels)
"""

from keras.layers import Input, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def build_cifar10_resnet_discriminator(input_shape, n_filters,
                                       n_residual_blocks, kernel_size=(3, 3),
                                       stride=1):
    """ adapted from wgan-gp """
    input = Input(shape=input_shape)
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=stride, padding='same')(input)
    x = LeakyReLU(0.2)(x)
    x = ConvMeanPool(x, n_filters, kernel_size)

    # shortcut
    inputs = MeanPoolConv(input, n_filters)

    x = Add()([inputs, x])

    x = resnet_block_discriminator(x, n_residual_blocks, n_filters)
    x = LeakyReLU(0.2)(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(1)(x)

    # create model graph
    model = Model(inputs=input, outputs=x, name='Discriminator')

    print("\nDiscriminator ResNet")
    model.summary()
    return model


"""
ndf = 128
input_shape_discriminator = (32, 32, n_channels)
n_residual_blocks_discriminator = 3
d = build_cifar10_resnet_discriminator(input_shape_discriminator, ndf, n_residual_blocks_discriminator)
"""
