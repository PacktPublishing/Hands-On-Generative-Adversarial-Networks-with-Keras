from keras.models import Sequential
from keras.layers import LeakyReLU, Dense
from keras.layers.core import Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Flatten
from keras import initializers


def build_cifar10_dcgan_discriminator(ndf=64, image_shape=(32, 32, 3),
                                      nonlinearity=None):
    """ Builds CIFAR10 DCGAN Discriminator Model
    PARAMS
    ------
    ndf: number of discriminator filters
    image_shape: 32x32x3

    RETURN
    ------
    D: keras sequential
    """
    init = initializers.RandomNormal(stddev=0.05)

    D = Sequential()

    # Conv 1: 16x16x64
    D.add(Conv2D(ndf, kernel_size=5, strides=2, padding='same',
                 use_bias=True, kernel_initializer=init,
                 input_shape=image_shape))
    D.add(LeakyReLU(0.2))

    # Conv 2: 8x8x128
    D.add(Conv2D(ndf*2, kernel_size=5, strides=2, padding='same',
          use_bias=True, kernel_initializer=init))
    D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    # Conv 3: 4x4x256
    D.add(Conv2D(ndf*4, kernel_size=5, strides=2, padding='same',
                 use_bias=True, kernel_initializer=init))
    D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    # Conv 4:  2x2x512
    D.add(Conv2D(ndf*8, kernel_size=5, strides=2, padding='same',
                 use_bias=True, kernel_initializer=init))
    D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    # Flatten: 2x2x512 -> (2048)
    D.add(Flatten())

    # Dense Layer
    D.add(Dense(1, kernel_initializer=init))

    print("\nDiscriminator")
    D.summary()

    return D


def build_cifar10_dcgan_generator(ngf=64, z_dim=128, n_channels=3):
    """ Builds CIFAR10 DCGAN Generator Model
    PARAMS
    ------
    ngf: number of generator filters
    z_dim: number of dimensions in latent vector

    RETURN
    ------
    G: keras sequential
    """
    init = initializers.RandomNormal(stddev=0.02)

    G = Sequential()

    # Dense 1: 2x2x512
    G.add(Dense(2*2*ngf*8, input_shape=(z_dim, ),
        use_bias=True, kernel_initializer=init))
    G.add(Reshape((2, 2, ngf*8)))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # Conv 1: 4x4x256
    G.add(Conv2DTranspose(ngf*4, kernel_size=5, strides=2, padding='same',
          use_bias=True, kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # Conv 2: 8x8x128
    G.add(Conv2DTranspose(ngf*2, kernel_size=5, strides=2, padding='same',
          use_bias=True, kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # Conv 3: 16x16x64
    G.add(Conv2DTranspose(ngf, kernel_size=5, strides=2, padding='same',
          use_bias=True, kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # Conv 4: 32x32xn_channels
    G.add(Conv2DTranspose(n_channels, kernel_size=5, strides=2, padding='same',
          use_bias=True, kernel_initializer=init))
    G.add(Activation('tanh'))

    print("\nGenerator")
    G.summary()

    return G


