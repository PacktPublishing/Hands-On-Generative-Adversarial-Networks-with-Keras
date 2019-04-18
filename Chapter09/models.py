""" https://github.com/reedscot/icml2016/blob/master/main_cls.lua inspired """
from keras.initializers import RandomNormal

w_init = RandomNormal(mean=0., stddev=0.02)
g_init = RandomNormal(mean=1., stddev=0.02)

from keras.layers import Dense, Conv2D, Conv2DTranspose
from keras.layers import Input, Concatenate, Activation
from keras.layers import Add, Lambda, Reshape, LeakyReLU
# from BN16 import BatchNormalizationFP16 as BatchNormalization
from keras.layers.normalization import BatchNormalization
from keras.backend import repeat
from keras.models import Model


def ConvBatchnormRelu(x, n_filters, kernel_size, strides, padding, relu=True):
    x = Conv2D(n_filters, kernel_size=kernel_size, strides=strides,
               padding=padding, kernel_initializer=w_init)(x)
    x = BatchNormalization(gamma_initializer=g_init)(x)
    if relu:
        x = Activation('relu')(x)
    return x


def build_generator(z_input_shape=(128,), text_input_shape=(1024,),
                    embedding_dim=128, ngf=64, n_channels=3):
    # define inputs
    z_inputs = Input(shape=z_input_shape, name='z_input')
    text_inputs = Input(shape=text_input_shape, name='text_input')

    # project text embeddings
    text_embedded = Dense(embedding_dim, input_shape=text_input_shape,
                          use_bias=True, kernel_initializer=w_init)(text_inputs)
    text_embedded = LeakyReLU(0.2)(text_embedded)

    # concatenate text_embedded and z and reshape
    x = Concatenate()([text_embedded, z_inputs])
    x = Reshape((1, 1, embedding_dim+z_input_shape[0]))(x)
    x = Conv2DTranspose(ngf*8, kernel_size=4, strides=1, padding='valid',
                        use_bias=False, kernel_initializer=w_init)(x)
    h0 = BatchNormalization(gamma_initializer=g_init)(x)

    # project text_embedded and z
    x = ConvBatchnormRelu(h0, ngf*2, 1, 1, 'same')
    x = ConvBatchnormRelu(x, ngf*2, 3, 1, 'same')
    x = ConvBatchnormRelu(x, ngf*8, 3, 1, 'same', relu=False)
    x = Activation('relu')(Add()([h0, x]))

    x = Conv2DTranspose(ngf*4, kernel_size=4, strides=2, padding='same',
                        use_bias=False, kernel_initializer=w_init)(x)
    h1 = BatchNormalization(gamma_initializer=g_init)(x)

    x = ConvBatchnormRelu(h1, ngf, 1, 1, 'same')
    x = ConvBatchnormRelu(x, ngf, 3, 1, 'same')
    x = ConvBatchnormRelu(x, ngf*4, 3, 1, 'same', relu=False)
    x = Activation('relu')(Add()([h1, x]))

    x = Conv2DTranspose(ngf*2, kernel_size=4, strides=2, padding='same',
                        use_bias=False, kernel_initializer=w_init)(x)
    x = BatchNormalization(gamma_initializer=g_init)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(ngf, kernel_size=4, strides=2, padding='same',
                        use_bias=False, kernel_initializer=w_init)(x)
    x = BatchNormalization(gamma_initializer=g_init)(x)

    x = Conv2DTranspose(n_channels, kernel_size=4, strides=2, padding='same',
                        use_bias=False, kernel_initializer=w_init)(x)
    x = Activation('tanh')(x)

    # create model graph
    model = Model(inputs=[z_inputs, text_inputs], outputs=x, name='Generator')

    print("\nGenerator ResNet")
    model.summary()
    return model


from keras.layers import RepeatVector, Reshape
def build_discriminator(image_input_shape=(64, 64, 3), text_input_shape=(1024,),
                        embedding_dim=128, ndf=64, activation='linear'):
    image_inputs = Input(shape=image_input_shape, name='image_input')
    text_inputs = Input(shape=text_input_shape, name='text_input')

    text_embedded = Dense(embedding_dim, input_shape=text_input_shape,
                          use_bias=True, kernel_initializer=w_init)(text_inputs)
    text_embedded = LeakyReLU(0.2)(text_embedded)
    text_embedded = RepeatVector(16)(text_embedded)
    text_embedded = Reshape((4, 4, embedding_dim))(text_embedded)

    x = Conv2D(ndf, kernel_size=4, strides=2, padding='same',
               kernel_initializer=w_init,
               input_shape=image_input_shape)(image_inputs)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(ndf*2, kernel_size=4, strides=2, padding='same',
               kernel_initializer=w_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(ndf*4, kernel_size=4, strides=2, padding='same',
               kernel_initializer=w_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(ndf*8, kernel_size=4, strides=2, padding='same',
               kernel_initializer=w_init)(x)
    x = BatchNormalization()(x)
    image_embedded = LeakyReLU(0.2)(x)

    x = Concatenate()([text_embedded, image_embedded])
    x = Conv2D(ndf*8, kernel_size=1, strides=1, padding='valid',
               kernel_initializer=w_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(1, kernel_size=4, strides=1, padding='valid',
               kernel_initializer=w_init)(x)
    x = Activation(activation)(x)

    print("\nDiscriminator")
    model = Model(inputs=[image_inputs, text_inputs], outputs=x,
                  name='Discriminator')
    model.summary()
    return model
