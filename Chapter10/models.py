from keras.initializers import RandomNormal, Ones
weight_init = RandomNormal(mean=0., stddev=0.02)


from keras.layers import Input, Activation, Concatenate, Flatten
from keras.layers import Conv2D, Conv2DTranspose, Dense, UpSampling2D
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K
from keras.layers import Layer

class ScaleLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ScaleLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            # Create a trainable weight variable # for this layer.
            self.weights = self.add_weight(
                name='weights',
                shape=(input_shape[1], self.output_dim),
                initializer=Ones(),
                trainable=True)
            super(ScaleLayer, self).build(input_shape)
            # Be sure to call  this at the end

        def call(self, x):
            return x * self.weights

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.output_dim)


def build_segan_generator(noisy_input_shape, z_input_shape,
                          n_filters=[64, 128, 256, 512, 1024],
                          kernel_size=(1, 31), use_upsampling=False):
    noisy_input = Input(shape=noisy_input_shape)
    z_input = Input(shape=z_input_shape)

    # skip connections
    skip_connections = []

    # encode
    x = noisy_input
    for i in range(len(n_filters)):
        x = Conv2D(filters=n_filters[i], kernel_size=kernel_size,
                   strides=(1, 4), padding='same', use_bias=True,
                   kernel_initializer=weight_init)(x)
        x = PReLU()(x)
        skip_connections.append(ScaleLayer(n_filters[i])(x))

    # prepend single channel filter and remove the last filter size
    n_filters = [1] + n_filters[:-1]

    # update current x input
    x = z_input

    # decode
    for i in range(len(n_filters)-1, -1, -1):
        x = Concatenate(3)([x, skip_connections[i]])
        if use_upsampling:
            x = UpSampling2D(size=(1, 4))(x)
            x = Conv2D(filters=n_filters[i], kernel_size=kernel_size,
                       strides=(1, 1), padding='same',
                       kernel_initializer=weight_init, use_bias=True)(x)
        else:
            x = Conv2DTranspose(filters=n_filters[i], kernel_size=kernel_size,
                                strides=(1, 4), padding='same',
                                kernel_initializer=weight_init)(x)

        x = PReLU()(x) if i > 0 else Activation("tanh")(x)

    # create model graph
    model = Model(inputs=[noisy_input, z_input], outputs=x, name='Generator')

    print("\nGenerator")
    model.summary()
    return model


"""
z_input_shape = (1, 8, 1024)
audio_input_shape = (1, 16384, 1)
g = build_segan_generator(z_input_shape, audio_input_shape)
"""

from keras.layers import Input, Reshape
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def build_segan_discriminator(noisy_input_shape, clean_input_shape,
                              n_filters=[64, 128, 256, 512, 1024],
                              kernel_size=(1, 31)):

    clean_input = Input(shape=clean_input_shape)
    noisy_input = Input(shape=noisy_input_shape)
    x = Concatenate(-1)([clean_input, noisy_input])

    # convolution layers
    for i in range(len(n_filters)):
        x = Conv2D(filters=n_filters[i], kernel_size=kernel_size,
                   strides=(1, 4), padding='same', use_bias=True,
                   kernel_initializer=weight_init)(x)
        x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
        x = PReLU()(x)

    x = Reshape((16384, ))(x)

    # dense layers
    x = Dense(256, activation=None, use_bias=True)(x)
    x = PReLU()(x)
    x = Dense(128, activation=None, use_bias=True)(x)
    x = PReLU()(x)
    x = Dense(1, activation=None, use_bias=True)(x)

    # create model graph
    model = Model(inputs=[noisy_input, clean_input], outputs=x, name='Discriminator')

    print("\nDiscriminator")
    model.summary()
    return model

"""
clean_input_shape = (1, 16384, 1)
noisy_input_shape = (1, 16384, 1)
build_segan_discriminator(clean_input_shape, noisy_input_shape)
"""
