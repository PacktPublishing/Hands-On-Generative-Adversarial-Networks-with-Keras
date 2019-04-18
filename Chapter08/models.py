from keras.layers import Conv2D, Activation
from keras.layers import Add, Lambda
from keras.initializers import RandomNormal

weight_init = RandomNormal(mean=0., stddev=0.02)


def resnet_block(input, n_blocks, n_filters, kernel_size=(1, 3)):
    output = input
    for i in range(n_blocks):
        output = Activation('relu')(output)
        output = Conv2D(filters=n_filters, kernel_size=kernel_size,
                        strides=1, padding='same',
                        kernel_initializer=weight_init)(output)
        output = Activation('relu')(output)
        output = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=1,
                        padding='same', kernel_initializer=weight_init)(output)

        output = Lambda(lambda x: x * 0.3)(output)

        # Residual Connection
        output = Add()([input, output])

    return output

from keras.layers import Input, Activation, Softmax
from keras.layers import Dense, Conv2D
from keras.layers.core import Reshape
from keras.models import Model


def build_resnet_generator(input_shape, n_filters, n_residual_blocks,
                           seq_len, vocabulary_size):
    inputs = Input(shape=input_shape)

    # Dense 1: 1 x seq_len x n_filters
    x = Dense(1 * seq_len * n_filters, input_shape=input_shape)(inputs)
    x = Reshape((1, seq_len, n_filters))(x)

    # ResNet blocks
    x = resnet_block(x, n_residual_blocks, n_filters)

    # Output layer
    x = Conv2D(filters=vocabulary_size, kernel_size=1, padding='same')(x)
    x = Softmax(axis=3)(x)

    # create model graph
    model = Model(inputs=inputs, outputs=x, name='Generator')

    print("\nGenerator ResNet")
    model.summary()
    return model


from keras.layers import Input, Flatten
from keras.layers import Dense, Conv2D
from keras.models import Model


def build_resnet_discriminator(input_shape, n_filters, n_residual_blocks,
                               kernel_size=(1, 1), stride=1):
    input = Input(shape=input_shape)
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=stride,
               padding='same')(input)
    x = resnet_block(x, n_residual_blocks, n_filters)
    x = Flatten()(x)
    x = Dense(1)(x)

    # create model graph
    model = Model(inputs=input, outputs=x, name='Discriminator')

    print("\nDiscriminator ResNet")
    model.summary()
    return model
