from functools import partial
import numpy as np
import tensorflow as tf
from keras.layers import Layer, InputSpec, Reshape
from keras.layers import Input, Add, Concatenate, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Activation
from keras.models import Model
import keras.backend as K


class InstanceWiseAveragePooling(Layer):
    def __init__(self):
        super(InstanceWiseAveragePooling, self).__init__()

    def call(self, v):
        n_classes = 3
        feats, inst = v[0], v[1]
        classes, _ = tf.unique(tf.reshape(inst, [-1]))
        for c in range(n_classes):
            mask = K.equal(K.cast(inst, np.int32), c)
            mask = K.tile(mask, (1, 1, 1, 3))
            feats_mean = K.mean(tf.boolean_mask(feats, mask))
            mask = K.cast(mask, float)
            feats = (1 - mask) * feats
            feats = feats + mask * feats_mean

        return feats


class ReflectionPadding2D(Layer):
    """https://stackoverflow.com/questions/50677544/reflection-padding-conv2d"""
    def __init__(self, padding=(1, 1), **kwargs):
        if type(padding) == int:
            padding = (padding, padding)
        self.padding = padding
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')


def resnet_block(input, n_filters):
    x = input
    x = ReflectionPadding2D(padding=1)(x)
    x = Conv2D(n_filters, kernel_size=3, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ReflectionPadding2D(padding=1)(x)
    x = Conv2D(n_filters, kernel_size=3, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)

    # Residual Connection
    x = Add()([input, x])
    return x


def build_global_generator(segmap_shape=(2048, 1024, 1),
                           edges_shape=(2048, 1024, 1),
                           feats_shape=(2048, 1024, 3), ngf=64,
                           n_downsampling=3, n_resnet_blocks=9, n_channels=3):
    segmaps = Input(shape=segmap_shape)
    edges = Input(shape=edges_shape)
    feats = Input(shape=feats_shape)

    inputs = Concatenate(axis=-1)([segmaps, edges, feats])

    # x = AveragePooling2D(3, strides=2, padding='same')(inputs)
    x = ReflectionPadding2D(3)(inputs)
    x = Conv2D(ngf, kernel_size=7, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # downsample
    for i in range(n_downsampling):
        mult = 2**i
        x = Conv2D(ngf * mult * 2, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # residual blocks
    mult = 2**n_downsampling
    for i in range(n_resnet_blocks):
        x = resnet_block(x, ngf*mult)

    # upsample
    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=3,
                            strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # residual connection for local generator
    residual = x

    # final convolution
    x = ReflectionPadding2D(3)(x)
    x = Conv2D(n_channels, kernel_size=7, strides=1, padding='valid')(x)
    x = Activation('tanh')(x)

    model = Model(inputs=[segmaps, edges, feats], outputs=[x, residual],
                  name='GlobalGenerator')

    print("\nGlobal Generator")
    model.summary()
    return model


def build_local_enhancer(global_generator, input_shape=(2048, 1024, 3), ngf=32,
                         n_blocks=3, n_local_enhancers=1):

    inputs = Input(shape=input_shape)
    n_channels = input_shape[-1]
    _, x_global = global_generator(inputs)

    ###################
    # local enhancers #
    ###################
    x = inputs
    for n in range(1, n_local_enhancers+1):
        ngf_global = ngf * (2**(n_local_enhancers-n))
        # downsample
        x = ReflectionPadding2D(3)(x)
        x = Conv2D(ngf_global, kernel_size=7, strides=1, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(ngf_global*2, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # residual connection from global generator
        x = Add()([x_global, x])

        # residual blocks
        for i in range(n_blocks):
            x = resnet_block(x, ngf_global * 2)

        # upsample
        x = Conv2DTranspose(filters=ngf_global, kernel_size=3,
                            strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # final convolution
    x = ReflectionPadding2D(3)(x)
    x = Conv2D(n_channels, kernel_size=7, strides=1, padding='valid')(x)
    x = Activation('tanh')(x)

    # create model graph
    model = Model(inputs=inputs, outputs=x, name='Local Enhancer')

    print("\nLocal Enhancer")
    model.summary()
    return model


def build_discriminator(input_shape_a=(2048, 1024, 1),
                        input_shape_b=(2048, 1024, 3),
                        input_shape_c=(2048, 1024, 3),
                        ndf=64, n_layers=3,
                        kernel_size=4, strides=2, activation='linear',
                        n_downsampling=1, name='Discriminator'):
    input_a = Input(shape=input_shape_a)
    input_b = Input(shape=input_shape_b)
    input_c = Input(shape=input_shape_c)

    features = []
    x = Concatenate(axis=-1)([input_a, input_b, input_c])
    for i in range(n_downsampling):
        x = AveragePooling2D(3, strides=2, padding='same')(x)

    x = Conv2D(ndf, kernel_size=kernel_size, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    features.append(x)

    nf = ndf
    for i in range(1, n_layers):
        nf = min(ndf * 2, 512)
        x = Conv2D(nf, kernel_size=kernel_size, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        features.append(x)

    nf = min(nf * 2, 512)
    x = Conv2D(nf, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    features.append(x)

    x = Conv2D(1, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = Activation(activation)(x)

    # create model graph
    model = Model(inputs=[input_a, input_b, input_c], outputs=[x] + features, name=name)
    print("\nDiscriminator")
    model.summary()
    return model


def build_encoder(img_shape=(2048, 1024, 3), instance_shape=(2048, 1024, 1),
                  n_out_channels=3, ngf=32,
                  n_downsampling=4):
    img = Input(shape=img_shape)
    inst = Input(shape=instance_shape)

    inputs = Concatenate(axis=-1)([img, inst])

    x = ReflectionPadding2D(3)(inputs)
    x = Conv2D(ngf, kernel_size=7, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # downsample
    for i in range(n_downsampling):
        mult = 2**i
        x = Conv2D(ngf * mult * 2, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # upsample
    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=3,
                            strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # final convolution
    x = ReflectionPadding2D(3)(x)
    x = Conv2D(n_out_channels, kernel_size=7, strides=1, padding='valid')(x)
    x = Activation('tanh')(x)

    # x = InstanceWiseAveragePooling()([x, K.cast(inst, np.int32)])

    # create model graph
    model = Model(inputs=[img, inst], outputs=x, name='Encoder')

    print("\nEncoder")
    model.summary()
    return model


"""
global_generator = build_global_generator()
generator = build_local_enhancer(global_generator)
discriminator_0 = build_discriminator(n_downsampling=0)
discriminator_1 = build_discriminator(n_downsampling=1)
discriminator_2 = build_discriminator(n_downsampling=2)
e = build_encoder()
"""
