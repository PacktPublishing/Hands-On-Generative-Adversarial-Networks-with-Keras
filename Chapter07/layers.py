# adapted from https://github.com/MSC-BUAA/Keras-progressive_growing_of_gans

import keras.backend as K
from keras.layers import Layer
from keras.layers.merge import _Merge
from keras import initializers
from keras.layers import Dense, Conv2D, LeakyReLU, Reshape
import tensorflow as tf
import numpy as np

w_init = initializers.random_normal(0, 1)
activation = LeakyReLU(alpha=0.2)


class Bias(Layer):
    def __init__(self, initializer='zeros', **kwargs):
        super(Bias, self).__init__(**kwargs)
        self.initializer = initializers.get(initializer)

    def build(self, input_shape):
        self.bias = self.add_weight(
            name='{}_bias'.format(self.name), shape=(input_shape[-1], ),
            initializer=self.initializer, trainable=True)

    def call(self, x):
        return K.bias_add(x, self.bias, data_format="channels_last")


def DenseBlock(x, size, gain=np.sqrt(2), use_pixelnorm=False,
               use_activation=True, reshape=None):
    dense = Dense(size, activation=None, use_bias=False,
                  kernel_initializer=w_init)

    n_in_channels = K.int_shape(x)[-1]
    x = dense(x)
    x = WeightScalingLayer((n_in_channels, size), gain=gain)(x)

    if reshape is not None:
        x = Reshape(reshape)(x)

    x = Bias()(x)

    if use_activation:
        x = activation(x)

    if use_pixelnorm:
        x = PixelNormLayer()(x)

    return x


def ConvBlock(x, n_filters, filter_size, padding, use_pixelnorm):
    conv = Conv2D(n_filters, filter_size, padding=padding, activation=None,
                  kernel_initializer=w_init, use_bias=False)
    n_in_channels = K.int_shape(x)[-1]
    x = conv(x)
    x = WeightScalingLayer(
        (filter_size, filter_size, n_in_channels, n_filters))(x)
    x = Bias()(x)
    x = activation(x)

    if use_pixelnorm:
        x = PixelNormLayer()(x)

    return x


def Conv1x1(x, n_channels, gain=np.sqrt(2), use_activation=True):
    conv = Conv2D(n_channels, 1, padding='same', activation=None,
                  use_bias=False, kernel_initializer=w_init)
    n_in_channels = K.int_shape(x)[-1]
    x = conv(x)
    x = WeightScalingLayer((1, 1, n_in_channels, n_channels), gain=gain)(x)
    x = Bias()(x)

    if use_activation:
        x = activation(x)
    return x


class ResizeLayer(Layer):
    def __init__(self, input_dims, output_dims, **kwargs):
        self.input_dims = input_dims
        self.output_dims = output_dims
        super(ResizeLayer, self).__init__(**kwargs)

    def call(self, v, **kwargs):
        assert (len(self.input_dims) == len(self.output_dims) and
                self.input_dims[0] == self.output_dims[0])

        # possibly shrink spatial axis by pooling elements
        if len(self.input_dims) == 4 and (self.input_dims[1] > self.output_dims[1] or self.input_dims[2] > self.output_dims[2]):
            assert (self.input_dims[1] % self.output_dims[1] == 0 and
                    self.input_dims[2] % self.output_dims[2] == 0)

            pool_sizes = (self.input_dims[1] / self.output_dims[1],
                          self.input_dims[2] / self.output_dims[2])
            strides = pool_sizes
            v = K.pool2d(
                v, pool_size=pool_sizes, strides=strides,
                padding='same', data_format='channels_last', pool_mode='avg')

        # possibly extend spatial axis by repeating elements
        for i in range(1, len(self.input_dims) - 1):
            if self.input_dims[i] < self.output_dims[i]:
                assert self.output_dims[i] % self.input_dims[i] == 0
                v = K.repeat_elements(
                    v, rep=int(self.output_dims[i] / self.input_dims[i]),
                    axis=i)

        return v

    def compute_output_shape(self, input_shape):
        return self.output_dims


class WeightScalingLayer(Layer):
    def __init__(self, shape, gain=np.sqrt(2), **kwargs):
        super(WeightScalingLayer, self).__init__(**kwargs)
        fan_in = np.prod(shape[:-1])
        std = gain / np.sqrt(fan_in)

        # add the scale term as a non-trainable parameter
        self.wscale = self.add_weight(name='wscale', shape=std.shape,
                                      trainable=False, initializer='zeros')
        K.set_value(self.wscale, std)

    def call(self, input, **kwargs):
        # here scaling the layer outputs is equivalent to scaling the weights
        return input * self.wscale

    def compute_output_shape(self, input_shape):
        return input_shape


class UpsamplingLayer(Layer):
    def __init__(self, factor=2, **kwargs):
        super(UpsamplingLayer, self).__init__(**kwargs)
        self.factor = factor

    def call(self, x, **kwargs):
        n, h, w, c = x.shape
        x = tf.reshape(x, [-1, h, 1, w, 1, c])
        x = tf.tile(x, [1, 1, self.factor, 1, self.factor, 1])
        x = tf.reshape(x, [-1, h * self.factor, w * self.factor, c])
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*self.factor,
                input_shape[2]*self.factor, input_shape[3])


class DownsamplingLayer(Layer):
    def __init__(self, factor=2, **kwargs):
        super(DownsamplingLayer, self).__init__(**kwargs)
        self.factor = factor

    def call(self, x, **kwargs):
        return K.pool2d(x, pool_size=(self.factor, self.factor),
                        strides=(self.factor, self.factor), padding='valid',
                        data_format='channels_last', pool_mode='avg')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]//self.factor,
                input_shape[2]//self.factor, input_shape[3])


class BlockSelectionLayer(Layer):
    def __init__(self, cur_block, first_incoming_block=0, ref_idx=0,
                 min_block=None, max_block=None):
        super(BlockSelectionLayer, self).__init__()
        self.cur_block = cur_block
        self.first_incoming_block = first_incoming_block
        self.ref_idx = ref_idx
        self.min_block = min_block
        self.max_block = max_block

    def call(self, inputs):
        self.input_shapes = [K.int_shape(input) for input in inputs]
        v = [ResizeLayer(K.int_shape(input),
             self.input_shapes[self.ref_idx])(input) for input in inputs]

        if self.min_block is not None:
            min_from_first = self.min_block - self.first_incoming_block
            min_from_first = int(np.floor(min_from_first))
            lo = np.clip(min_from_first, 0, len(v) - 1)
        else:
            lo = 0

        if self.max_block is not None:
            max_from_first = self.max_block - self.first_incoming_block
            max_from_first = int(np.ceil(max_from_first))
            hi = np.clip(max_from_first, lo, len(v) - 1)
        else:
            hi = len(v) - 1

        t = self.cur_block - self.first_incoming_block
        r = v[hi]
        for i in range(hi-1, lo-1, -1):  # i = hi-1, hi-2, ..., lo
            r = K.switch(K.less(t, i+1), v[i] * ((i+1)-t) + v[i+1] * (t-i), r)

        if lo < hi:
            r = K.switch(K.less_equal(t, lo), v[lo], r)

        return r

    def compute_output_shape(self, input_shape):
        return self.input_shapes[self.ref_idx]


class PixelNormLayer(Layer):
    def __init__(self, **kwargs):
        super(PixelNormLayer, self).__init__(**kwargs)

    def call(self, x, eps=1e-8, **kwargs):
        return x * tf.rsqrt(K.mean(K.square(x), axis=-1, keepdims=True) + eps)

    def compute_output_shape(self, input_shape):
        return input_shape


class MinibatchStatConcatLayer(Layer):
    def __init__(self, group_size=1, **kwargs):
        super(MinibatchStatConcatLayer, self).__init__(**kwargs)
        self.group_size = int(group_size)

    def call(self, x):
        # standard deviation over minibatch for each feature and location
        y = K.sqrt(K.mean(K.square(x - K.mean(x, axis=0, keepdims=True)),
                          axis=0, keepdims=True) + 1e-8)

        # compute the mean standard deviation
        y = K.mean(y, keepdims=True)

        # repeat to match n, h and w and have 1 feature map
        x_shape = K.shape(x)
        y = K.tile(y, [x_shape[0], x_shape[1], x_shape[2], 1])

        return K.concatenate([x, y], axis=-1)

    def compute_output_shape(self, input_shape):
        # output shape is input shape plus one feature map
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)


class Subtract(_Merge):
    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = output - inputs[i]
        return output


class GradNorm(Layer):
    def __init__(self, **kwargs):
        super(GradNorm, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(GradNorm, self).build(input_shapes)

    def call(self, inputs):
        target, wrt = inputs
        grads = K.gradients(target, wrt)
        assert len(grads) == 1
        grad = grads[0]
        return K.sqrt(
            K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)
