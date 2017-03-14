""" model.py """

from __future__ import division
from __future__ import print_function

if __name__ == '__main__':
    raise Exception('This file is not executable. Use birdsong.py')

import os

import numpy as np
import scipy.spatial.distance as sc_dist

import keras
from keras import backend as K

import gandlf

import h5py


def interpolate_latent_space(model, nb_points=60):
    """Interpolates between two random points in the latent vector space.

    The interpolation is done along a path with more or less constant magnitude.
    This is so that all the vectors along the contour could plausibly have been
    drawn from the normal distribution.

    Args:
        model: the gandlf model to use.
        nb_points: int, the number of points to interpolate.

    Returns:
        Numpy array with shape (nb_points, time, freq), where each slice is
            a spectrogram.
    """

    if nb_points % 2:
        raise ValueError('The number of points to interpolate must be '
                         'divisible by 2 - got %d.' % nb_points)

    # Does integer division (see `from __future__ import division` above).
    nb_points //= 2

    latent_size = model.generator.input_shape[1:]

    def _normalize(pt):
        return pt / np.linalg.norm(pt)

    def _to_radial(pt):
        norm = np.linalg.norm(pt)
        return norm, pt / norm

    def _from_radial(magnitude, direction):
        return magnitude * _normalize(direction)

    # Generates the latent vector.
    x = np.expand_dims(np.linspace(0, 1, nb_points), 1)
    d1, r1 = _to_radial(np.random.normal(size=latent_size))
    d2, r2 = _to_radial(np.random.normal(size=latent_size))

    latent_vecs = []
    dx, rx = np.copy(d1), np.copy(r1)

    # p1 -> p2
    dd = (d2 - d1) / nb_points
    rd = (r2 - r1) / nb_points
    for _ in range(nb_points):
        latent_vecs.append(_from_radial(dx, rx))
        dx += dd
        rx += rd

    # p2 -> p1
    dd = (d1 - d2) / nb_points
    rd = (r1 - r2) / nb_points
    for _ in range(nb_points):
        latent_vecs.append(_from_radial(dx, rx))
        dx += dd
        rx += rd

    # Stacks to get a single Numpy array.
    latent_vecs = np.stack(latent_vecs)

    # Samples from the model.
    samples = model.sample([latent_vecs])

    return samples


def get_discriminator_filters(cache='/tmp/birdsong.h5'):
    """Returns the discriminator filters to visualize."""

    if not os.path.exists(cache):
        raise ValueError('No weights at "%s" exist; train the model and '
                         'save the weights to plot the filters.' % cache)

    f = h5py.File(cache, mode='r')

    if 'discriminator_filters' not in f:
        raise ValueError('The cached weights should have a layer that '
                         'is named "discriminator_filters" to identify which '
                         'layer to visualize.')
    if 'discriminator_weights' not in f:
        raise ValueError('The cached weights should have a layer that '
                         'is named "discriminator_weights" to identify '
                         'the weights on each filter.')

    x = f['discriminator_filters']['discriminator_filters_W:0'].value
    w = f['discriminator_weights']['discriminator_weights_W:0'].value

    x = x.transpose(3, 0, 1, 2).squeeze()

    # Weights the filters.
    y = x * np.expand_dims(w, -1)

    return y


def get_generator_filters(time_length, cache='/tmp/birdsong.h5'):
    """Returns the generator filters to visualize."""

    if not os.path.exists(cache):
        raise ValueError('No weights at "%s" exist; train the model and '
                         'save the weights to plot the filters.' % cache)

    f = h5py.File(cache, mode='r')

    if 'generator_filters' not in f:
        raise ValueError('The cached weights should have a layer that '
                         'is named "generator_filters" to identify which '
                         'layer to visualize.')
    if 'generator_weights' not in f:
        raise ValueError('The cached weights should have a layer that '
                         'is named "generator_weights" to identify '
                         'the weights on each filter.')

    x = f['generator_filters']['generator_filters_W:0'].value
    w = f['generator_weights']['generator_weights_W:0'].value

    y = np.dot(w, x)
    y = y.reshape((y.shape[0], time_length, -1))

    return y


def build_generator(time_length, freq_length):
    """Builds the generator model."""

    latent = keras.layers.Input(shape=(100,))

    x = latent

    x = keras.layers.Dense(time_length,
            activation=None,
            name='generator_weights',
            init='glorot_normal')(x)
    x = keras.layers.Activation('tanh')(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Dense(time_length * freq_length,
            name='generator_filters',
            activation=None,
            # W_regularizer='l2',
            init='glorot_uniform')(x)
    x = keras.layers.Reshape((time_length, freq_length))(x)

    return keras.models.Model([latent], [x], name='generator')


def build_discriminator(time_length, freq_length):
    """Builds the discriminator model."""

    real_sound = keras.layers.Input(shape=(time_length, freq_length))

    x = real_sound

    x = keras.layers.Convolution1D(512, time_length,
            activation='tanh',
            # W_regularizer='l2',
            name='discriminator_filters',
            init='glorot_normal',
            border_mode='same')(x)
    # x = keras.layers.PReLU()(x)

    x = keras.layers.GlobalMaxPooling1D()(x)
    # x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(1,
            # W_regularizer='l2',
            name='discriminator_weights')(x)

    x = keras.layers.Activation('sigmoid')(x)

    return keras.models.Model([real_sound], [x], name='discriminator')


def train(X_data, nb_epoch=10, rebuild=False, cache='/tmp/birdsong.h5'):
    """Trains the model.

    Args:
        X_data: numpy array with shape (batch_size, time, freq), the input
            spectrograms.
        nb_epoch: int, number of training epochs.
        rebuild: bool, if set, rebuilds the model, otherwise uses the existing
            weights (if they exist).
        cache: str, where to cache weights.
    """

    _, time_length, freq_length = X_data.shape
    generator = build_generator(time_length, freq_length)
    discriminator = build_discriminator(time_length, freq_length)

    # Compile the model.
    loss = {'dis': 'binary_crossentropy', 'gen': 'binary_crossentropy'}
    # loss = {'gen_real': 'maximize', 'fake': 'minimize'}
    # optimizer = ['sgd', keras.optimizers.Adam(lr=1e-3)]
    optimizer = keras.optimizers.Nadam()
    # optimizer = [keras.optimizers.Adam(lr=1e-4),
    #         keras.optimizers.Adam(lr=1e-3)]
    model = gandlf.Model(generator=generator, discriminator=discriminator)

    # Loads existing weights.
    if not rebuild and os.path.exists(cache):
        model.load_weights(cache)

    # Trains the model.
    model.compile(loss=loss, optimizer=optimizer)
    targets = {'real_gen': 1, 'fake': 0}
    model.fit(['normal', X_data], targets, nb_epoch=nb_epoch, batch_size=32)

    # Saves the weights.
    model.save_weights(cache)
    print('Saved weights to "%s"' % cache)

    return model
