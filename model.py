""" model.py """

from __future__ import division
from __future__ import print_function

if __name__ == '__main__':
    raise Exception('This file is not executable. Use birdsong.py')

import os

import numpy as np

import keras
from keras import backend as K

import gandlf


def interpolate_latent_space(model, nb_points=60):
    """Interpolates between two random points in the latent vector space.

    Args:
        model: the gandlf model to use.
        nb_points: int, the number of points to interpolate.

    Returns:
        Numpy array with shape (nb_points, time, freq), where each slice is
            a spectrogram.
    """

    if nb_points % 3:
        raise ValueError('The number of points to interpolate must be '
                         'divisible by 3 - got %d.' % nb_points)

    # Does integer division (see `from __future__ import division` above).
    nb_points //= 3

    latent_size = model.generator.input_shape[1:]

    # Generates the latent vector.
    x = np.expand_dims(np.linspace(0, 1, nb_points), 1)
    p1 = np.random.normal(size=latent_size)
    p2 = np.random.normal(size=latent_size)
    p3 = np.random.normal(size=latent_size)

    latent_vecs = []
    x = np.copy(p1)

    # p1 -> p2
    d1 = (p2 - p1) / nb_points
    for _ in range(nb_points):
        latent_vecs.append(np.copy(x))
        x += d1

    # p2 -> p3
    d2 = (p3 - p2) / nb_points
    for _ in range(nb_points):
        latent_vecs.append(np.copy(x))
        x += d2

    # p3 -> p1
    d3 = (p1 - p3) / nb_points
    for _ in range(nb_points):
        latent_vecs.append(np.copy(x))
        x += d3

    # Stacks to get a single Numpy array.
    latent_vecs = np.stack(latent_vecs)

    # Samples from the model.
    samples = model.sample([latent_vecs])

    return samples


def build_generator(time_length, freq_length):
    """Builds the generator model."""

    latent = keras.layers.Input(shape=(100,))

    x = latent

    x = keras.layers.Dense(100, init='glorot_normal')(x)
    x = keras.layers.Activation('tanh')(x)

    x = keras.layers.Dense(time_length * freq_length, init='glorot_uniform')(x)
    x = keras.layers.LeakyReLU(0.01)(x)

    output = keras.layers.Reshape((time_length, freq_length))(x)

    return keras.models.Model([latent], [output], name='generator')


def build_discriminator(time_length, freq_length):
    """Builds the discriminator model."""

    real_sound = keras.layers.Input(shape=(time_length, freq_length))

    x = real_sound

    x = keras.layers.Reshape((time_length, freq_length, 1))(x)
    x = keras.layers.Convolution2D(64, 7, freq_length - 4,
            activation=None,
            border_mode='same')(x)

    x = keras.layers.GlobalMaxPooling2D()(x)

    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(128,
            activation='tanh')(x)

    output = keras.layers.Dense(1,
            activation='sigmoid')(x)

    return keras.models.Model([real_sound], [output], name='discriminator')


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
    # optimizer = keras.optimizers.Adam(lr=1e-3, clipnorm=1.)
    optimizer = ['sgd', keras.optimizers.Adam(lr=1e-3, clipnorm=1.)]
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
