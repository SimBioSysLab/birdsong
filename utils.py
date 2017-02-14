""" utils.py """

from __future__ import division
from __future__ import print_function

import os

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

import matplotlib.pyplot as plt

import numpy as np

import scipy.signal as scipy_signal
import scipy.io as sio
import scipy.io.wavfile as wavfile

# Default spectrogram parameters.
NFFT = 256
NOVERLAP = 128


def halve_image(x):
    """Takes average between four points to get one point."""

    if len(x.shape) not in (2, 3):
        raise ValueError('The Numpy array should either have dimensions '
                         '(time, freq) or (batch_size, time, freq). Got '
                         '%d dimensions.' % len(x.shape))

    if len(x.shape) == 3:
        if x.shape[1] % 2:
            x = x[:, :-1, :]
        if x.shape[2] % 2:
            x = x[:, :, :-1]

        return (x[:, ::2, ::2] + x[:, ::2, 1::2] +
                x[:, 1::2, ::2] + x[:, 1::2, 1::2]) / 4

    else:
        if x.shape[0] % 2:
            x = x[:-1, :]
        if x.shape[1] % 2:
            x = x[:, :-1]

        return (x[::2, ::2] + x[::2, 1::2] +
                x[1::2, ::2] + x[1::2, 1::2]) / 4


def plot_sample(x, width=5, height=5, shuffle=True, downsample=0):
    """Plots a sample of the data.

    Args:
        x: numpy array with shape (batch_size, time, freq).
        width: int, the number of images wide.
        height: int, the number of images tall.
        shuffle: bool, if set, select randomly, otherwise select in order.
        downsample: int, number of times to downsample image.
    """

    n = width * height

    if shuffle:
        idx = np.random.choice(np.arange(x.shape[0]), n)
        data = x[idx]
    else:
        data = x[:n]

    plt.figure()
    for i in range(n):
        d = data[i].T
        for _ in range(downsample):
            d = halve_image(d)

        plt.subplot(height, width, i + 1)
        plt.imshow(d)
        plt.gca().invert_yaxis()
    plt.show()


def get_spectrogram(signal, fs):
    """Gets spectrogram with the default parameters."""

    freq, time, data = scipy_signal.spectrogram(signal,
        fs=fs,
        # window=('gaussian', 0.1),
        nperseg=NFFT,
        noverlap=NOVERLAP,
        nfft=NFFT)

    return data, (freq, time)

def get_data_path():
    """Gets the path to the data as a string, safely."""

    if 'DATA_PATH' not in os.environ:
        os.environ['DATA_PATH'] = 'data'

    DATA_PATH = os.environ['DATA_PATH']

    if not os.path.isdir(DATA_PATH):
        raise RuntimeError('No data directory found at "%s". Make sure '
                           'the DATA_PATH environment variable is set to '
                           'point at the correct directory.' % DATA_PATH)

    return DATA_PATH


def get_directory(directory):
    """Gets a subdirectory of the datapath, safely."""

    d = os.path.join(get_data_path(), directory)

    if not os.path.isdir(d):
        raise RuntimeError('No directory found at "%s".' % d)

    return d


def get_all_spectrograms(time_length,
                         cache='/tmp/spectrograms.npy',
                         rebuild=False):
    """Yields spectrograms as Numpy arrays, caching them in `cache`."""

    if not rebuild and os.path.exists(cache):
        all_sgrams = np.load(cache)
    else:
        all_sgrams = []

        WAV_SEGMENTS = get_directory('USV_Segments/WAV')
        fnames = os.listdir(WAV_SEGMENTS)

        for fname_nb, fname in enumerate(fnames):
            if not fname.endswith('wav'):
                continue

            fpath = os.path.join(WAV_SEGMENTS, fname)
            fs, data = wavfile.read(fpath)
            sgram, _ = get_spectrogram(data, fs)  # (freq, time)

            # Gets the pixel dimensions of the spectrogram.
            nb_freq, nb_time = sgram.shape
            all_sgrams += [sgram[:, i - time_length:i]
                           for i in range(time_length, nb_time, time_length)]

            print('Processed %d / %d' % (fname_nb, len(fnames)), end='\r')

        all_sgrams = np.stack(all_sgrams).transpose(0, 2, 1)
        np.save(cache, all_sgrams)

    print('%d spectrograms of length %d' % (len(all_sgrams), time_length))

    return all_sgrams

if __name__ == '__main__':
    raise Exception('This file is not executable.')
