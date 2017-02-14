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


def downsample_arr(x):
    """Takes average between four points to get one point."""

    ndim = len(x.shape)

    if ndim == 1:
        if x.shape[0] % 2:
            x = x[:-1]

        return (x[::2] + x[1::2]) / 2

    elif ndim == 2:
        if x.shape[0] % 2:
            x = x[:-1, :]
        if x.shape[1] % 2:
            x = x[:, :-1]

        return (x[::2, ::2] + x[::2, 1::2] +
            x[1::2, ::2] + x[1::2, 1::2]) / 4

    else:
        if x.shape[1] % 2:
            x = x[:, :-1, :]
        if x.shape[2] % 2:
            x = x[:, :, :-1]

        return (x[:, ::2, ::2] + x[:, ::2, 1::2] +
                x[:, 1::2, ::2] + x[:, 1::2, 1::2]) / 4


def plot_sample(x,
                width=3,
                height=2,
                shuffle=True,
                downsample=0,
                normalize=True):
    """Plots a sample of the data.

    Args:
        x: numpy array with shape (batch_size, time, freq).
        width: int, the number of images wide.
        height: int, the number of images tall.
        shuffle: bool, if set, select randomly, otherwise select in order.
        downsample: int, number of times to downsample image.
        normalize: bool, if set, increase the small values.
    """

    n = width * height

    # Gets the frequency and time labels.
    time_length = x.shape[1]
    flabels, tlabels = get_freq_time_labels(time_length)

    for _ in range(downsample):
        flabels = downsample_arr(flabels)
        tlabels = downsample_arr(tlabels)

    # Converts to more readable version.
    flabels = ['%.2f' % (i / 1000) for i in flabels]
    tlabels = ['%.2f' % (i * 1000 - 1) for i in tlabels]

    if shuffle:
        idx = np.random.choice(np.arange(x.shape[0]), n)
        data = x[idx]
    else:
        data = x[:n]

    plt.figure()
    for i in range(n):
        d = data[i].T
        for _ in range(downsample):
            d = downsample_arr(d)
        if normalize:
            d = np.power(d, 0.45) * 1e3

        ax = plt.subplot(height, width, i + 1)
        ax.imshow(d, vmin=0, vmax=1)
        ax.invert_yaxis()
        ax.set_xticklabels(tlabels)
        ax.set_yticklabels(flabels)
        ax.set_xlabel('Time (msec)')
        ax.set_ylabel('Freq (kHz)')
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

# Some important directories.
WAV_SEGMENTS = get_directory('USV_Segments/WAV')


def get_freq_time_labels(time_length):
    """Gets the frequency and time labels for the default settings."""

    fname = os.listdir(WAV_SEGMENTS)[0]
    fpath = os.path.join(WAV_SEGMENTS, fname)
    fs, data = wavfile.read(fpath)
    _, (freq, time) = get_spectrogram(data, fs)

    return freq, time[:time_length]


def get_all_spectrograms(time_length,
                         cache='/tmp/spectrograms.npy',
                         rebuild=False):
    """Yields spectrograms as Numpy arrays, caching them in `cache`."""

    if not rebuild and os.path.exists(cache):
        all_sgrams = np.load(cache)
    else:
        all_sgrams = []
        fnames = os.listdir(WAV_SEGMENTS)

        def _check(sgram):
            """Makes sure the spectrogram is something we want to use."""

            if np.max(sgram) < 1e-6:
                return False

            return True

        for fname_nb, fname in enumerate(fnames):
            if not fname.endswith('wav'):
                continue

            fpath = os.path.join(WAV_SEGMENTS, fname)
            fs, data = wavfile.read(fpath)
            sgram, _ = get_spectrogram(data, fs)  # (freq, time

            # Gets the pixel dimensions of the spectrogram.
            nb_time = sgram.shape[1]

            # Adds spectrogram segments which satisfy the check from above.
            all_sgrams += [sgram[:, i - time_length:i]
                           for i in range(time_length, nb_time, time_length)
                           if _check(sgram[:, i - time_length:i])]

            print('Processed %d / %d' % (fname_nb, len(fnames)), end='\r')

        # Concatenates all samples and puts in order (batch_size, time, freq).
        all_sgrams = np.stack(all_sgrams).transpose(0, 2, 1)
        np.save(cache, all_sgrams)

    print('%d spectrograms of length %d' % (len(all_sgrams), time_length))

    return all_sgrams

if __name__ == '__main__':
    raise Exception('This file is not executable.')
