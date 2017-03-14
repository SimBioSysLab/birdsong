""" utils.py """

from __future__ import division
from __future__ import print_function

if __name__ == '__main__':
    raise Exception('This file is not executable. Use birdsong.py')

import os

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

import scipy.signal as scipy_signal
import scipy.io as sio
import scipy.io.wavfile as wavfile

# Default spectrogram parameters.
NFFT = 256
NOVERLAP = 128
DOWNSAMPLE = 0
CLIP_BOTTOM = 20  # Clip the bottom frequencies.


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

    else:  # Batch-wise.
        if x.shape[1] % 2:
            x = x[:, :-1, :]
        if x.shape[2] % 2:
            x = x[:, :, :-1]

        return (x[:, ::2, ::2] + x[:, ::2, 1::2] +
                x[:, 1::2, ::2] + x[:, 1::2, 1::2]) / 4


def plot_as_gif(x,
                interval=50,
                save_path='/tmp/generated.gif',
                normalize=False):
    """Plots data as a gif.

    Args:
        x: numpy array with shape (gif_length, time, freq).
        interval: int, the time between frames in milliseconds.
        save_path: str, where to save the resulting gif.
        normalize: bool, if set, normalize the spectrogram intensities.
    """

    # Gets the axis labels.
    flabels, tlabels = get_freq_time_labels(x.shape[1])
    extent = [tlabels[0] * 1000, tlabels[-1] * 1000,
              flabels[-1] / 1000, flabels[0] / 1000]

    if normalize:
        x = np.power(x, 0.45)

    # Plots the first sample.
    fig, ax = plt.subplots()
    im = plt.imshow(x[0].T,
            interpolation='none',
            aspect='auto',
            animated=True,
            vmin=0,
            vmax=1,
            extent=extent)

    # Fixes the dimensions.
    ax.invert_yaxis()
    ax.set_xlabel('Time (msec)')
    ax.set_ylabel('Freq (kHz)')

    def updatefig(i, *args):
        im.set_array(x[i].T)
        return im,

    anim = FuncAnimation(fig, updatefig,
            frames=np.arange(0, x.shape[0]),
            interval=interval)

    anim.save(save_path, dpi=80, writer='imagemagick')
    print('Saved gif to "%s".' % save_path)

    plt.show()


def plot_sample(x,
                title,
                width=3,
                height=2,
                shuffle=True,
                vmin=0,
                vmax=1,
                normalize=False):
    """Plots a sample of the data.

    Args:
        x: numpy array with shape (batch_size, time, freq).
        title: str, the title of the plot.
        width: int, the number of images wide.
        height: int, the number of images tall.
        vmin: float or None, the min of imshow.
        vmax: float or None, the max of imshow.
        shuffle: bool, if set, select randomly, otherwise select in order.
        normalize: bool, if set, increase the small values.
    """

    n = width * height

    # Gets the frequency and time labels.
    time_length = x.shape[1]
    flabels, tlabels = get_freq_time_labels(time_length)

    # Limits tlabels to the number of time steps in x.
    tlabels = tlabels[:x.shape[1]]

    if shuffle:
        idx = np.random.choice(np.arange(x.shape[0]), n)
        data = x[idx]
    else:
        data = x[:n]

    extent = [tlabels[0] * 1000, tlabels[-1] * 1000,
              flabels[-1] / 1000, flabels[0] / 1000]

    plt.figure(figsize=(width * 5, height * 5))
    for i in range(n):
        d = data[i].T

        if normalize:
            d = np.power(d, 0.45)

        ax = plt.subplot(height, width, i + 1)
        ax.imshow(d,
                interpolation='none',
                aspect='auto',
                vmin=vmin,
                vmax=vmax,
                extent=extent)
        ax.invert_yaxis()
        ax.set_xlabel('Time (msec)')
        ax.set_ylabel('Freq (kHz)')

    plt.suptitle(title)
    plt.show()


def get_spectrogram(signal, fs):
    """Gets spectrogram with the default parameters."""

    freq, time, data = scipy_signal.spectrogram(signal,
        fs=fs,
        # window=('gaussian', 0.1),
        nperseg=NFFT,
        noverlap=NOVERLAP,
        nfft=NFFT)

    # Clips the bottom part.
    data = data[CLIP_BOTTOM:]
    freq = freq[CLIP_BOTTOM:]
    time = time[CLIP_BOTTOM:]

    for i in range(DOWNSAMPLE):
        freq = downsample_arr(freq)
        time = downsample_arr(time)
        data = downsample_arr(data)

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
    fs, data = wavfile.read(fpath)  # fs usually ~250,000
    _, (freq, time) = get_spectrogram(data, fs)

    return freq, time[:time_length]


def get_all_spectrograms(time_length,
                         cache='/tmp/spectrograms.npy',
                         rebuild=False):
    """Returns a Numpy array with shape (batch_size, time_length, freq).

    Args:
        time_length: int, number of time steps per spectrogram. Each step is
            about 1/100 ms. Using ~300 seems to look good.
        cache: str, where to cache the array to avoid regenerating every time
            the method is called.
        rebuild: bool, if set, the dataset is rebuilt (useful if changes are
            made to the spectrogram parameters, for example).

    Returns:
        Numpy array with shape (batch_size, time_length, freq), the stacked
            spectrograms.
    """

    if not rebuild and os.path.exists(cache):
        all_sgrams = np.load(cache)
    else:
        all_sgrams = []
        fnames = os.listdir(WAV_SEGMENTS)

        def _check(sgram):
            """Makes sure the spectrogram is something we want to use."""

            if np.max(sgram) < 0.4:
                return False

            return True

        for fname_nb, fname in enumerate(fnames):
            if not fname.endswith('wav'):
                continue

            fpath = os.path.join(WAV_SEGMENTS, fname)
            fs, data = wavfile.read(fpath)
            sgram, _ = get_spectrogram(data, fs)  # (freq, time)

            # Scales the spectrogram to something reasonable.
            sgram *= 1e5

            # Gets the pixel dimensions of the spectrogram.
            nb_time = sgram.shape[1]

            # Adds spectrogram segments which satisfy the check from above.
            all_sgrams += [sgram[:, i - time_length:i]
                    for i in range(time_length, nb_time, time_length // 2)
                    if _check(sgram[:, i - time_length:i])]

            print('Processed %d / %d' % (fname_nb, len(fnames)), end='\r')

        # Concatenates all samples and puts in order (batch_size, time, freq).
        all_sgrams = np.stack(all_sgrams).transpose(0, 2, 1)
        np.save(cache, all_sgrams)

    print('%d spectrograms of length %d' % (len(all_sgrams), time_length))

    return all_sgrams
