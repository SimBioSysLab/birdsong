# Images

This directory will keep a general log of the network configurations that produced the various types of images.

## Spectrograms

For all the spectrograms (as per Sihao's parameter choices):

 - NFFT: 256
 - NOVERLAP: 128

Downsampling is done by taking a local average of every four adjacent pixels. In Numpy:

````python
return (x[::2, ::2] + x[::2, 1::2] +
        x[1::2, ::2] + x[1::2, 1::2]) / 4
````

First, the spectrogram of the whole song is generated. Then the user specifies how many time bins to break it up into, and it is split into patches of that size (no overlap between patches - maybe there should be?). Then another step is to only use patches that have a maximum amplitude above some threshold.

# Experiments

## 1

Downsampled twice (image dimensions were 50 time bins by 32 frequency bins).

### Generator

Two Dense layers. 100-dimensional latent vector, 100-dimensional "chooser" layer (chooses which of the output filters to use), then 100 filters with size `(time_length, freq_length)`.

````python
latent = keras.layers.Input(shape=(100,))

x = latent

x = keras.layers.Dense(100, init='glorot_normal')(x)
x = keras.layers.Activation('tanh')(x)

x = keras.layers.Dense(time_length * freq_length, init='glorot_uniform')(x)
x = keras.layers.LeakyReLU(0.01)(x)

output = keras.layers.Reshape((time_length, freq_length))(x)

return keras.models.Model([latent], [output], name='generator')
````

### Discriminator

Convolutional part: Looks for filters with total invariance with respect to time, but frequency invariance with +/- 2 frequency bins. Global max pooling just checks to see if any of those filters are active in the input image. Then there are two dense layers that decide how to combine those filters.

````python
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
````

## 2

Same data and generator as 1.

### Discriminator

Only one output Dense layer.

````python
real_sound = keras.layers.Input(shape=(time_length, freq_length))

x = real_sound

x = keras.layers.Reshape((time_length, freq_length, 1))(x)
x = keras.layers.Convolution2D(64, 7, freq_length - 4,
        activation=None,
        border_mode='same')(x)

x = keras.layers.GlobalMaxPooling2D()(x)

output = keras.layers.Dense(1,
        activation='sigmoid')(x)

return keras.models.Model([real_sound], [output], name='discriminator')
````

## 3

The generator and discriminator were the same as the one used in 2. I experimented with L2 normalization in the generator, but all it seemed to do was to make the outputs non-multimodal. The spectrograms were not downsampled at all for this version.

