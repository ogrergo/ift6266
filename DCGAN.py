import datetime
import os

from PIL import Image
from aniso8601 import date
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Convolution2D
from keras.layers.core import Dense, Reshape, Activation, Dropout, Flatten
from keras import backend as K

import numpy as np

import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

from dataset import get_dataset, get_center, save_image


def generator(input_size=100):
    """
    Return a model for the generator.
    This model take a random noise in parameter and generate
    :return:
    """
    features = [(256, 5, 2), (256, 5, 2), (128, 5, 2)]

    input = Input(shape=(input_size,))

    _out = Dense(units=16384)(input)

    _out = Reshape((1024, 4, 4))(_out)

    _out = BatchNormalization()(_out)

    for f, kernel, strides in features:
        _out = Conv2DTranspose(filters=f,
                               kernel_size=kernel,
                               strides=strides,
                               padding="same",
                               data_format="channels_first")(_out)
        _out = BatchNormalization()(_out)
        _out = Activation('relu')(_out)


    _out = Conv2DTranspose(filters=3,
                           kernel_size=5,
                           strides=2,
                           padding="same",
                           activation='tanh',
                           data_format="channels_first")(_out)

    model = Model(input, _out)
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    return model


def discriminator(dropout_rate=0.5):

    features = [(64, 5, 2), (128, 5, 2), (256, 5, 2),(512, 5, 2)]

    input = Input(shape=(3, 64, 64))

    _out = input

    for f, kernel, strides in features:
        _out = Convolution2D(filters=f,
                             kernel_size=kernel,
                             padding='same',
                             strides=strides,
                             data_format="channels_first")(_out)

        _out = BatchNormalization()(_out)
        _out = LeakyReLU(alpha=0.2)(_out)

        _out = Dropout(rate=dropout_rate)(_out)

    _out = Flatten()(_out)

    # _out = Dense(1000)(_out)
    # _out = BatchNormalization()(_out)

    # _out = LeakyReLU()(_out)
    #
    # _out = Dropout(rate=dropout_rate)(_out)

    _out = Dense(1, activation='sigmoid')(_out)


    from keras.losses import categorical_crossentropy
    def discrimator_loss(y_true, y_pred):
        # y_true is the label (generated or dataset)
        # y_true == 0 if generated
        # - 0.5 * K.equal(y_true, 0) * (1 - K.log(y_pred) K.not_equal(y_true, 0) *


        return -0.5 * K.mean(y_true * K.log(y_pred) +
                            (1 - y_true) * K.log(1 - y_pred), axis=-1)
        # return - 0.5 * K.mean(K.log(y_pred), axis=0)

    model = Model(input, _out)


    model.compile(optimizer=Adam(lr=.0002, beta_1=.5),
                  loss=discrimator_loss)
    model.summary()
    return model

noise_size = 100

gen = generator(noise_size)
dis = discriminator()

for l in dis.layers:
    l.trainable = False

_input = Input(shape=(noise_size,))
_out = gen(_input)
_out = dis(_out)


def gan_loss(y_true, y_pred):
    return -0.5 * K.mean(K.log(y_pred), axis=-1)

GAN = Model(_input, _out)
GAN.compile(optimizer=Adam(lr=.0002, beta_1=.5),
            loss=gan_loss)

GAN.summary()

for l in dis.layers:
    l.trainable = True

save_index = 0
save_folder = str(datetime.datetime.now())

def test_generator():

    r = np.random.normal(size=(100, noise_size))

    res = gen.predict_on_batch(r)

    res = (res.transpose(0, 2,3,1) * 127. + 127.).astype('uint8')

    size = 64
    rows = 10
    img = np.zeros(shape=(size * 10, size * rows, 3), dtype='uint8')
    index = lambda i: ((i % 10) * size, (i // 10) * size)

    for i in range(100):
        x, y = index(i)
        img[x:(size + x), y:(size + y), :] = res[i]


    Image.fromarray(img).show()

    global save_index

    save_image(os.path.join("generator", save_folder), "epoch_%d" % save_index, img)
    save_index += 1

def plot_loss(losses):
    plt.figure(figsize=(10, 8))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.show()


def train(dataset, GAN, discriminator, generator):
    batch_size = 32

    losses = {'d': [], 'g': []}

    def batchs_discrimator():
        np.random.shuffle(dataset)
        b = 0
        g = False

        while True:
            if g:
                t = (generator.predict_on_batch(np.random.normal(loc=0, scale=1, size=(batch_size, noise_size))),
                     np.zeros(shape=(batch_size, 1), dtype='float32'))
            else:
                if b + batch_size > dataset.shape[0]:
                    b = 0

                t = (dataset[b:b + batch_size, :, :, :],
                     np.random.normal(loc=.8, scale=.1, size=(batch_size, 1)))

                b += batch_size

            yield t

            g = not g


    def batchs_GAN():
        while True:
            yield (np.random.normal(loc=0, scale=1, size=(batch_size, noise_size)),
                   np.ones(shape=(batch_size, 1), dtype='float32'))

    state = 'd'

    steps = dataset.shape[0] // batch_size
    epoch = 1

    for i in range(10):
        if state == 'd':
            gen = batchs_discrimator()
            model = discriminator
        else:
            gen = batchs_GAN()
            model = GAN

        print("Training %s"%state)

        history = model.fit_generator(gen, steps_per_epoch=steps, epochs=epoch)
        epoch = 1

        losses[state].append(history.history['loss'])

        if state == 'g':
            test_generator()

        state = 'g' if state == 'd' else 'd'
    print(losses)
    plot_loss(losses)


test_generator()

data = get_dataset(100, 0, 0, min=-1., max=1.)[0]

train(data, GAN, dis, gen)
test_generator()
plot_loss({
    'd': [1,2,3,4,5,1,0],
    'g': [3,2,7,5,3]
})