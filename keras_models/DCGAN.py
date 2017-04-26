import datetime
import os
from collections import OrderedDict
from sys import stdout
from utils import *
from PIL import Image
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Convolution2D
from keras.layers.core import Dense, Reshape, Activation, Dropout, Flatten
from keras import backend as K

import numpy as np

import matplotlib.pyplot as plt
from keras.layers.merge import Concatenate, concatenate
from keras.layers.normalization import BatchNormalization
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
import progressbar as pb
import theano.tensor as T
from dataset import get_dataset, get_center, save_image


def make_generator(input_size=100):
    """
    Return a model for the generator.
    This model take a random noise in parameter and generate
    :return:
    """

    features = [(512, 5, 2), (256, 5, 2), (128, 5, 2)]

    input = Input(shape=(input_size,), name="input_z")

    _out = Dense(units=1024 * 4 * 4)(input)

    _out = BatchNormalization()(_out)
    _out = Activation('relu')(_out)

    _out = Reshape((1024, 4, 4))(_out)

    for f, kernel, strides in features:
        _out = Conv2DTranspose(filters=f,
                               kernel_size=5,
                               strides=2,
                               padding="same",
                               data_format="channels_first")(_out)
        _out = BatchNormalization()(_out)
        _out = Activation('relu')(_out)

    _out = Conv2DTranspose(filters=3,
                           kernel_size=5,
                           padding="same",
                           activation='tanh',
                           data_format="channels_first")(_out)

    model = Model(input, _out)
    return model


def make_discriminator(dropout_rate=0.5):
    features = [(256, 5, 2), (512, 5, 2), (1024, 5, 2)]

    _input = Input(shape=(3, 32, 32))

    _out = _input
    _out = Convolution2D(filters=128,
                         kernel_size=5,
                         padding='same',
                         strides=2,
                         data_format="channels_first")(_out)

    _out = LeakyReLU(alpha=0.2)(_out)

    for f,  kernel, strides in features:
        _out = Convolution2D(filters=f,
                             kernel_size=5,
                             padding='same',
                             strides=2,
                             data_format="channels_first")(_out)

        _out = BatchNormalization()(_out)
        _out = LeakyReLU(alpha=0.2)(_out)

        # _out = Dropout(rate=dropout_rate)(_out)

    _out = Flatten()(_out)

    # _out = Dense(1000)(_out)
    # _out = BatchNormalization()(_out)
    #
    # _out = LeakyReLU()(_out)
    #
    # _out = Dropout(rate=dropout_rate)(_out)

    _out = Dense(1, activation='sigmoid')(_out)

    model = Model(_input, _out)

    return model


def set_trainable(m, v):
    for l in m.layers:
        l.trainable = v


noise_size = 100

_generator_model = make_generator(noise_size)
_discriminator_model = make_discriminator()


set_trainable(_generator_model, False)

# compile discriminator on top of non-trainable generator to train with generated img

_input_real_img = Input(shape=(3, 32, 32), name="input_img")
_out_disc_fake = _discriminator_model(_input_real_img)


def dis_loss(y_true, y_pred):
    return binary_crossentropy(
        K.random_normal(mean=.9, stddev=.1, shape=y_pred.shape),
        y_pred)



discriminator = Model(inputs=_input_real_img, outputs=_out_disc_fake)

discriminator.compile(optimizer=Adam(lr=.0002, beta_1=.5),
                      loss=dis_loss)

_input_gen = Input(shape=(noise_size,), name="input_noise")
_out_gen = _generator_model(_input_gen)
_out_disc_fake = _discriminator_model(_out_gen)

discriminator_noise = Model(inputs=_input_gen, outputs=_out_disc_fake)

def dis_loss_noise(y_true, y_pred):
    return binary_crossentropy(T.zeros(shape=y_pred.shape),y_pred)

discriminator_noise.compile(optimizer=Adam(lr=.0002, beta_1=.5), loss = dis_loss_noise)


set_trainable(_generator_model, True)
set_trainable(_discriminator_model, False)

# compile generator
_generator_model.compile(optimizer='sgd',
                  loss='mse')

_input_gen = Input(shape=(noise_size,))
_out_gen = _generator_model(_input_gen)
_out_disc_fake = _discriminator_model(_out_gen)


def gen_loss(y_true, y_pred):
    return binary_crossentropy(
                        T.ones(shape=y_pred.shape),
                        y_pred)

# compile generator with generator trainable (fixed discriminator)
generator_loss = Model(_input_gen, _out_disc_fake)
generator_loss.compile(
    optimizer=Adam(lr=.0002, beta_1=.5),
    loss=gen_loss)

generator = _generator_model

generator.summary()
generator_loss.summary()
set_trainable(_generator_model, False)
set_trainable(_discriminator_model, True)
discriminator.summary()
discriminator_noise.summary()




save_index = 0
save_folder = str(datetime.datetime.now())

def test_generator(size=32):
    r = np.random.normal(size=(100, noise_size))

    res = generator.predict_on_batch(r)

    res = (res.transpose(0, 2, 3, 1) * 127. + 127.).astype('uint8')

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


losses = {'d': [], 'g': []}
batch_size = 32


def train(dataset, nb_pass=1):

    def batchs_discrimator():
        np.random.shuffle(dataset)
        b = 0
        g = False

        while True:
            if g:
                t = np.random.normal(loc=0, scale=1, size=(batch_size, noise_size))
            else:
                if b + batch_size > dataset.shape[0]:
                    b = 0

                t = dataset[b:b + batch_size, :, :, :]

                b += batch_size

            yield (t, g)

            g = not g

    def batchs_GAN():
        while True:
            yield (np.random.normal(loc=0, scale=1, size=(batch_size, noise_size)),
                   True)

    state = 'd'

    steps = dataset.shape[0] // batch_size

    for i in range(nb_pass):
        if state == 'd':
            set_trainable(_generator_model, False)
            set_trainable(_discriminator_model, True)

            epoch = int(- i * 0.2 + 3.)
        else:
            set_trainable(_generator_model, True)
            set_trainable(_discriminator_model, False)

            epoch = 1

        print("Training %s" % state)
        for e in range(epoch):
            print("-- Pass n# %d --"%e,flush=True)

            loss_display = TextBar(format="loss: %.4f", mapping=0.)

            bar = pb.ProgressBar(widgets=[
                '[', loss_display,'] ',
                pb.Percentage(), ' ',
                pb.Bar(), ' ',
                pb.Timer(),
            ])

            if state == 'd':
                gen = batchs_discrimator()

                for s in bar(range(steps)):
                    data, from_gen = next(gen)
                    model = discriminator_noise if from_gen else discriminator

                    loss = model.train_on_batch(data, np.empty(shape=(batch_size,)))

                    loss_display.update_mapping(loss)
                    losses[state].append(loss)


            else:

                gen = batchs_GAN()

                for s in bar(range(steps)):
                    data, from_gen = next(gen)

                    loss = generator_loss.train_on_batch(data, np.empty(shape=(batch_size,)))

                    loss_display.update_mapping(loss=loss)
                    losses[state].append(loss)

        if state == 'g':
            test_generator()

        state = 'g' if state == 'd' else 'd'

    print(losses)
    plot_loss(losses)


# test_generator()

data = get_center(get_dataset(100, 0, 0, min=-1., max=1.)[0])

train(data, generator_loss, discriminator)
test_generator()