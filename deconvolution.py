import datetime
import os
from datetime import date
from math import sqrt

import keras
import numpy as np
import theano
from PIL import Image
from keras.engine.topology import Layer
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.utils.visualize_util import plot
from theano import tensor as T
from theano.tensor.nnet import conv2d

from dataset import MODELS_FOLDER, load_data

theano.config.optimizer = 'fast_compile'
theano.config.floatX = 'float32'
theano.config.exception_verbosity = 'high'

def build_deconv_function():


    def shared_dataset(data_xy):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))

        return shared_x, T.cast(shared_y, 'int32')


    # convolution
    #

    FEATURES_SPACE_SIZES = [3, 10, 50, 100]
    INPUTS_SPACE = [(64,64), (32,32), (16,16), (8, 8)]

    KERNELS_SIZE = [(3,3), (3,3), (3,3)]
    STRIDES = [1, 1, 1]
    PADDINGS = [1, 1, 1]

    # n*64*64*3
    input_imgs = T.ftensor4("input")

    filters = [theano.shared(np.random.normal(
            scale=1.0/sqrt(9 * FEATURES_SPACE_SIZES[i]),
            size=(
                FEATURES_SPACE_SIZES[i + 1],
                FEATURES_SPACE_SIZES[i],
                *KERNELS_SIZE[i]
            )).astype('float32')) for i in range(3)]

    _input = input_imgs
    convolutions = []
    for i in range(3):
        _input = conv2d(_input, filters[i], border_mode=PADDINGS[i], subsample=(STRIDES[i], STRIDES[i]))
        convolutions.append(_input)

    return theano.function([input_imgs], convolutions[0])



def mod1():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 64, 64)))
    model.add(Convolution2D(50, 3, 3, border_mode='same'))
    model.add(MaxPooling2D())

    #
    #
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Convolution2D(100, 3, 3, border_mode='same'))
    model.add(MaxPooling2D())
    # #
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Convolution2D(200, 3, 3, border_mode='same'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(4,4)))

    model.add(Unpooling2D(poolsize=(4,4)))
    model.add(Deconvolution2D(256, 3, 3, border_mode='same', output_shape=(None, None, 8, 8)))

    model.add(Unpooling2D())
    model.add(Deconvolution2D(60, 3, 3, border_mode='same', output_shape=(None, None, 16, 16)))
    model.add(Deconvolution2D(25, 3, 3, border_mode='same', output_shape=(None, None, 16, 16)))
    #
    model.add(Unpooling2D())
    model.add(Deconvolution2D(8, 3, 3, border_mode='same', output_shape=(None, None, 32, 32)))
    model.add(Deconvolution2D(3, 3, 3, border_mode='same', output_shape=(None, None, 32, 32)))

    model.compile(optimizer='adam',
                  loss='kullback_leibler_divergence',
                  metrics=['accuracy'])

    trainset_x, trainset_y = load_data(0, 2000)
    f = build_deconv_function()

    trainset_x = trainset_x.transpose((0,3,1,2))
    trainset_y = trainset_y.transpose((0,3,1,2))

    model.fit(trainset_x, trainset_y, nb_epoch=10, batch_size=32)

def mod2():
    model = Sequential()

    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(3, 64, 64)))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(4,4)))

    #
    #
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(4,4)))
    # #

    model.add(Unpooling2D(pool_size=(8,8)))
    model.add(Deconvolution2D(50, 3, 3, border_mode='same', output_shape=(None, None, 32, 32)))
    model.add(Deconvolution2D(20, 3, 3, border_mode='same', output_shape=(None, None, 32, 32)))
    model.add(Deconvolution2D(3, 3, 3, border_mode='same', output_shape=(None, None, 32, 32)))

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    plot(model, to_file='model.png', show_shapes=True)

    trainset_x, trainset_y = load_data(0, 100)
    f = build_deconv_function()

    trainset_x = trainset_x.transpose((0, 3, 1, 2))
    trainset_y = trainset_y.transpose((0, 3, 1, 2))
    x, y = load_data(2001, 1)

    res = model.predict(x.transpose((0, 3, 1, 2)))

    # Image.fromarray(np.transpose(res,(0, 2, 3, 1)).squeeze(axis=0).astype('uint8')).show()

    # model.fit(trainset_x, trainset_y, nb_epoch=1, batch_size=32)


    # model.save(os.path.join(MODELS_FOLDER, "deconvolution_%s.h5"%datetime.datetime.now()))

    model = keras.models.load_model('/home/louis/code/ift6266/models/deconvolution_2017-02-21 14:00:12.417065.h5',
                                    custom_objects={'Unpooling2D': Unpooling2D})

    Image.fromarray(x.squeeze(axis=0)).show()

    res = model.predict(x.transpose((0, 3, 1, 2)))
    Image.fromarray(np.transpose(res,(0, 2, 3, 1)).squeeze(axis=0).astype('uint8')).show()



    x, y = load_data(2001, 1)

mod2()