from functools import partial

import theano
from PIL import Image
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.engine.topology import Input, InputSpec
from keras.engine.training import Model
from keras.layers.convolutional import Convolution2D, Deconvolution2D, Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.core import Dense, Flatten, Reshape, Lambda, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model, model_from_json
from keras.objectives import mean_squared_error
# from keras.utils.visualize_util import plot
import theano.tensor as T
from dataset import load_data, save_image, get_border, get_dataset, get_center
from keras import objectives, metrics
import numpy as np

import keras.backend as K

# input : 3*64*64
# output : 3*32*32

# normalisation
# constraint on


# theano.exception_verbosity='high'
from utils import conv_layer, DenseTied, test_model


def model_variational_autoencoder():
    # size of the intermediate convolutions
    # features = [(64, 2), (64, 2), (128, 2), (128, 2)]
    # reversed_features = features[len(features) - 2::-1] + [3]
    # reversed_features = [(128, 2), (64, 1), (64, 2), (64, 2), (32, 2)]
    features = [(32, 2), (64, 2), (64, 1)]
    reversed_features = [(64, 1), (64, 2), (32, 2)]


    _hidden_dense = 500

    # size of the blank noise vector
    _hidden_noise = 60

    # take the border in input
    _input = Input(shape=(3, 64, 64))

    _out = _input
    for f, s in features:
        _out = Conv2D(
            f,
            kernel_size=(2,2), strides=s,
            padding='same', activation='elu')(_out)

        # _out = conv_layer(f, _out)

    # save the current shape (without batch size) for later deconvolutions
    _shape = _out._keras_shape[1:]

    _out = Flatten()(_out)

    _out = Dense(units=_hidden_dense, activation='elu')(_out)
    _out = Dense(units=_hidden_dense, activation='elu')(_out)

    # the mean and variance for blank noise
    z_mean = Dense(units=_hidden_noise)(_out)
    z_log_var = Dense(units=_hidden_noise)(_out)

    def sampling(args):
        z_mean, z_log_var = args

        epsilon = K.random_normal(shape=(z_mean.shape[0], _hidden_noise),
                                  mean=0., stddev=1.0)

        return z_mean + K.exp(z_log_var) * epsilon

    _out = Lambda(function=sampling, output_shape=(_hidden_noise,))([z_mean, z_log_var])

    _out = Dense(units=_hidden_dense, activation='elu')(_out)
    _out = Dense(units=_hidden_dense, activation='elu')(_out)

    units = _shape[0] * _shape[1] * _shape[2]
    # in autoencoder, this was implicit by DenseTied
    _out = Dense(units=units, activation='elu')(_out)
    # _out = DenseTied(tied_to=_dense, activation='sigmoid')(_out)
    _out = Reshape(_shape)(_out)

    for f, s in reversed_features[:]:
        _out = Conv2DTranspose(f,
                               kernel_size=3,
                               strides=s,
                               padding='same',
                               activation='elu',
                               data_format='channels_first')(_out)
        # _out = conv_layer(f, _out, deconv=True)

    _out = Conv2D(3, kernel_size=3, strides=1, padding='same', activation='sigmoid')(_out)
    # _out = conv_layer(reversed_features[-1], _out, deconv=True, activation='sigmoid')

    # def reconstitution(x):
    #     return T.set_subtensor(_input[:, :, 16:48, 16:48], K.clip(x, 0.0, 1.0)[:, :, 16:48, 16:48])
    #     # return K.clip(x, 0.0, 1.0)
    #
    # _out = Lambda(function=reconstitution, output_shape=(3, 64, 64))(_out)

    # definition of the loss (reconstruction error + KL divergence)
    def vae_loss(y_true, y_pred):
        y_true = T.flatten(y_true)
        y_pred = T.flatten(y_pred)

        latent_loss = 64**2 * metrics.binary_crossentropy(y_true, y_pred)

        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        return latent_loss + kl_loss

    m = Model(_input, _out)
    m.compile(optimizer='rmsprop',
              loss=vae_loss)

    m.summary()
    return m


def train_vae(model, sets, batch_size=10, show=True):
    output_train, output_valid, output_test = sets
    input_train = get_border(output_train)
    input_valid = get_border(output_valid)
    input_test = get_border(output_test)

    middle_train = get_center(output_train)
    middle_valid = get_center(output_valid)
    middle_test = get_center(output_test)


    def visualisation(epoch, log):
        test_model("variational_autoencoder", [output_test[i] for i in range(100)], epoch, model, save=True, show=show)

    visualisation(0, None)

    model.fit(output_train, output_train,
            shuffle=True,
            epochs=150,
            batch_size=batch_size,
            validation_data=(output_valid, output_valid),
            callbacks=[LambdaCallback(on_epoch_end=visualisation),
                       ModelCheckpoint(filepath="weights/variational_autoencoder/weights.{epoch:02d}-{val_loss:.3f}.hdf5", save_weights_only=True),
                       EarlyStopping(monitor='val_loss', patience=2)])


if __name__ == '__main__':

    vae = model_variational_autoencoder()

    train_vae(vae, get_dataset(1000, 1000), show=True)