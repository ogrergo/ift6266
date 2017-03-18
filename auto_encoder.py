from PIL import Image
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.core import Flatten, Dense, Reshape, Lambda
import theano.tensor as T
import numpy as np
import keras.backend as K

from dataset import save_image, get_border, load_data, get_dataset
from utils import conv_layer, DenseTied


def model_autoencoder():
    features = [16, 32]

    reversed_features = features[len(features) - 2::-1] + [3]
    _ae_hid = 256

    _input = Input(shape=(3, 64, 64))

    _out = _input
    for f in features:
        _out = conv_layer(f, _out)

    _shape = _out._keras_shape[1:]

    _out = Flatten()(_out)
    _hidden_dense = Dense(units=_ae_hid, activation='sigmoid')


    _out = _hidden_dense(_out)
    _out = DenseTied(tied_to=_hidden_dense, activation='sigmoid')(_out)
    _out = Reshape(_shape)(_out)

    for f in reversed_features:
        _out = conv_layer(f, _out, deconv=True)

    def reconstitution(x):
        return T.set_subtensor(_input[:, :, 16:48, 16:48], K.clip(x, 0.0, 1.0)[:, :, 16:48, 16:48])
        # return K.clip(x, 0.0, 1.0)

    _out = Lambda(function=reconstitution, output_shape=(3, 64, 64))(_out)

    m = Model(_input, _out, "auto_encoder")
    m.compile(
        optimizer='adam',
        loss='mean_squared_error',
    )
    m.summary()
    return m


def test_ae(imgs, epoch, ae, save=True, show=True):
    result = [(ae.predict(np.expand_dims(e, axis=0)).transpose((0, 2,3,1)).squeeze(axis=0) * 255).astype('uint8') for e in imgs]
    rows = len(imgs) // 10
    img = np.zeros(shape=(64 * 10, 64 * rows, 3), dtype='uint8')
    index = lambda i: ((i % 10) * 64, (i // 10) * 64)

    for i, r in enumerate(result):
        x, y = index(i)
        img[x:(64 + x), y:(64 + y), :] = result[i]

    if show:
        Image.fromarray(img).show()

    if save:
        save_image('auto_encoder', "epoch_%d"%epoch, img)


def train_ae(model, sets, show=False):
    output_train, output_valid, output_test = sets
    input_train = get_border(output_train)
    input_valid = get_border(output_valid)
    input_test = get_border(output_test)

    def visualisation(epoch, log):
        test_ae([input_test[i] for i in range(100)], epoch, model, show=show)

    visualisation(0, None)

    model.fit(input_train, output_train,
            shuffle=True,
            epochs=30,
            batch_size=16,
            validation_data=(input_valid, output_valid),
            callbacks=[LambdaCallback(on_epoch_end=visualisation),
                       ModelCheckpoint(filepath="weights/auto_encoder/weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_weights_only=True),
                       EarlyStopping(monitor='val_loss', patience=2)])


if __name__ == '__main__':

    model = model_autoencoder()

    train_ae(model, get_dataset(3000, 1000))