import os

from PIL import Image
from keras.engine.topology import InputSpec
from keras.layers.convolutional import Conv2DTranspose, Conv2D, UpSampling2D
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
import numpy as np
from progressbar.widgets import FormatWidgetMixin, WidthWidgetMixin

from dataset import save_image


def conv_layer(_latent, previous_layer, activation='relu', batch_norm=True, deconv=False):
    _out = previous_layer

    if deconv:
        M = Conv2DTranspose
        _out = UpSampling2D((2,2))(_out)
    else:
        M = Conv2D

    _out = M(_latent, kernel_size=3, padding="same", data_format="channels_first", use_bias=not batch_norm)(_out)

    if batch_norm:
        _out = BatchNormalization()(_out)

    if activation != None:
        _out = Activation(activation=activation)(_out)

    if not deconv:
        _out = MaxPooling2D((2,2))(_out)

    return _out


class DenseTied(Dense):
    def __init__(self, tied_to, **args):

        self.tied_to = tied_to
        if not isinstance(self.tied_to, Dense):
            raise ValueError("Can be only tied to a Dense layer.")

        units = self.tied_to.input_shape[-1]

        super().__init__(units=units, **args)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to:
            self.kernel = K.transpose(self.tied_to.kernel)
        else:
            self.kernel = self.add_weight((input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight((self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


def test_model(name, imgs, epoch, model, save=True, show=True):
    size = imgs[0].shape[2]

    result = [(model.predict(np.expand_dims(e, axis=0)).transpose((0, 2, 3, 1)).squeeze(axis=0) * 255).astype('uint8') for
              e in imgs]

    rows = len(imgs) // 10
    img = np.zeros(shape=(size * 10, size * rows, 3), dtype='uint8')
    index = lambda i: ((i % 10) * size, (i // 10) * size)

    for i, r in enumerate(result):
        x, y = index(i)
        img[x:(size + x), y:(size + y), :] = result[i]

    if show:
        Image.fromarray(img).show()

    if save:
        save_image(name, "epoch_%d" % epoch, img)



class TextBar(FormatWidgetMixin, WidthWidgetMixin):
    mapping = {}

    def __init__(self, format, mapping=mapping, **kwargs):
        self.format = format
        self.mapping = mapping
        FormatWidgetMixin.__init__(self, format=format, **kwargs)
        WidthWidgetMixin.__init__(self, **kwargs)

    def update_mapping(self, mapping):
        self.mapping = mapping

    def __call__(self, progress, data, format=None):
        return FormatWidgetMixin.__call__(self, progress, self.mapping,
                                          self.format)