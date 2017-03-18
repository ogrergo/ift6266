from keras.engine.topology import InputSpec
from keras.layers.convolutional import Conv2DTranspose, Conv2D, UpSampling2D
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K

def conv_layer(_latent, previous_layer, activation='relu', batch_norm=True, deconv=False):
    _out = previous_layer

    if deconv:
        M = Conv2DTranspose
        _out = UpSampling2D((2,2))(_out)
    else:
        M = Conv2D

    _out = M(_latent, (3, 3), padding="same", data_format="channels_first", use_bias=not batch_norm)(_out)

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
