from __future__ import print_function
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Concatenate
)
from tensorflow.keras import activations, layers
from tensorflow.keras.regularizers import l2
import tensorflow as tf

L2_REG = l2(1.e-4)
K_INIT = "glorot_normal"

def clipped_relu(x):

    return activations.relu(x, max_value=1.0)

class DenseLayerForSparse(layers.Layer):
    def __init__(self, num_units, input_size, activation=clipped_relu, name=None,**kwargs):
        super(DenseLayerForSparse, self).__init__(name=name)
        self.num_units = num_units
        self.input_size = input_size
        self.activation = activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.input_size, self.num_units],
            regularizer = L2_REG,
            initializer = K_INIT
        )
        self.bias = self.add_weight(
            "bias",
            shape=[self.num_units],
            regularizer = L2_REG,
            initializer = K_INIT)

    def call(self, inputs, **kwargs):
        outputs = tf.add(tf.sparse.sparse_dense_matmul(inputs, self.kernel), self.bias)
        return self.activation(outputs)

    def get_config(self):
        config = super(DenseLayerForSparse, self).get_config()
        config.update({
            "input_size": self.input_size,
            "num_units": self.num_units,
        })
        return config

def dense(x, n, name, act=clipped_relu):

    x = Dense(n, activation=act,
              kernel_regularizer=L2_REG,
              kernel_initializer=K_INIT,
              name=name)(x)

    return x

def input_head(main_input_shape):

    main_input = Input(shape=main_input_shape,sparse=True)
    x = main_input

    x = DenseLayerForSparse(256, main_input_shape[0], name="sparse_input_dense")(x)

    model = Model(inputs=[main_input], outputs=[x])

    return model

def value_head(x):

    x = dense(x,  32, "value_dense_2")
    x = dense(x,  32, "value_dense_3")
    value = dense(x, 1, "value", act='sigmoid')

    return value

def build_net(main_input_shape):

    input_model = input_head(main_input_shape)

    player_input = Input(shape=main_input_shape,sparse=True,name='player_input')
    opponent_input = Input(shape=main_input_shape,sparse=True,name='opponent_input')
    hp = input_model(player_input)
    ho = input_model(opponent_input)
    x = Concatenate()([hp, ho])

    #value
    value = value_head(x)

    #model
    model = Model(inputs=[player_input,opponent_input], outputs=[value])

    return model
