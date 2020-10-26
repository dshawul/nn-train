from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Concatenate,
    Reshape
)
from tensorflow.keras import activations
from tensorflow.keras.regularizers import l2

L2_REG = l2(1.e-4)
K_INIT = "glorot_normal"

def clipped_relu(x):

    return activations.relu(x, max_value=1.0)

def dense(x, n, name, act=clipped_relu):

    x = Dense(n, activation=act,
              kernel_regularizer=L2_REG,
              kernel_initializer=K_INIT,
              name=name)(x)

    return x

def input_head(main_input_shape):

    main_input = Input(main_input_shape)
    x = main_input

    x = Flatten(name='value_flatten')(x)
    x = dense(x, 256, "value_dense_1")

    model = Model(inputs=[main_input], outputs=[x])

    return model

def value_head(x):

    x = dense(x,  32, "value_dense_2")
    x = dense(x,  32, "value_dense_3")
    value = dense(x, 1, "value", act='sigmoid')

    return value

def build_net(main_input_shape):

    input_model = input_head(main_input_shape)

    player_input = Input(shape=main_input_shape, name='player_input')
    opponent_input = Input(shape=main_input_shape, name='opponent_input')
    hp = input_model(player_input)
    ho = input_model(opponent_input)
    x = Concatenate()([hp, ho])

    #value
    value = value_head(x)

    #model
    model = Model(inputs=[player_input,opponent_input], outputs=[value])

    return model
