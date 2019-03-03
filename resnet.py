from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Add,
    Concatenate,
    Reshape
)
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

CHANNEL_AXIS = 3

def conv_bn(x, filters, size):
    x = Conv2D(filters=filters, kernel_size=size,
                  strides=(1,1), padding="same",
                  use_bias=False,
                  kernel_initializer="he_normal",
                  kernel_regularizer=l2(1.e-4))(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    return x

def conv_bn_relu(x, filters, size):
    x = conv_bn(x, filters, size)
    x = Activation("relu")(x)
    return x

def build_a0net(x, blocks,filters, policy):
    #Convolution block
    x = conv_bn_relu(x,filters,3)

    #Residual blocks
    for i in range(blocks-1):
        inp = x
        x = conv_bn_relu(x,filters,3)
        x = conv_bn(x,filters,3)
        x = Add()([x,inp])
        x = Activation("relu")(x)

    #value head
    vx = conv_bn_relu(x,1,1)

    #policy head
    if policy == 1:
        px = conv_bn_relu(x,filters,3)
    else:
        px = x
    px = conv_bn_relu(px,73,1)

    return vx,px

def build_net(main_input_shape, aux_input_shape, blocks, filters, policy):
    """Builds a custom ResNet like architecture.
    """

    main_input = Input(shape=main_input_shape, name='main_input')
    aux_input = Input(shape=aux_input_shape, name = 'aux_input')
    x = main_input
    y = aux_input

    # body
    vx,px = build_a0net(x, blocks, filters, policy)

    # value head
    x = Flatten()(vx)
    x = Dense(256, activation='tanh')(x)
    y = Dense( 32, activation='tanh')(y)
    x = Concatenate()([x, y])
    x = Dense( 32, activation='tanh')(x)
    value = Dense(3, activation='softmax', name='value')(x)

    # policy head
    if policy == 0:
        x = Flatten('channels_first')(px)
        policy = Dense(1858, activation='softmax', name='policy')(x)
    else:
        x = Reshape((4672,), name='policy')(px)
        policy = Activation("softmax", name='policya')(x)

    # model
    model = Model(inputs=[main_input, aux_input], outputs=[value, policy])
    return model
