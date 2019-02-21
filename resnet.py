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

NVALUE = 3
CHANNEL_AXIS = 3

def build_a0net(x, blocks,filters, policy):
    """Build A0 style resnet
    """

    #Convolution block
    x = Conv2D(filters=filters, kernel_size=(3,3),
                      strides=(1,1), padding="same",
                      kernel_initializer="he_normal",
                      use_bias=False,
                      kernel_regularizer=l2(1.e-4))(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation("relu")(x)

    #Residual blocks
    for i in range(blocks-1):
        inp = x
        x = Conv2D(filters=filters, kernel_size=(3,3),
                  strides=(1,1), padding="same",
                  kernel_initializer="he_normal",
                  use_bias=False,
                  kernel_regularizer=l2(1.e-4))(x)
        x = BatchNormalization(axis=CHANNEL_AXIS)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=filters, kernel_size=(3,3),
                  strides=(1,1), padding="same",
                  kernel_initializer="he_normal",
                  use_bias=False,
                  kernel_regularizer=l2(1.e-4))(x)
        x = BatchNormalization(axis=CHANNEL_AXIS)(x)
        x = Add()([x,inp])
        x = Activation("relu")(x)

    #value head
    vx = Conv2D(filters=1, kernel_size=(1,1),
                      strides=(1,1), padding="same",
                      kernel_initializer="he_normal",
                      use_bias=False,
                      kernel_regularizer=l2(1.e-4))(x)
    vx = BatchNormalization(axis=CHANNEL_AXIS)(vx)
    vx = Activation("relu")(vx)

    #policy head
    if policy == 1:
        px = Conv2D(filters=filters, kernel_size=(3,3),
                          strides=(1,1), padding="same",
                          kernel_initializer="he_normal",
                          use_bias=False,
                          kernel_regularizer=l2(1.e-4))(x)
        px = BatchNormalization(axis=CHANNEL_AXIS)(px)
        px = Activation("relu")(px)
    else:
      px = x

    px = Conv2D(filters=73, kernel_size=(1,1),
                      strides=(1,1), padding="same",
                      kernel_initializer="he_normal",
                      use_bias=False,
                      kernel_regularizer=l2(1.e-4))(px)
    px = BatchNormalization(axis=CHANNEL_AXIS)(px)
    px = Activation("relu")(px)

    return vx,px

def build_net(main_input_shape, aux_input_shape, blocks, filters, policy):
    """Builds a custom ResNet like architecture.
    """

    main_input = Input(shape=main_input_shape, name='main_input')
    aux_input = Input(shape=aux_input_shape, name = 'aux_input')
    x = main_input
    y = aux_input

    vx,px = build_a0net(x, blocks, filters, policy)

    # value head
    x = Flatten()(vx)
    x = Dense(256, activation='tanh')(x)
    y = Dense( 32, activation='tanh')(y)
    x = Concatenate()([x, y])
    x = Dense( 32, activation='tanh')(x)
    value = Dense(NVALUE, activation='softmax', name='value')(x)

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
