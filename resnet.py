from __future__ import division
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Add,
    concatenate
)
from keras.layers.convolutional import (
    Conv2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3

def build_a0net(x, blocks,filters):
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
    x = Conv2D(filters=1, kernel_size=(1,1),
                      strides=(1,1), padding="same",
                      kernel_initializer="he_normal",
                      use_bias=False,
                      kernel_regularizer=l2(1.e-4))(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation("relu")(x)

    return x

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            input = block_function(filters=filters, init_strides=(1,1),
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f

def build_a1net(x, blocks, filters):

    # convolution block
    x = _conv_bn_relu(filters=filters, kernel_size=(3, 3), strides=(1, 1))(x)

    # residual blocks
    repetitions = [2]*int(blocks/2)
    repetitions[0] = 1
    for i, r in enumerate(repetitions):
        x = _residual_block(basic_block, filters=filters, repetitions=r, is_first_layer=(i == 0))(x)
    x = _bn_relu(x)
    
    # value head
    x = _conv_bn_relu(filters=1, kernel_size=(1, 1), strides=(1, 1))(x)

    return x

def build_net(main_input_shape, aux_input_shape, blocks, filters):
    """Builds a custom ResNet like architecture.
    """

    main_input = Input(shape=main_input_shape, name='main_input')
    aux_input = Input(shape=aux_input_shape, name = 'aux_input')
    x = main_input
    y = aux_input

    # x = build_a1net(x, blocks, filters)
    x = build_a0net(x, blocks, filters)

    # value head
    x = Flatten()(x)
    x = Dense(256, activation='tanh')(x)
    y = Dense( 32, activation='tanh')(y)
    x = concatenate([x, y])
    x = Dense( 32, activation='tanh')(x)
    output = Dense(3, activation='softmax', name='value')(x)

    # model
    model = Model(inputs=[main_input, aux_input], outputs=output)
    return model
