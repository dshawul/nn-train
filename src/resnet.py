from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Activation,
    Conv2D,
    BatchNormalization,
    Dense,
    Flatten,
    Add,
    Concatenate,
    Reshape
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import learning_phase

CHANNEL_AXIS = 3
V_BATCH_SIZE = None
RENORM = False
RENORM_RMAX = 1.0
RENORM_DMAX = 0.0
RENORM_MOM  = 0.99

def conv_bn_relu(x, filters, size, name):

    #convolution
    x = Conv2D(filters=filters, kernel_size=size,
                  strides=(1,1), padding="same",
                  use_bias=False,
                  kernel_initializer="he_normal",
                  kernel_regularizer=l2(1.e-4),name=name+"_conv")(x)

    #batch normalization
    if learning_phase() == 1:
        if RENORM:
            clipping = {
                "rmin": 1.0 / RENORM_RMAX,
                "rmax": RENORM_RMAX,
                "dmax": RENORM_DMAX
            }
            x = BatchNormalization(axis=CHANNEL_AXIS, epsilon=1e-5,
                fused=False, scale=False, center=True,
                renorm=True, renorm_clipping=clipping,renorm_momentum=RENORM_MOM,
                virtual_batch_size=V_BATCH_SIZE, name=name+"_bnorm")(x)
        else:
            x = BatchNormalization(axis=CHANNEL_AXIS, epsilon=1e-5,
                fused=(V_BATCH_SIZE == None), scale=False, center=True,
                virtual_batch_size=V_BATCH_SIZE, name=name+"_bnorm")(x)
    else:
        x = BatchNormalization(axis=CHANNEL_AXIS, epsilon=1e-5,
            fused=True, scale=False, center=True,
            name=name+"_bnorm")(x)

    #activation
    x = Activation('relu',name=name+"_relu")(x)

    return x

def build_a0net(x, blocks,filters, policy):
    #Convolution block
    x = conv_bn_relu(x,filters,3,"input")

    #Residual blocks
    for i in range(blocks-1):
        inp = x
        x = conv_bn_relu(x,filters,3,"res"+str(i+1)+"_1")
        x = conv_bn_relu(x,filters,3,"res"+str(i+1)+"_2")
        x = Add(name="shortcut_"+str(i))([x,inp])

    #value head
    vx = conv_bn_relu(x,1,1,"value")

    #policy head
    if policy == 1:
        px = conv_bn_relu(x,filters,3,"policy_1")
        px = conv_bn_relu(px,73,1,"policy_2")
    else:
        px = conv_bn_relu(x,4,1,"policy_1")

    return vx,px

def build_net(main_input_shape, aux_input_shape, blocks, filters, policy, NPOLICY, auxinp):
    """Builds a custom ResNet like architecture.
    """

    main_input = Input(shape=main_input_shape, name='main_input')
    x = main_input

    # body
    vx,px = build_a0net(x, blocks, filters, policy)

    # value head
    x = Flatten()(vx)
    x = Dense(256, activation='relu')(x)

    if auxinp:
        aux_input = Input(shape=aux_input_shape, name = 'aux_input')
        y = aux_input
        y = Dense( 32, activation='relu')(y)
        x = Concatenate()([x, y])

    x = Dense( 32, activation='relu')(x)
    value = Dense(3, activation='softmax', name='value')(x)

    # policy head
    if policy == 0:
        x = Flatten('channels_first')(px)
        policy = Dense(NPOLICY, activation='softmax', name='policy')(x)
    else:
        x = Reshape((NPOLICY,), name='policy')(px)
        policy = Activation("softmax", name='policya')(x)

    # model
    if auxinp:
        xi = [main_input, aux_input]
    else:
        xi = [main_input]
    model = Model(inputs=xi, outputs=[value, policy])
    return model
