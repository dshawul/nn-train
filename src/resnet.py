from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Activation,
    Conv2D,
    BatchNormalization,
    Dense,
    Flatten,
    Add,
    Reshape,
    GlobalAveragePooling2D,
    Multiply,
    Permute,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import learning_phase

from transformer import Encoder

CHANNEL_AXIS = 3
L2_REG = l2(5.0e-5)
K_INIT = "glorot_normal"

V_BATCH_SIZE = None
RENORM = False
RENORM_RMAX = 1.0
RENORM_DMAX = 0.0
RENORM_MOM = 0.99

USE_SE = True

# Specify body type
#   0 = post-activation
#   1 = pre-activation
#   2 = transformer
BODY_TYPE = 0


def dense(x, n, name, act="relu"):

    x = Dense(
        n,
        activation=act,
        kernel_regularizer=L2_REG,
        kernel_initializer=K_INIT,
        name=name,
    )(x)

    return x


def conv(x, filters, size, name):

    x = Conv2D(
        filters=filters,
        kernel_size=size,
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer=K_INIT,
        kernel_regularizer=L2_REG,
        name=name + "_conv",
    )(x)

    return x


def batch_norm(x, scale, name):

    if learning_phase() == 1:
        if RENORM:
            clipping = {
                "rmin": 1.0 / RENORM_RMAX,
                "rmax": RENORM_RMAX,
                "dmax": RENORM_DMAX,
            }
            x = BatchNormalization(
                axis=CHANNEL_AXIS,
                epsilon=1e-5,
                fused=False,
                scale=scale,
                center=True,
                renorm=True,
                renorm_clipping=clipping,
                renorm_momentum=RENORM_MOM,
                virtual_batch_size=V_BATCH_SIZE,
                name=name + "_bnorm",
            )(x)
        else:
            x = BatchNormalization(
                axis=CHANNEL_AXIS,
                epsilon=1e-5,
                fused=(V_BATCH_SIZE == None),
                scale=scale,
                center=True,
                virtual_batch_size=V_BATCH_SIZE,
                name=name + "_bnorm",
            )(x)
    else:
        x = BatchNormalization(
            axis=CHANNEL_AXIS,
            epsilon=1e-5,
            fused=True,
            scale=scale,
            center=True,
            name=name + "_bnorm",
        )(x)

    return x


def bn_relu(x, scale, name):

    x = batch_norm(x, scale, name)
    x = Activation("relu", name=name + "_relu")(x)

    return x


def conv_bn(x, filters, size, scale, name):

    x = conv(x, filters, size, name)
    x = batch_norm(x, scale, name)

    return x


def conv_bn_relu(x, filters, size, scale, name):

    x = conv(x, filters, size, name)
    x = bn_relu(x, scale, name)

    return x


def bn_relu_conv(x, filters, size, scale, name):

    x = bn_relu(x, scale, name)
    x = conv(x, filters, size, name)

    return x


def squeeze_excite_block(inp, filters, ratio, name):

    inp = Permute((3, 1, 2), name=name + "_permute_1")(inp)
    x = GlobalAveragePooling2D("channels_first", name=name + "_avg_pool")(inp)
    x = dense(x, filters // ratio, name + "_dense_1")
    x = dense(x, filters, name + "_dense_2", act="sigmoid")
    x = Reshape((filters, 1, 1), name=name + "_reshape")(x)
    x = Multiply(name=name + "_multiply")([inp, x])
    x = Permute((2, 3, 1), name=name + "_permute_2")(x)
    return x


def pool_layer(x, name):

    x = Permute((3, 1, 2), name=name + "_permute")(x)
    x = GlobalAveragePooling2D("channels_first", name=name + "_avg_pool")(x)
    return x


def build_post_net(x, blocks, filters):

    for i in range(blocks - 1):
        inp = x
        x = conv_bn_relu(x, filters, 3, False, "res" + str(i + 1) + "_1")
        x = conv_bn(x, filters, 3, True, "res" + str(i + 1) + "_2")
        if USE_SE:
            x = squeeze_excite_block(
                x, filters, filters // 32, "res" + str(i + 1) + "_se"
            )
        x = Add(name="shortcut_" + str(i))([x, inp])
        x = Activation("relu", name="res" + str(i + 1) + "_relu")(x)

    return x


def build_pre_net(x, blocks, filters, pol_channels):

    for i in range(blocks - 1):
        inp = x
        if i == 0:
            x = conv(x, filters, 3, "res" + str(i + 1) + "_1")
        else:
            x = bn_relu_conv(x, filters, 3, False, "res" + str(i + 1) + "_1")
        x = bn_relu_conv(x, filters, 3, True, "res" + str(i + 1) + "_2")
        if USE_SE:
            x = squeeze_excite_block(
                x, filters, filters // 32, "res" + str(i + 1) + "_se"
            )
        x = Add(name="shortcut_" + str(i))([x, inp])

    x = bn_relu(x, True, "final")

    return x


def value_head(x):

    vx = conv_bn_relu(x, 128, 1, False, "value")
    vx = pool_layer(vx, "value")

    # value head
    x = Flatten(name="value_flatten")(vx)
    x = dense(x, 128, "value_dense_1")
    x = dense(x, 32, "value_dense_3")
    x = dense(x, 3, "value", act=None)
    value = Activation("softmax", name="valuea", dtype="float32")(x)

    return value


def policy_head(x, filters, pol_channels):

    px = conv_bn_relu(x, filters, 3, False, "policy_1")
    px = conv_bn(px, pol_channels, 1, True, "policy_2")

    # policy head
    x = Reshape((-1,), name="policy")(px)
    policy = Activation("softmax", name="policya", dtype="float32")(x)

    return policy


def score_head(x, filters, pol_channels):

    px = conv_bn_relu(x, filters, 3, False, "score_1")
    px = conv_bn(px, pol_channels, 1, True, "score_2")

    # score head
    x = Reshape((-1,), name="score")(px)
    score = Activation("sigmoid", name="scorea", dtype="float32")(x)

    return score


def build_net(main_input_shape, blocks, filters, num_heads, pol_channels, HEAD_TYPE):

    main_input = Input(shape=main_input_shape, name="main_input")
    x = main_input

    # residual/transformer tower
    if BODY_TYPE == 2:
        training = learning_phase() == 1
        x = Encoder(
            num_layers=blocks,
            d_model=filters,
            num_heads=num_heads,
            dff=4*filters
        )(x, training=training)
    else:
        x = conv_bn_relu(x, filters, 3, True, "input")
        if BODY_TYPE == 0:
            x = build_post_net(x, blocks, filters)
        elif BODY_TYPE == 1:
            x = build_pre_net(x, blocks, filters)

    # value
    value = value_head(x)

    # policy
    if HEAD_TYPE == 0:
        policy = policy_head(x, filters, pol_channels)
        outputs = [value, policy]
    elif HEAD_TYPE == 1:
        score = score_head(x, filters, pol_channels)
        outputs = [value, score]
    elif HEAD_TYPE == 2:
        policy = policy_head(x, filters, pol_channels)
        score = score_head(x, filters, pol_channels)
        outputs = [value, policy, score]
    else:
        outputs = [value]

    # model
    model = Model(inputs=[main_input], outputs=outputs)

    return model
