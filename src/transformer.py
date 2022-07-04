from tensorflow.keras.layers import (
    Dense,
    LayerNormalization,
    Dropout,
    Layer,
    Reshape,
    Permute,
)
from tensorflow.keras.regularizers import l2
import tensorflow as tf

CHANNEL_AXIS = 3
L2_REG = l2(5.0e-5)
K_INIT = "glorot_normal"

# attention
class MultiHeadAttention(Layer):
    def __init__(self, *, d_model, num_heads, name=None):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = Dense(
            d_model,
            kernel_initializer=K_INIT,
            kernel_regularizer=L2_REG,
            name="linear_q",
        )
        self.wk = Dense(
            d_model,
            kernel_initializer=K_INIT,
            kernel_regularizer=L2_REG,
            name="linear_k",
        )
        self.wv = Dense(
            d_model,
            kernel_initializer=K_INIT,
            kernel_regularizer=L2_REG,
            name="linear_v",
        )
        self.fd = Dense(
            d_model,
            kernel_initializer=K_INIT,
            kernel_regularizer=L2_REG,
            name="linear_f",
        )

    def split_heads(self, x):
        x = Reshape((-1, self.num_heads, self.depth))(x)
        x = Permute((2, 1, 3))(x)
        return x

    @staticmethod
    def scaled_dot_product_attention(q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], q.dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def call(self, q, k, v):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        scaled_attention = Permute((2, 1, 3))(scaled_attention)
        concat_attention = Reshape((-1, self.d_model), name="concat")(scaled_attention)

        output = self.fd(concat_attention)

        return output, attention_weights


# encoder layer
class EncoderLayer(Layer):
    def __init__(self, *, d_model, num_heads, dff, name=None, rate=0.1):
        super(EncoderLayer, self).__init__(name=name)

        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            name="attention",
        )
        self.ffn1 = Dense(
            dff,
            activation="relu",
            kernel_initializer=K_INIT,
            kernel_regularizer=L2_REG,
            name="ffn_dense_1",
        )
        self.ffn2 = Dense(
            d_model,
            kernel_initializer=K_INIT,
            kernel_regularizer=L2_REG,
            name="ffn_dense_2",
        )
        self.norm1 = LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = LayerNormalization(epsilon=1e-6, name="norm2")
        self.dropout1 = Dropout(rate, name="dropout1")
        self.dropout2 = Dropout(rate, name="dropout2")

    def call(self, x, training):
        attn_output, _ = self.mha(x, x, x)
        attn_output = tf.reshape(attn_output, tf.shape(x))
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)

        ffn_output = self.ffn1(out1)
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)
        return out2


# transformer encoder
class Encoder(Layer):
    def __init__(
        self, num_layers, d_model, num_heads, dff, rate=0.1, name=None, **kwargs
    ):
        super(Encoder, self).__init__(name=name)

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.embedding = Dense(
            self.d_model,
            kernel_initializer=K_INIT,
            kernel_regularizer=L2_REG,
            activation="relu",
            name="embedding",
        )

        self.enc_layers = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                rate=rate,
                name="layer_" + str(idx + 1),
            )
            for idx in range(num_layers)
        ]

    def call(self, x, training):
        x = self.embedding(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        return x

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "rate": self.rate,
            }
        )
        return config
