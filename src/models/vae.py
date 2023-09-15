import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Input, Layer, Reshape

from src.models.layers import ResidualBlock


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def create_encoder(
    img_height,
    img_width,
    latent_dim=2,
    nfilters=64,
    ):

    input_layer = Input(shape=(img_height, img_width, 1))

    conv_1 = Conv2D(nfilters, 4, activation="relu", strides=2, padding="same")
    conv_2 = Conv2D(nfilters, 4, activation="relu", strides=2, padding="same")

    res_block_1 = ResidualBlock(nfilters, l2=0)
    res_block_2 = ResidualBlock(nfilters, l2=0)

    dense = Dense(latent_dim, activation="relu")

    dense_mean = Dense(latent_dim, name="z_mean")
    dense_log_var = Dense(latent_dim, name="z_log_var")

    flatten = Flatten()
    sampling = Sampling()

    x = conv_1(input_layer)
    x = conv_2(x)
    x = res_block_1(x) + x
    x = res_block_2(x) + x
    x = dense(flatten(x))

    z_mean, z_log_var = dense_mean(x), dense_log_var(x)
    z = sampling([z_mean, z_log_var])

    return Model(input_layer, [z_mean, z_log_var, z], name="encoder")


def create_decoder(
    img_height,
    img_width,
    latent_dim=2,
    nfilters=256,
    ):
    input_layer = Input(shape=(latent_dim,))

    # Two deconvolutions with stride 2, means out_dim = 2 * 2 * in_dim
    in_height, in_width = img_height // 4, img_width // 4

    dense_1 = Dense(in_height * in_width * 256, activation="relu")
    reshape_layer = Reshape((in_height, in_width, 256))

    res_block_1 = ResidualBlock(256, l2=0)
    res_block_2 = ResidualBlock(256, l2=0)

    conv_1 = Conv2DTranspose(nfilters, 4, activation="relu", strides=2, padding="same")
    conv_2 = Conv2DTranspose(nfilters, 4, activation="relu", strides=2, padding="same")
    conv_3 = Conv2DTranspose(1, 4, activation=None, padding="same")

    x = dense_1(input_layer)
    x = reshape_layer(x)

    x = res_block_1(x) + x
    x = res_block_2(x) + x

    x = conv_1(x)
    x = conv_2(x)

    decoder_outputs = conv_3(x)
    return Model(input_layer, decoder_outputs, name="decoder")
