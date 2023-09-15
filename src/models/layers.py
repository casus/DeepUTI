from typing import Optional

import tensorflow as tf
from tensorflow import nn
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, Layer, Softmax,
    concatenate
)
from tensorflow.keras.regularizers import L2
from tensorflow_addons.layers import InstanceNormalization


class ConvBlock(Layer):
    def __init__(
        self,
        nfilters: int,
        size: int = 3,
        padding: str = "same",
        initializer: str = "he_normal",
        l2: float = 1e-4,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)

        self.nfilters = nfilters
        self.size = size
        self.padding = padding
        self.initializer = initializer
        self.l2 = l2

        self.conv_1 = Conv2D(
            filters=nfilters,
            kernel_size=(size, size),
            padding=padding,
            kernel_initializer=initializer,
            kernel_regularizer=L2(l2),
            bias_regularizer=L2(l2),
        )
        self.norm_1 = InstanceNormalization(
            axis=-1,
            center=True,
            scale=True,
            beta_initializer="random_uniform",
            gamma_initializer="random_uniform",
        )
        self.batch_norm_1 = BatchNormalization()
        self.relu_1 = Activation("relu")


        self.conv_2 = Conv2D(
            filters=nfilters,
            kernel_size=(size, size),
            padding=padding,
            kernel_initializer=initializer,
            kernel_regularizer=L2(l2),
            bias_regularizer=L2(l2),
        )
        self.norm_2 = InstanceNormalization(
            axis=-1,
            center=True,
            scale=True,
            beta_initializer="random_uniform",
            gamma_initializer="random_uniform",
        )
        self.batch_norm_2 = BatchNormalization()
        self.relu_2 = Activation("relu")

    def get_config(self) -> dict[str, any]:
        config = super().get_config()

        config.update({
            "nfilters": self.nfilters,
            "size": self.size,
            "padding": self.padding,
            "initializer": self.initializer,
            "l2": self.l2,
        })

        return config

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv_1(inputs)
        x = self.norm_1(x, training=training)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.norm_2(x, training=training)
        x = self.relu_2(x)

        return x

class DeconvBlock(Layer):
    def __init__(
        self,
        nfilters: int,
        size: int = 3,
        padding: str = "same",
        strides: tuple[int, int] = (2, 2),
        l2: float = 1e-4,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)

        self.nfilters = nfilters
        self.size = size
        self.padding = padding
        self.l2 = l2

        self.conv_transpose = Conv2DTranspose(
            nfilters,
            kernel_size=(size, size),
            strides=strides,
            padding=padding,
            kernel_regularizer=L2(l2),
            bias_regularizer=L2(l2),
            )

        self.conv_block = ConvBlock(nfilters)

    def get_config(self) -> dict[str, any]:
        config = super().get_config()

        config.update({
            "nfilters": self.nfilters,
            "size": self.size,
            "padding": self.padding,
            "l2": self.l2,
        })

        return config

    def call(self, inputs: tf.Tensor, residual: Optional[tf.Tensor] = None, training: bool = False) -> tf.Tensor:
        x = self.conv_transpose(inputs)
        if residual is not None:
            x = concatenate([x, residual], axis=3)
        x = self.conv_block(x, training=training)
        return x

class PixelClassifier(Layer):
    def __init__(
        self,
        nclasses: int,
        activation: bool = True,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)

        self.nclasses = nclasses

        self.output_layer = Conv2D(filters=nclasses, kernel_size=(1,1))
        self.batch_norm = BatchNormalization()

        if activation:
            if nclasses == 1:
                self.activation = nn.sigmoid
            else:
                self.activation = Softmax(axis=-1)
        else:
            self.activation = None

    def get_config(self) -> dict[str, any]:
        config = super().get_config()
        config.update({
            "nclasses": self.nclasses,
        })

        return config

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        output = self.output_layer(inputs)
        output = self.batch_norm(output, training=training)
        if self.activation:
            output = self.activation(output)
        return output

class LinearClassifier(Layer):
    def __init__(self, nclasses: int, classification: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)

        self.nclasses = nclasses

        self.flatten =  Flatten()
        self.output_layer = Dense(self.nclasses)
        self.batch_norm = BatchNormalization()
        self.classification = classification
        if self.classification:
            self.softmax = Softmax(-1)
        else:
            self.sigmoid = Activation(sigmoid)

    def get_config(self) -> dict[str, any]:
        config = super().get_config()
        config.update({
            "nclasses": self.nclasses,
            "classification": self.classification,
        })

        return config

    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor: #pylint: disable=unused-argument
        output = self.flatten(inputs)
        output = self.output_layer(output)
        output = self.batch_norm(output, training=training)
        if self.classification:
            output = self.softmax(output)
        else:
            output = self.sigmoid(output)
        return output


class ResidualBlock(Layer):
    def __init__(
        self,
        nfilters: int,
        padding: str = "same",
        initializer: str = "he_normal",
        l2: float = 1e-4,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)

        self.nfilters = nfilters
        self.padding = padding
        self.initializer = initializer
        self.l2 = l2

        self.relu_1 = Activation("relu")
        self.conv_1 = Conv2D(
            filters=nfilters,
            kernel_size=(3, 3),
            padding=padding,
            kernel_initializer=initializer,
            kernel_regularizer=L2(l2),
            bias_regularizer=L2(l2),
        )


        self.relu_2 = Activation("relu")
        self.conv_2 = Conv2D(
            filters=nfilters,
            kernel_size=(1, 1),
            padding=padding,
            kernel_initializer=initializer,
            kernel_regularizer=L2(l2),
            bias_regularizer=L2(l2),
        )

    def get_config(self) -> dict[str, any]:
        config = super().get_config()

        config.update({
            "nfilters": self.nfilters,
            "padding": self.padding,
            "initializer": self.initializer,
            "l2": self.l2,
        })

        return config

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.relu_1(inputs, training=training)
        x = self.conv_1(x, training=training)

        x = self.relu_2(x, training=training)
        x = self.conv_2(x, training=training)

        return x
