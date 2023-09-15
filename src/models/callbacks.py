from pathlib import Path
from typing import Callable, Union

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback


class ModelPartCheckpoint(Callback):
    """
    Callback allowing for saving of the weights of any model.
    As opposed to keras' ModelCheckpoint, it takes the model to save as an argument.
    This allows for training the whole model while only saving one part of it.
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        model: Model,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        super().__init__()

        self.filepath = filepath
        self.encoder = model

        self.monitor = monitor
        self.monitor_op: Callable

        if mode == "min":
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError("Unknown mode")

    def on_epoch_end(self, epoch, logs=None) -> None:  # pylint: disable=unused-argument
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):
            self.best = current
            self.encoder.save_weights(self.filepath, overwrite=True)


class UnfreezeModel(Callback):
    def __init__(
        self,
        model: Model,
        epoch: int,
    ):
        self.model = model
        self.epoch = epoch

    def on_epoch_begin(self, epoch, logs=None) -> None:  # pylint: disable=unused-argument
        if epoch == self.epoch:
            self.model.trainable = True


def _get_trainable_layers(module, only_frozen=True):
    layers = []
    for layer in module.layers:
        if hasattr(layer, "layers"):
            layers.extend(_get_trainable_layers(layer))
        elif len(layer.weights) > 0:
            if not (only_frozen and layer.trainable):
                layers.append(layer)
    return layers


def unfreeze_all_layers(module):
    for layer in module.layers:
        if hasattr(layer, "trainable"):
            layer.trainable = True
        if hasattr(layer, "layers"):
            unfreeze_all_layers(layer)


def _count_trainable_params(module):
    return np.sum([K.count_params(p) for p in module.trainable_weights])


class GradualUnfreeze(Callback):
    def __init__(
        self,
        model: Model,
        start_after: int,
        frequency: int,
    ):
        self.model = model
        self.trainable_layers = _get_trainable_layers(model)

        unfreeze_all_layers(model)

        for layer in self.trainable_layers:
            layer.trainable = False

        print("Unfreeze callback initialized")
        print(
            f"Number of trainable params: {_count_trainable_params(self.model)}\n")

        self.start_after = start_after
        self.frequency = frequency

    def _unfreeze_top_layer(self):
        layer = self.trainable_layers.pop()
        layer.trainable = True

        print(f"Layer {layer.name} unfrozen")
        print(
            f"Number of trainable params: {_count_trainable_params(self.model)}\n")

    def on_epoch_begin(self, epoch, logs=None) -> None:  # pylint: disable=unused-argument
        if (
            epoch >= self.start_after
            and epoch % self.frequency == 0
            and self.trainable_layers
        ):
            self._unfreeze_top_layer()
