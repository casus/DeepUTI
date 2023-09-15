import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import (
    BinaryCrossentropy, CategoricalCrossentropy, Loss, MeanAbsoluteError, MeanSquaredError,
    Reduction
)

from src.models.metrics import dice_coeff


def get_loss(
    loss_name: str,
    **kwargs,
    ) -> Loss:
    match loss_name.lower():
        case "dice":
            return DiceLoss(**kwargs)
        case "dicecce" | "dicecategoricalcrossentropy":
            return DiceCrossentropyLoss(binary=False)
        case "dicebce" | "dicebinarycrossentropy":
            return DiceCrossentropyLoss(binary=True)
        case "mse" | "meansquarederror":
            return MeanSquaredError()
        case "bce" | "binarycrossentropy":
            return BinaryCrossentropy()
        case "cce" | "categoricalcrossentropy":
            return CategoricalCrossentropy()
        case _:
            raise ValueError(f"Unknown loss: {loss_name}")

class DiceLoss(Loss):
    def __init__(self, smooth: float = 1.):
        self.smooth = smooth

    def __call__(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        ) -> tf.Tensor:
        """ Calculate dice loss.

            For binary segmentation n_classes should be equal to 1, and not ommited.

        Args:
            y_true (tf.Tensor): Ground truth values.            shape = [batch_size, W, H, n_classes]
            y_pred (tf.Tensor): Predicted values.               shape = [batch_size, W, H, n_classes]

        Returns:
            tf.Tensor: Dice loss value.                         shape = [batch_size]
        """
        dice_score = dice_coeff(y_true, y_pred, smooth=self.smooth)

        loss = tf.ones_like(dice_score) - dice_score
        return loss

class DiceCrossentropyLoss(Loss):
    def __init__(self, binary=False, smooth: float = 1.) -> None:
        if binary:
            self.crossentropy = BinaryCrossentropy(reduction=Reduction.NONE)
        else:
            self.crossentropy = CategoricalCrossentropy(reduction=Reduction.NONE)

        self.dice_loss = DiceLoss(smooth)

    def __call__(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        ) -> tf.Tensor:
        """ Calculate combined dice and crossentropy losses.

            For binary segmentation n_classes should be equal to 1, and not ommited.

        Args:
            y_true (tf.Tensor): Ground truth values.            shape = [batch_size, W, H, n_classes]
            y_pred (tf.Tensor): Predicted values.               shape = [batch_size, W, H, n_classes]

        Returns:
            tf.Tensor: Dice cce loss value.                     shape = [batch_size]
        """

        # Crossentropy:                 [batch_size, W, H, n_classes] -> [batch_size, W, H]
        crossentropy = self.crossentropy(y_true, y_pred)

        # Mean reduction:               [batch_size, W, H] -> [1]
        crossentropy = K.mean(crossentropy)

        # Dice_loss:                    [batch_size, W, H, n_classes] -> [n_classes]
        dice = self.dice_loss(y_true, y_pred)

        # Mean reduction:               [n_classes] -> [1]
        dice = K.mean(dice)

        return crossentropy + dice

class DiscriminatorLoss(Loss):
    def __init__(self):
        self.binary_crossentropy = BinaryCrossentropy()

    def __call__(
        self,
        y_true: tf.Tensor,
        y_gen: tf.Tensor,
    ):
        true_loss = self.binary_crossentropy(tf.ones_like(y_true), y_true)
        gen_loss = self.binary_crossentropy(tf.zeros_like(y_gen), y_gen)

        return true_loss + gen_loss

class GeneratorLoss(Loss):
    def __init__(self):
        self.mae = MeanAbsoluteError()
        self.binary_crossentropy = BinaryCrossentropy()

    def __call__(
        self,
        y_true: tf.Tensor,
        y_gen: tf.Tensor,
        y_gen_disc: tf.Tensor,
    ):
        adversarial_loss = self.binary_crossentropy(tf.ones_like(y_gen_disc), y_gen_disc)
        l1_loss = self.mae(y_true, y_gen)

        return adversarial_loss + l1_loss, adversarial_loss, l1_loss


def jaccard_distance(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: int = 1) -> tf.Tensor:
    """ Calculate jaccard distance.

    Args:
        y_true (tf.Tensor): Ground truth values.    shape = [batch_size, d0, ..., dN]
        y_pred (tf.Tensor): Predicted values.       shape = [batch_size, d0, ..., dN]
        smooth (int): Smoothing factor to avoid division by zero.

    Returns:
        tf.Tensor: Jaccard distance value.          shape = [batch_size, d0, ..., dN-1]
    """
    # intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    # sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    # return (intersection + smooth) / (sum_ - intersection + smooth)
    raise DeprecationWarning("Needs refactoring")
