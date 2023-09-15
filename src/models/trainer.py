import io
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from icecream import ic
from neptune.new import Run as NeptuneRun
from PIL import Image
from tensorflow import keras

from src.data.image_datasets import ImageAbstractDataset as ImageDataset
from src.utils.data import combine_patches, split_into_patches

Mode = Enum("Mode", ["TRAIN", "VAL", "TEST"])

#pylint: disable=not-callable

class BaseTrainer(ABC):
    def __init__(
        self,
        metrics: Sequence[tf.keras.metrics.Metric],
        neptune_run: Optional[NeptuneRun] = None,
        checkpoint_manager: Optional[tf.train.CheckpointManager] = None,
        callbacks: Sequence[keras.callbacks.Callback] = None,
    ) -> None:

        self.metrics = metrics
        self.neptune_run = neptune_run
        self.checkpoint_manager = checkpoint_manager
        self.callbacks = keras.callbacks.CallbackList(callbacks, add_history=True)

        self.total_loss_tracker: keras.metrics.Metric = keras.metrics.Mean(name="total_loss")

        self.best_loss = np.inf

        self.epoch = 0

    @abstractmethod
    def train_step(self, data: tuple[np.ndarray, np.ndarray]):
        raise NotImplementedError

    @abstractmethod
    def test_step(self, data: tuple[np.ndarray, np.ndarray]):
        raise NotImplementedError


    def _report_metrics(self, mode_name: str):
        loss = self.total_loss_tracker.result().numpy()
        for metric in self.metrics:
            print(f"{mode_name.capitalize()} {metric.name}: {metric.result()}")
        if self.neptune_run:
            self.neptune_run[f"{mode_name.lower()}/epoch/loss"].log(loss)
            for metric in self.metrics:
                self.neptune_run[f"{mode_name.lower()}/epoch/{metric.name}"].log(metric.result())


    def _reset_metrics(self):
        self.total_loss_tracker.reset_state()
        for metric in self.metrics:
            metric.reset_state()


    def _run_epoch(self, mode: Mode, dataset: keras.utils.Sequence) -> float:

        for batch, data in enumerate(dataset):
            if mode == Mode.TRAIN:
                self.callbacks.on_train_batch_begin(batch)
                self.train_step(data)
                self.callbacks.on_train_batch_end(batch)
            else:
                self.callbacks.on_test_batch_begin(batch)
                self.test_step(data)
                self.callbacks.on_test_batch_end(batch)

        if hasattr(self, "_plot_sample"):
            if self.epoch % 5 == 0:
                self._plot_sample(mode.name, dataset[0])

        loss = self.total_loss_tracker.result().numpy()

        self._report_metrics(mode.name)
        self._reset_metrics()

        return loss


    def fit(
        self,
        epochs: int,
        train_dataset: keras.utils.Sequence,
        val_dataset: keras.utils.Sequence,
        ):
        self.callbacks.on_train_begin()
        for epoch in range(epochs):
            self.epoch += 1

            self.callbacks.on_epoch_begin(epoch)
            print(f"Epoch #{epoch}\n")

            train_loss = self._run_epoch(Mode.TRAIN, train_dataset)
            print(f"Train loss: {train_loss}\n")

            val_loss = self._run_epoch(Mode.VAL, val_dataset)
            print(f"Val loss: {val_loss}\n")

            if self.best_loss > val_loss:
                self.best_loss = val_loss
                if self.checkpoint_manager is not None:
                    self.checkpoint_manager.save(checkpoint_number=epoch)

            self.callbacks.on_epoch_end(epoch)

    def evaluate(self, test_dataset: keras.utils.Sequence, patch_height=256, patch_width=256):
        for img, mask in test_dataset:
            img_patches, breakpoints = split_into_patches(img[0], patch_height, patch_width)


            pred_patches = self.model(img_patches, training=False)

            pred = combine_patches(pred_patches, breakpoints)
            pred = pred[np.newaxis]


            for metric in self.metrics:
                metric.update_state(mask, pred)

        results = {metric.name: metric.result() for metric in self.metrics}
        self._reset_metrics()

        return results

class Trainer(BaseTrainer):
    def __init__(
        self,
        model: tf.keras.models.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss: tf.keras.losses.Loss,
        metrics: Sequence[tf.keras.metrics.Metric],
        neptune_run: Optional[NeptuneRun] = None,
        checkpoint_manager: Optional[tf.train.CheckpointManager] = None,
        callbacks: Sequence[keras.callbacks.Callback] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

        super().__init__(metrics, neptune_run, checkpoint_manager, callbacks)

        self.callbacks.set_model(self.model)


    def train_step(self, data: tuple[np.ndarray, np.ndarray]):
        x, y_true = data

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss_value = self.loss(y_true, y_pred)

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.total_loss_tracker.update_state(loss_value)
        for metric in self.metrics:
            metric.update_state(y_true, y_pred)


    def test_step(self, data: tuple[np.ndarray, np.ndarray]):
        x, y_true = data

        y_pred = self.model(x, training=False)
        loss_value = self.loss(y_true, y_pred)

        self.total_loss_tracker.update_state(loss_value)

        for metric in self.metrics:
            metric.update_state(y_true, y_pred)

class VAETrainer(BaseTrainer):
    def __init__(
        self,
        encoder: tf.keras.models.Model,
        decoder: tf.keras.models.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        neptune_run: Optional[NeptuneRun] = None,
        checkpoint_manager: Optional[tf.train.CheckpointManager] = None,
        callbacks: Sequence[keras.callbacks.Callback] = None,
        ) -> None:

        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer

        self.reconstruction_loss = keras.losses.MeanAbsoluteError()
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")

        self.kl_loss = keras.losses.KLDivergence()
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        metrics = [self.reconstruction_loss_tracker, self.kl_loss_tracker]

        super().__init__(metrics, neptune_run, checkpoint_manager, callbacks)

        self.callbacks.set_model(self.encoder)


    def train_step(self, data: np.ndarray):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = keras.losses.binary_crossentropy(data, reconstruction)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss, axis=(1, 2)))

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = reconstruction_loss + kl_loss

        trainable_weights = self.encoder.trainable_weights + self.decoder.trainable_weights
        grads = tape.gradient(total_loss, trainable_weights)
        self.optimizer.apply_gradients(zip(grads, trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)


    def test_step(self, data: np.ndarray):
        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)

        reconstruction_loss = keras.losses.binary_crossentropy(data, reconstruction)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss, axis=(1, 2)))

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        total_loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

    def _plot_sample(self, mode: str, data: np.ndarray, num_samples=5):
        indices = np.arange(len(data))
        indices = np.random.choice(indices, num_samples, replace=False)
        data = data[indices]

        _, _, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)

        fig, axes = plt.subplots(2, num_samples, figsize=(5*num_samples, 10))

        for idx in range(num_samples):
            axes[0, idx].imshow(data[idx], cmap="gray")
            axes[1, idx].imshow(reconstruction[idx], cmap="gray")

        fig.tight_layout()

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png")
        img = Image.open(img_buf)

        ic(img.size)

        if self.neptune_run is not None:
            self.neptune_run[f"{mode.lower()}/#{self.epoch}"].upload(img.resize((750, 300)))


class AutoEncoderTrainer:
    def __init__(
        self,
        generator: tf.keras.models.Model,
        generator_optimizer: tf.keras.optimizers.Optimizer,
        generator_loss: tf.keras.losses.Loss,
        metrics: Sequence[tf.keras.metrics.Metric],
        neptune_run: Optional[NeptuneRun] = None,
        checkpoint_manager: Optional[tf.train.CheckpointManager] = None,
        callbacks: Sequence[keras.callbacks.Callback] = None,
    ) -> None:

        self.generator = generator
        self.generator_optimizer = generator_optimizer
        self.generator_loss = generator_loss

        self.metrics = metrics
        self.neptune_run = neptune_run
        self.checkpoint_manager = checkpoint_manager

        self.callbacks = keras.callbacks.CallbackList(callbacks, add_history=True, model=generator)

        self.best_loss = np.inf

    def _run_epoch(self, mode: Mode, dataset: ImageDataset):

        loss_values = defaultdict(list)
        for batch, imgs in enumerate(dataset):
            if mode == Mode.TRAIN:
                self.callbacks.on_train_batch_begin(batch)
                losses = self.train_step(imgs)
                self.callbacks.on_train_batch_end(batch)
            else:
                self.callbacks.on_test_batch_begin(batch)
                losses = self.test_step(imgs)
                self.callbacks.on_test_batch_end(batch)

            for name, value in losses.items():
                loss_values[name].append(value)

        if self.neptune_run:
            for name, values in loss_values.items():
                self.neptune_run[f"{mode.name.lower()}/epoch/{name}"].log(np.mean(values))
            for metric in self.metrics:
                self.neptune_run[f"{mode.name.lower()}/epoch/{metric.name}"].log(metric.result())

        for metric in self.metrics:
            metric.reset_state()

        return np.mean(loss_values["generator_loss"])

    def train_step(self, imgs: np.ndarray):

        with tf.GradientTape() as gen_tape:
            gen_output = self.generator(imgs, training=True)

            gen_loss = self.generator_loss(imgs, gen_output)

        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)


        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )

        for metric in self.metrics:
            metric.update_state(imgs, gen_output)

        return {
            "generator_loss": gen_loss,
        }

    def test_step(self, imgs: np.ndarray):
        gen_output = self.generator(imgs, training=False)

        gen_loss = self.generator_loss(imgs, gen_output)

        for metric in self.metrics:
            metric.update_state(imgs, gen_output)

        return {
            "generator_loss": gen_loss,
        }


    def fit(
        self,
        epochs: int,
        train_dataset: keras.utils.Sequence,
        val_dataset: keras.utils.Sequence,
        ):
        self.callbacks.on_train_begin()
        for epoch in range(epochs):
            self.callbacks.on_epoch_begin(epoch)
            print(f"Epoch #{epoch}")

            train_loss = self._run_epoch(Mode.TRAIN, train_dataset)
            print(f"Train loss: {train_loss}")

            if self.best_loss > train_loss:
                self.best_loss = train_loss
                if self.checkpoint_manager is not None:
                    self.checkpoint_manager.save(checkpoint_number=epoch)

            val_loss = self._run_epoch(Mode.VAL, val_dataset)
            print(f"Val loss: {val_loss}")
            self.callbacks.on_epoch_end(epoch)


class AutoEncoderAdversarialTrainer(AutoEncoderTrainer):
    def __init__(
        self,
        generator: tf.keras.models.Model,
        generator_optimizer: tf.keras.optimizers.Optimizer,
        generator_loss: tf.keras.losses.Loss,
        discriminator: tf.keras.models.Model,
        discriminator_optimizer: tf.keras.optimizers.Optimizer,
        discriminator_loss: tf.keras.losses.Loss,
        metrics: Sequence[tf.keras.metrics.Metric],
        neptune_run: Optional[NeptuneRun] = None,
        checkpoint_manager: Optional[tf.train.CheckpointManager] = None,
        callbacks: Sequence[keras.callbacks.Callback] = None,
    ) -> None:

        super().__init__(
            generator,
            generator_optimizer,
            generator_loss,
            metrics,
            neptune_run,
            checkpoint_manager,
            callbacks,
        )

        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_loss = discriminator_loss

    def train_step(self, imgs: np.ndarray):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(imgs, training=True)

            disc_real_output = self.discriminator(imgs, training=True)
            disc_generated_output = self.discriminator(gen_output, training=True)

            gen_total_loss, adversarial_loss, l1_loss = self.generator_loss(imgs, gen_output, disc_generated_output)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)


        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        for metric in self.metrics:
            metric.update_state(imgs, gen_output)

        return {
            "generator_loss": gen_total_loss,
            "generator_adversarial_loss": adversarial_loss,
            "generator_l1_loss": l1_loss,
            "discriminator_loss": disc_loss,
        }

    def test_step(self, imgs: np.ndarray):
        gen_output = self.generator(imgs, training=False)

        disc_real_output = self.discriminator(imgs, training=False)
        disc_generated_output = self.discriminator(gen_output, training=False)

        gen_total_loss, adversarial_loss, l1_loss = self.generator_loss(imgs, gen_output, disc_generated_output)
        disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        for metric in self.metrics:
            metric.update_state(imgs, gen_output)

        return {
            "generator_loss": gen_total_loss,
            "generator_adversarial_loss": adversarial_loss,
            "generator_l1_loss": l1_loss,
            "discriminator_loss": disc_loss,
        }
