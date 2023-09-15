import time
from pathlib import Path

import click
import neptune.new as neptune
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.python.client import device_lib

try:
    from tensorflow.keras.optimizers.legacy import Adam
except ImportError:
    from tensorflow.keras.optimizers import Adam

from albumentations.core.serialization import load as load_albumentations_transform

from src.data.image_datasets import ImageRandomDataset
from src.models.layers import PixelClassifier
from src.models.losses import DiscriminatorLoss, GeneratorLoss
from src.models.metrics import get_metric
from src.models.modules import create_encoder
from src.models.trainer import AutoEncoderAdversarialTrainer
from src.models.unet import add_classifier, create_unet_backbone
from src.utils.config import read_json_config
from src.utils.unet import find_best_checkpoint


@click.command()
@click.argument("config_file_path", type=click.Path(exists=True))
def main(config_file_path):
    config = read_json_config(config_file_path)
    auth_keys = read_json_config(config["auth_config_path"])

    if config["results_dir"] is not None:
        results_dir = Path(config["results_dir"])
    else:
        results_dir = Path(__file__).parents[1]

    use_neptune = config["use_neptune"]

    if use_neptune:
        run = neptune.init_run(api_token=auth_keys["neptune_api_token"], **config["neptune_args"])
        run["config.json"].upload(config_file_path)
    else:
        run = None

    # code to check for GPU
    print("Num CPUs Available: ", len(tf.config.list_physical_devices("CPU")))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    print(device_lib.list_local_devices())

    # TF dimension ordering in this code
    K.set_image_data_format("channels_last")

    parameters = config["parameters"]

    rng = np.random.default_rng(parameters["seed"])
    tf.random.set_seed(parameters["seed"])

    running_time = time.strftime("%b-%d-%Y_%H-%M")
    model_dir = results_dir / "model"
    log_dir = model_dir / f"{running_time}"
    log_dir.mkdir(parents=True, exist_ok=True)

    if use_neptune:
        run["model/parameters"] = parameters
        run["model/parameters/running_time"] = running_time
        run["model/parameters/model_dir"] = model_dir
        run["model/parameters/log_dir"] = log_dir

    # Initialize generator

    unet = create_unet_backbone(
        img_height=parameters["target_height"],
        img_width=parameters["target_width"],
        filters=parameters["bottleneck_size"],
        do=parameters["dropout"],
        l2=parameters["l2_regularization"],
        use_residuals=False,
    )

    generator_classifier = PixelClassifier(1, activation=False)

    generator = add_classifier(
        img_height=parameters["target_height"],
        img_width=parameters["target_width"],
        backbone=unet["backbone"],
        classifier=generator_classifier,
    )

    if config["checkpoint_path"] is not None:
        checkpoint_path = Path(results_dir / config["checkpoint_path"])
        if checkpoint_path.is_dir():
            checkpoint_path = find_best_checkpoint(checkpoint_path)
        generator.load_weights(checkpoint_path)

    generator.summary()

    # Initialize discriminator

    discriminator_encoder = create_encoder(
        img_height=parameters["target_height"],
        img_width=parameters["target_width"],
        filters=2,
        do=parameters["dropout"],
        l2=parameters["l2_regularization"],
    )

    discriminator_classifier = PixelClassifier(1, activation=True)

    discriminator = add_classifier(
        img_height=parameters["target_height"],
        img_width=parameters["target_width"],
        backbone=discriminator_encoder,
        classifier=discriminator_classifier,
    )


    # Configure trainer

    generator_optimizer = Adam(ExponentialDecay(
        initial_learning_rate = parameters["start_lr"],
        decay_steps = parameters["samples_per_epoch"]*parameters["scheduler"]["step_size"],
        decay_rate = parameters["scheduler"]["gamma"],
        ))

    generator_loss = GeneratorLoss()

    discriminator_optimizer = Adam(ExponentialDecay(
        initial_learning_rate = parameters["start_lr"],
        decay_steps = parameters["samples_per_epoch"]*parameters["scheduler"]["step_size"],
        decay_rate = parameters["scheduler"]["gamma"],
        ))

    discriminator_loss = DiscriminatorLoss()


    metrics = [get_metric(metric) for metric in parameters["metrics"]]

    checkpoint = tf.train.Checkpoint(
        generator_optimier=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        encoder=unet["encoder"],
        backbone=unet["backbone"],
        generator=generator,
        discriminator=discriminator
    )

    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=log_dir,
        max_to_keep=5,
    )

    trainer = AutoEncoderAdversarialTrainer(
        generator,
        generator_optimizer,
        generator_loss,
        discriminator,
        discriminator_optimizer,
        discriminator_loss,
        metrics,
        run,
        manager,
    )


    # Create datasets

    transform_train = load_albumentations_transform(
        config["train"]["transform_filepath"])
    image_mask_dataset_train = ImageRandomDataset(
        Path(config["train"]["img_dir"]),
        transform=transform_train,
        batch_size=config["train"]["batch_size"],
        epoch_size=parameters["samples_per_epoch"],
        seed=rng,
    )

    transform_val = load_albumentations_transform(
        config["validation"]["transform_filepath"])
    image_mask_dataset_val = ImageRandomDataset(
        Path(config["validation"]["img_dir"]),
        transform=transform_val,
        batch_size=config["validation"]["batch_size"],
        epoch_size=parameters["samples_per_epoch"],
        seed=rng,
    )

    print(tf.shape(image_mask_dataset_train[0][0]))
    print(tf.shape(image_mask_dataset_val[0][0]))

    # Train model

    trainer.fit(
        parameters["epochs"],
        image_mask_dataset_train,
        image_mask_dataset_val,
    )

    run.stop()

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
