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

from src.data.image_datasets import PretextRandomDataset
from src.data.pretext_tasks import get_pretext_task
from src.models.layers import LinearClassifier
from src.models.losses import get_loss
from src.models.metrics import get_metric
from src.models.modules import create_encoder
from src.models.trainer import Trainer
from src.models.unet import add_classifier
from src.utils.config import read_json_config


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
        run = neptune.init(api_token=auth_keys["neptune_api_token"], **config["neptune_args"])
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

    if config["checkpoint_path"] is not None:
        log_dir = Path(config["checkpoint_path"])
    else:
        log_dir = model_dir / f"{running_time}"
    log_dir.mkdir(parents=True, exist_ok=True)

    if use_neptune:
        run["model/parameters"] = parameters
        run["model/parameters/running_time"] = running_time
        run["model/parameters/model_dir"] = model_dir
        run["model/parameters/log_dir"] = log_dir


    pretext_task_creator = get_pretext_task(
        task=config["pretext_task"],
        seed=rng,
        **config["pretext_kwargs"],
    )

    # Initialize model

    encoder = create_encoder(
        img_height=parameters["target_height"],
        img_width=parameters["target_width"],
        filters=parameters["bottleneck_size"],
        do=parameters["dropout"],
        l2=parameters["l2_regularization"],
        )

    classifier = LinearClassifier(pretext_task_creator.output_len, pretext_task_creator.classification)

    model = add_classifier(
        img_height=parameters["target_height"],
        img_width=parameters["target_width"],
        backbone=encoder,
        classifier=classifier,
    )

    model.summary()

    # Configure trainer

    optimizer = Adam(ExponentialDecay(
        initial_learning_rate = parameters["start_lr"],
        decay_steps = parameters["samples_per_epoch"]*parameters["scheduler"]["step_size"],
        decay_rate = parameters["scheduler"]["gamma"],
        ))

    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model,
        encoder=encoder,
    )

    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=log_dir,
        max_to_keep=5,
    )

    if config["checkpoint_path"] is not None:
        manager.restore_or_initialize()

    loss = get_loss(parameters["loss_function"])

    metrics = [get_metric(metric) for metric in parameters["metrics"]]

    trainer = Trainer(model, optimizer, loss, metrics, run, manager)

    transform_train = load_albumentations_transform(config["train"]["transform_filepath"])
    pretext_dataset_train = PretextRandomDataset(
        Path(config["train"]["img_dir"]),
        pretext_task_creator,
        transform=transform_train,
        batch_size=config["train"]["batch_size"],
        epoch_size=parameters["samples_per_epoch"],
        seed=rng,
    )

    transform_val = load_albumentations_transform(config["validation"]["transform_filepath"])
    pretext_dataset_val = PretextRandomDataset(
        Path(config["validation"]["img_dir"]),
        pretext_task_creator,
        transform=transform_val,
        batch_size=config["validation"]["batch_size"],
        epoch_size=parameters["samples_per_epoch"],
        seed=rng,
    )

    print(tf.shape(pretext_dataset_train[0][0]))
    print(tf.shape(pretext_dataset_train[0][1]))

    trainer.fit(
        parameters["epochs"],
        pretext_dataset_train,
        pretext_dataset_val,
    )

    if use_neptune:
        run.stop()

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
