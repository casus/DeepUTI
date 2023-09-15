import time
from pathlib import Path

import click
import neptune.new as neptune
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.python.client import device_lib

try:
    from tensorflow.keras.optimizers.legacy import Adam
except ImportError:
    from tensorflow.keras.optimizers import Adam

from albumentations.core.serialization import load as load_albumentations_transform

from src.data.image_mask_datasets import ImageMaskRandomDataset, ImageMaskStaticDataset
from src.models.callbacks import GradualUnfreeze, unfreeze_all_layers
from src.models.layers import PixelClassifier
from src.models.losses import get_loss
from src.models.metrics import get_metric
from src.models.trainer import Trainer
from src.models.unet import add_classifier, create_unet_backbone
from src.utils.config import read_json_config


@click.command()
@click.argument("config_file_path", type=click.Path(exists=True))
@click.option("--seed", "-s", default=None, type=int)
def main(config_file_path, seed):
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

    if seed is None:
        seed = parameters["seed"]

    tf.random.set_seed(seed)

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
        run["model/parameters/seed"] = seed

    # Initialize model

    unet = create_unet_backbone(
        img_height=parameters["target_height"],
        img_width=parameters["target_width"],
        filters=parameters["bottleneck_size"],
        do=parameters["dropout"],
        l2=parameters["l2_regularization"],
        use_residuals=parameters["residual_connections"],
    )

    classifier = PixelClassifier(parameters["num_classes"])

    model = add_classifier(
        img_height=parameters["target_height"],
        img_width=parameters["target_width"],
        backbone=unet["backbone"],
        classifier=classifier,
    )

    if config["weights_path"] is not None:
        match config["restore"]:
            case "encoder":
                checkpoint = tf.train.Checkpoint(
                    encoder=unet["encoder"]
                )
            case "backbone":
                checkpoint = tf.train.Checkpoint(
                    backbone=unet["backbone"]
                )
            case "model":
                checkpoint = tf.train.Checkpoint(
                    model=model
                )
            case _:
                raise ValueError("Unknown part to restore")

        checkpoint_path = tf.train.latest_checkpoint(config["weights_path"])
        if checkpoint_path is None:
            raise RuntimeError(f"No checkpoint found in directory {config['weights_path']}")
        checkpoint.restore(checkpoint_path).expect_partial()

        unfreeze_all_layers(model)

    match config["freeze"]:
        case None:
            pass
        case "encoder":
            unet["encoder"].trainable = False
        case "backbone":
            unet["backbone"].trainable = False
        case "model":
            model.trainable = False
        case _:
            raise ValueError("Unknown part to freeze")

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
        encoder=unet["encoder"],
        backbone=unet["backbone"],
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

    # Set up callbacks

    callbacks = [
        GradualUnfreeze(
            model,
            start_after=config["unfreeze"]["start_after"],
            frequency=config["unfreeze"]["frequency"],
        ),
    ]

    trainer = Trainer(model, optimizer, loss, metrics, run, manager, callbacks)

    # Create datasets

    transform_train = load_albumentations_transform(
        config["train"]["transform_filepath"])
    image_mask_dataset_train = ImageMaskRandomDataset(
        Path(config["train"]["img_dir"]),
        Path(config["train"]["mask_dir"]),
        transform=transform_train,
        batch_size=config["train"]["batch_size"],
        epoch_size=parameters["samples_per_epoch"],
        num_classes=parameters["num_classes"],
        threshold_obj_perc=parameters["thresh_obj_perc"],
        max_iters=parameters["max_iters"],
        seed=seed,
        num_samples=parameters["num_samples"]
    )

    if use_neptune and parameters["num_samples"] is not None:
        run["train_samples"] = image_mask_dataset_train.img_names

    static_dataset_dir = config["validation"].get("static_dir", None)

    if static_dataset_dir is None:
        transform_val = load_albumentations_transform(
            config["validation"]["transform_filepath"])
        image_mask_dataset_val = ImageMaskRandomDataset(
            Path(config["validation"]["img_dir"]),
            Path(config["validation"]["mask_dir"]),
            transform=transform_val,
            batch_size=config["validation"]["batch_size"],
            epoch_size=parameters["samples_per_epoch"],
            num_classes=parameters["num_classes"],
            threshold_obj_perc=parameters["thresh_obj_perc"],
            max_iters=parameters["max_iters"],
            seed=seed,
        )
    else:
        image_mask_dataset_val = ImageMaskStaticDataset(
            static_dataset_dir
        )

    print(tf.shape(image_mask_dataset_train[0][0]))
    print(tf.shape(image_mask_dataset_val[0][0]))


    # Train model

    trainer.fit(
        parameters["epochs"],
        image_mask_dataset_train,
        image_mask_dataset_val,
    )

    if use_neptune:
        run.stop()

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
