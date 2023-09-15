import time
from pathlib import Path

import click
import neptune.new as neptune
import numpy as np
import tensorflow as tf
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.client import device_lib

try:
    from tensorflow.keras.optimizers.legacy import Adam
except ImportError:
    from tensorflow.keras.optimizers import Adam

from albumentations.core.serialization import load as load_albumentations_transform

from src.data.image_mask_datasets import ImageMaskRandomDataset
from src.models.hydranet import create_hydranet
from src.models.losses import dice_bce_loss
from src.models.metrics import dice_coeff
from src.utils.config import read_json_config
from src.utils.train import get_scheduler
from src.utils.visualization import plot_metrics


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
        run = neptune.init(
            api_token=auth_keys["neptune_api_token"], **config["neptune_args"])
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
    log_dir = model_dir / "logs" / f"{running_time}"
    log_dir.mkdir(parents=True, exist_ok=True)

    if use_neptune:
        run["model/parameters"] = parameters
        run["model/parameters/running_time"] = running_time
        run["model/parameters/model_dir"] = model_dir
        run["model/parameters/log_dir"] = log_dir

    # Initialize model

    model = create_hydranet(
        img_height=parameters["target_height"],
        img_width=parameters["target_width"],
        filters=parameters["bottleneck_size"],
        do=parameters["dropout"],
        l2=parameters["l2_regularization"],
        num_classes=parameters["num_classes"]
    )

    model.compile(
        optimizer=Adam(learning_rate=parameters["start_lr"]),
        loss=dice_bce_loss,
        metrics=[dice_coeff]
    )

    model.summary()

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
        seed=rng,
    )

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
        seed=rng,
    )

    print(tf.shape(image_mask_dataset_train[0][0]))
    print(tf.shape(image_mask_dataset_val[0][0]))

    # Set up callbacks

    tensorboard = TensorBoard(
        log_dir=model_dir / f"{running_time}",
        histogram_freq=1,
    )

    scheduler = get_scheduler(**parameters["scheduler"])

    callbacks_list = [tensorboard, scheduler]

    if config["save_weights"] is not None:
        if config["save_weights"] == "all":
            checkpoint_suffix = ".epoch{epoch:03d}.hdf5"
        elif config["save_weights"] == "best":
            checkpoint_suffix = ".best.hdf5"
        else:
            raise ValueError(
                f"""Unknown options for save_weights parameter.
                Got: {config["save_weights"]}, expected one of ["all", "best", None]."""
            )
        model_checkpoint = ModelCheckpoint(
            filepath=log_dir / ("model" + checkpoint_suffix),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min",
        )

        callbacks_list.extend([model_checkpoint])

    if use_neptune:
        callbacks_list.append(NeptuneCallback(run=run))

    print(f"start tensorboard, cmd: tensorboard --logdir='{log_dir}'")

    # Train model

    history = model.fit(
        image_mask_dataset_train,
        epochs=parameters["epochs"],
        validation_freq=parameters["val_freq"],
        validation_data=image_mask_dataset_val,
        callbacks=callbacks_list
    )

    plot_metrics(history, "dice_coeff", log_dir / "dice_coeff.svg")
    plot_metrics(history, "loss", log_dir / "loss.svg")

    if use_neptune:
        run["model/evaluation/dice_coeff_plot"].upload(str(log_dir / "dice_coeff.svg"))
        run["model/evaluation/loss_plot"].upload(str(log_dir / "loss.svg"))

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
