import time
from pathlib import Path

import click
import numpy as np
import tensorflow as tf
from albumentations.core.serialization import load as load_albumentations_transform
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib

from src.data.image_mask_datasets import ImageMaskDataset, ImageMaskStaticDataset
from src.models.layers import PixelClassifier
from src.models.losses import get_loss
from src.models.metrics import get_metric
from src.models.trainer import Trainer
from src.models.unet import add_classifier, create_unet_backbone
from src.utils.config import read_json_config
from src.utils.unet import evaluate_model


@click.command()
@click.argument("config_file_path", type=click.Path(exists=True))
@click.argument("checkpoints_dir", type=click.Path(exists=True))
def main(config_file_path, checkpoints_dir):
    config = read_json_config(config_file_path)

    if config["results_dir"] is not None:
        results_dir = Path(config["results_dir"])
    else:
        results_dir = Path(__file__).parents[1]

    # code to check for GPU
    print("Num CPUs Available: ", len(tf.config.list_physical_devices("CPU")))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    print(device_lib.list_local_devices())

    # TF dimension ordering in this code
    K.set_image_data_format("channels_last")

    parameters = config["parameters"]

    tf.random.set_seed(parameters["seed"])

    running_time = time.strftime("%b-%d-%Y_%H-%M")
    model_dir = results_dir / "model"

    log_dir = model_dir / "logs"/ f"{running_time}"
    log_dir.mkdir(parents=True, exist_ok=True)

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

    checkpoint = tf.train.Checkpoint(
        model=model
    )


    # Set up trainer
    loss = get_loss(parameters["loss_function"])
    metrics = [get_metric(metric) for metric in parameters["metrics"]]
    trainer = Trainer(model, None, loss, metrics)

    transform_val = load_albumentations_transform(
        config["validation"]["transform_filepath"])
    image_mask_dataset_val = ImageMaskDataset(
        Path(config["validation"]["img_dir"]),
        Path(config["validation"]["mask_dir"]),
        num_classes=parameters["num_classes"],
        transform=transform_val,
        batch_size=1,
        shuffle=False,
    )

    transform_test = load_albumentations_transform(
        config["test"]["transform_filepath"])
    image_mask_dataset_test = ImageMaskDataset(
        Path(config["test"]["img_dir"]),
        Path(config["test"]["mask_dir"]),
        num_classes=parameters["num_classes"],
        transform=transform_test,
        batch_size=1,
        shuffle=False,
    )

    val_dice, test_dice = [], []
#    for path in sorted(Path(checkpoints_dir).iterdir()):
    path = Path(checkpoints_dir)
    print(path)

    checkpoint_path = tf.train.latest_checkpoint(path)
    if checkpoint_path is None:
        raise RuntimeError(f"No checkpoint found in directory {path}")
    print(checkpoint_path)

    checkpoint.restore(checkpoint_path).expect_partial()

    metrics = trainer.evaluate(image_mask_dataset_val)
    val_dice.append(metrics["dice_coeff"].numpy())
    print(metrics["dice_coeff"].numpy())

    metrics = trainer.evaluate(image_mask_dataset_test)
    test_dice.append(metrics["dice_coeff"].numpy())
    print(metrics["dice_coeff"].numpy())

    print(f"Val dice: mean - {np.mean(val_dice)}, std - {np.std(val_dice)}")
    print(f"Test dice: mean - {np.mean(test_dice)}, std - {np.std(test_dice)}")


    # transform = load_albumentations_transform(config["test"]["transform_filepath"])

    # metrics = evaluate_model(
    #     model,
    #     image_mask_dataset_val,
    #     parameters["target_height"],
    #     parameters["target_width"],
    # )

    # print(f"Validation metrics: {metrics}")

    # image_mask_dataset_test = ImageMaskDataset(
    #     Path(config["test"]["img_dir"]),
    #     Path(config["test"]["mask_dir"]),
    #     transform=transform,
    #     num_classes=parameters["num_classes"],
    #     batch_size=1,
    #     shuffle=False,
    # )

    # metrics = evaluate_model(
    #     model,
    #     image_mask_dataset_test,
    #     parameters["target_height"],
    #     parameters["target_width"],
    # )

    # print(f"Test metrics: {metrics}")

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
