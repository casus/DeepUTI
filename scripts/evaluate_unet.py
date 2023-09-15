import time
from pathlib import Path

import click
import tensorflow as tf
from albumentations.core.serialization import load as load_albumentations_transform
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib

try:
    from tensorflow.keras.optimizers.legacy import Adam
except ImportError:
    from tensorflow.keras.optimizers import Adam

from src.data.image_mask_datasets import ImageMaskDataset
from src.models.layers import PixelClassifier
from src.models.losses import dice_bce_loss
from src.models.metrics import dice_coeff
from src.models.unet import add_classifier, create_unet_backbone
from src.utils.config import read_json_config
from src.utils.unet import evaluate_model, find_best_checkpoint


@click.command()
@click.argument("config_file_path", type=click.Path(exists=True))
def main(config_file_path):
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
    )

    classifier = PixelClassifier(parameters["num_classes"])

    model = add_classifier(
        img_height=parameters["target_height"],
        img_width=parameters["target_width"],
        backbone=unet["backbone"],
        classifier=classifier,
    )

    if config["checkpoint_path"] is not None:
        checkpoint_path = Path(results_dir / config["checkpoint_path"])
        if checkpoint_path.is_dir():
            checkpoint_path = find_best_checkpoint(checkpoint_path)
        model.load_weights(checkpoint_path)
    else:
        raise RuntimeError("Weights must be provided for the model to be evaluated")

    model.compile(
        optimizer=Adam(learning_rate=parameters["start_lr"]),
        loss=dice_bce_loss,
        metrics=[dice_coeff]
    )

    model.summary()


    transform = load_albumentations_transform(config["test"]["transform_filepath"])

    image_mask_dataset_val = ImageMaskDataset(
        Path(config["validation"]["img_dir"]),
        Path(config["validation"]["mask_dir"]),
        transform=transform,
        num_classes=parameters["num_classes"],
        batch_size=1,
        shuffle=False,
    )

    metrics = evaluate_model(
        model,
        image_mask_dataset_val,
        parameters["target_height"],
        parameters["target_width"],
    )

    print(f"Validation metrics: {metrics}")

    image_mask_dataset_test = ImageMaskDataset(
        Path(config["test"]["img_dir"]),
        Path(config["test"]["mask_dir"]),
        transform=transform,
        num_classes=parameters["num_classes"],
        batch_size=1,
        shuffle=False,
    )

    metrics = evaluate_model(
        model,
        image_mask_dataset_test,
        parameters["target_height"],
        parameters["target_width"],
    )

    print(f"Test metrics: {metrics}")

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
