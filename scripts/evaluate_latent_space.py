import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tensorflow as tf
from albumentations import (
    Affine, ColorJitter, Compose, HorizontalFlip, RandomCrop, ToFloat, VerticalFlip
)
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
from tqdm import tqdm

from src.data.image_mask_datasets import ImageMaskRandomDataset
from src.models.metrics import Alignment, Uniformity
from src.models.modules import create_encoder
from src.utils.config import read_json_config


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
    rng = np.random.default_rng(parameters["seed"])

    tf.random.set_seed(parameters["seed"])

    running_time = time.strftime("%b-%d-%Y_%H-%M")
    model_dir = results_dir / "model"

    log_dir = model_dir / "logs"/ f"{running_time}"
    log_dir.mkdir(parents=True, exist_ok=True)

    if config["encoder_weights_path"] is None:
        raise ValueError("Checkpoint must be provided for evaluation")

    encoder = create_encoder(
        img_height=parameters["target_height"],
        img_width=parameters["target_width"],
        filters=parameters["bottleneck_size"],
        do=parameters["dropout"],
        l2=parameters["l2_regularization"],
    )


    checkpoint = tf.train.Checkpoint(
        encoder=encoder,
    )

    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=config["encoder_weights_path"],
        max_to_keep=5,
    )

    manager.restore_or_initialize()

    transform = Compose([
        ToFloat(max_value=255),
        Affine(scale=0.5, fit_output=True, keep_ratio=True, p=1),
        RandomCrop(width=256, height=256),
    ])

    image_mask_dataset = ImageMaskRandomDataset(
        Path(config["validation"]["img_dir"]),
        Path(config["validation"]["mask_dir"]),
        transform=transform,
        batch_size=config["validation"]["batch_size"],
        epoch_size=parameters["samples_per_epoch"],
        num_classes=parameters["num_classes"],
        threshold_obj_perc=parameters["thresh_obj_perc"],
        max_iters=parameters["max_iters"],
        seed=rng,
        expand_masks=False,
    )

    augmentation = Compose([
        VerticalFlip(),
        HorizontalFlip(),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0, always_apply=True),
    ])

    evaluate(encoder, image_mask_dataset, augmentation)


def generate_positive_pairs(imgs, masks, num_samples=50):
    positive_df = pd.DataFrame({
        "image_index": np.arange(len(imgs)),
        "class": [np.unique(mask)[1:].astype(int) for mask in masks]
        }).explode("class")

    class_counts = positive_df["class"].value_counts()
    class_counts = class_counts[class_counts >= 2]
    class_counts /= np.sum(class_counts)

    class_indices = np.random.choice(class_counts.index, num_samples, p=class_counts.values)

    positive_pairs = []
    for class_idx in class_indices:
        positive_samples = positive_df[positive_df["class"] == class_idx]["image_index"]
        positive_pairs.append(np.random.choice(positive_samples, 2, replace=False))
    positive_pairs = np.array(positive_pairs)

    return imgs[positive_pairs.T]

def evaluate(model, image_mask_dataset, augmentation):
    alignment_augmented = Alignment(name="alignment_augmented")
    alignment_categorical = Alignment(name="alignment_categorical")
    uniformity_augmented = Uniformity(name="uniformity_augmented", t=0.2)

    for imgs, masks in tqdm(image_mask_dataset):
        xs, ys = [], []
        for img in imgs:
            x, y = augmentation(image=img)["image"], augmentation(image=img)["image"]
            xs.append(x)
            ys.append(y)

        xs, ys = tf.stack(xs, axis=0), tf.stack(ys, axis=0)
        xs, ys = model(xs, training=False), model(ys, training=False)

        alignment_augmented.update_state(xs, ys)
        uniformity_augmented.update_state(xs)
        uniformity_augmented.update_state(ys)

        xs, ys = generate_positive_pairs(imgs, masks, num_samples=image_mask_dataset.batch_size)
        xs, ys = model(xs, training=False), model(ys, training=False)
        alignment_categorical.update_state(xs, ys)

    for metric in (alignment_augmented, uniformity_augmented, alignment_categorical):
        print(f"{metric.name}: {metric.result().numpy()}")

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
