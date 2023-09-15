from pathlib import Path

import click
from albumentations.core.serialization import load as load_albumentations_transform

from src.data.image_mask_datasets import ImageMaskStaticDataset
from src.utils.config import read_json_config


@click.command()
@click.argument("config_file_path", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def main(config_file_path, output_dir):

    config = read_json_config(config_file_path)
    parameters = config["parameters"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # transform_val = load_albumentations_transform(
    #     config["validation"]["transform_filepath"])

    # ImageMaskStaticDataset.create_from_directory(
    #     Path(config["validation"]["img_dir"]),
    #     Path(config["validation"]["mask_dir"]),
    #     output_dir,
    #     transform=transform_val,
    #     batch_size=config["validation"]["batch_size"],
    #     epoch_size=parameters["samples_per_epoch"],
    #     num_classes=parameters["num_classes"],
    #     threshold_obj_perc=parameters["thresh_obj_perc"],
    #     max_iters=parameters["max_iters"],
    #     seed=parameters["seed"],
    # )
    transform_val = load_albumentations_transform(
        config["validation"]["transform_filepath"])

    ImageMaskStaticDataset.create_from_directory(
        Path(config["test"]["img_dir"]),
        Path(config["test"]["mask_dir"]),
        output_dir,
        transform=transform_val,
        batch_size=config["test"]["batch_size"],
        epoch_size=parameters["samples_per_epoch"],
        num_classes=parameters["num_classes"],
        threshold_obj_perc=parameters["thresh_obj_perc"],
        max_iters=parameters["max_iters"],
        seed=parameters["seed"],
    )

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
