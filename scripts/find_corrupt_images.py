from pathlib import Path

import click
from tensorflow.keras.utils import load_img


@click.command()
@click.argument("directory_to_check", type=click.Path(exists=True))
def main(directory_to_check):

    img_list = list(Path(directory_to_check).rglob("*.tif"))
    for img_path in img_list:
        try:
            load_img(img_path, color_mode="grayscale")
        except:
            print(img_path)

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
