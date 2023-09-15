from pathlib import Path
from typing import Iterable, Union

import numpy as np
from albumentations import Affine, Compose, RandomCrop, ToFloat
from albumentations.core.serialization import Serializable as Transform
from tensorflow.keras.utils import Sequence, load_img

from src.models.metrics import get_metric

MAN_MASK_DIR = "data/binary/validation/man_mask"
WEAK_MASK_DIR = "data/binary/validation/weak_mask"
TRANSFORM_PATH = "configs/transforms/0_5_fold_res/val.json"


class MaskRandomDataset(Sequence):

    def __init__(
        self,
        man_mask_dir: Union[str, Path],
        weak_mask_dir: Union[str, Path],
        transform: Transform = None,
        batch_size: int = 50,
        epoch_size: int = 1000,
        seed: Union[None, int, np.random.Generator] = None,
        thresh_obj_perc: float = 0.001,
        max_iters: int = 10,
        **kwargs,
    ) -> None:
        """ Initializes the dataset.

        Args:
            img_dir (Union[str, Path]): Directory containing images to use.
            transform (Transform, optional): Albumentations transforms to apply to images.
            batch_size (int, optional): Size of each batch. Defaults to 50.
            epoch_size (int, optional): Total number of samples in each epoch. Defaults to 1000.
        """
        super().__init__(**kwargs)
        self._rng = np.random.default_rng(seed)

        self._man_mask_list = np.sort([
            path for path in Path(man_mask_dir).rglob("*.tif")
            if not path.stem.startswith(".")
        ])

        self._weak_mask_list = np.sort([
            path for path in Path(weak_mask_dir).rglob("*.tif")
            if not path.stem.startswith(".")
        ])


        self._transform = transform

        self.man_mask_names = [mask.name for mask in self._man_mask_list]
        self.weak_mask_names = [mask.name for mask in self._weak_mask_list]

        assert np.all(self.man_mask_names == self.weak_mask_names)

        self._threshold_obj_perc = thresh_obj_perc
        self._max_iters = max_iters

        self.batch_size = batch_size
        self.epoch_size = epoch_size

    def _check_threshhold_condition(self, masks: np.ndarray) -> bool:
        """ Given a batch of masks, checks if they satisfy the condition of ROI density
        """
        num_positive = np.sum(masks.astype(bool))
        num_pixels = np.prod(masks.shape)

        return (num_positive / num_pixels) > self._threshold_obj_perc


    def _generate_data(self, batch_indices: Iterable[int]) -> np.ndarray:
        """ Given indices of masks, generates a batch of samples from them

        Args:
            batch_indices (Iterable[int]): Indices of masks to be included in the batch

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of arrays representing masks
        """
        man_masks = []
        weak_masks = []
        for mask_idx in batch_indices:
            man_mask = load_img(self._man_mask_list[mask_idx], color_mode="grayscale")
            man_mask = np.expand_dims(man_mask, -1)

            weak_mask = load_img(self._weak_mask_list[mask_idx], color_mode="grayscale")
            weak_mask = np.expand_dims(weak_mask, -1)

            if self._transform is not None:
                transformed = self._transform(
                    image=np.zeros_like(man_mask),
                    man_mask=man_mask,
                    weak_mask=weak_mask,
                )
                man_mask = transformed["man_mask"]
                weak_mask = transformed["weak_mask"]

            man_masks.append(man_mask)
            weak_masks.append(weak_mask)

        return np.stack(man_masks).astype(bool), np.stack(weak_masks).astype(bool)

    def __len__(self) -> int:
        return int(np.ceil(self.epoch_size / self.batch_size))

    def __getitem__(self, index: int) -> np.ndarray:
        """ Generates a random batch of masks.

            Index argument is not used, it is only an argument because Sequence requires it.
        """
        for _ in range(self._max_iters):
            batch_indices = self._rng.choice(len(self._man_mask_list), size=self.batch_size, replace=True)
            man_masks, weak_masks = self._generate_data(batch_indices)
            if self._check_threshhold_condition(man_masks):
                break

        return man_masks.astype(np.float32), weak_masks.astype(np.float32)

def main():
    transform = Compose(
        [
            ToFloat(max_value=255),
            Affine(scale=0.5, fit_output=True, keep_ratio=True, p=1),
            RandomCrop(width=256, height=256),
        ],
        additional_targets={
            "man_mask": "mask",
            "weak_mask": "mask",
        }
    )

    mask_dataset = MaskRandomDataset(
        MAN_MASK_DIR,
        WEAK_MASK_DIR,
        transform=transform,
        batch_size=512,
        epoch_size=1024,
    )

    metrics = [get_metric('dice'), get_metric('bce')]

    for weak_masks, man_masks in mask_dataset:
        for metric in metrics:
            metric.update_state(man_masks, weak_masks)
    for metric in metrics:
        print(f"{metric.name}: {metric.result()}")

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
