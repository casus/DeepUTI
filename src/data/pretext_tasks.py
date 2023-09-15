from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from skimage.util import view_as_blocks


def _split_into_blocks(image: np.ndarray, num_blocks: tuple[int, int]):
    if image.shape[0] % num_blocks[0] != 0 or image.shape[1] % num_blocks[1] != 0:
        raise ValueError(f"Image shape {image.shape} is not cleanly divisible by blocks number {num_blocks}")

    blocks = view_as_blocks(image, (image.shape[0]//num_blocks[0], image.shape[1]//num_blocks[1]))

    blocks = np.vstack(blocks.reshape((*blocks.shape[:2], -1)))
    return blocks


@dataclass
class ClassificationPretextTask:
    image: np.ndarray
    label: int

@dataclass
class PredictionPretextTask:
    image: np.ndarray
    label: np.ndarray

class PretextTaskCreator(ABC):
    def __init__(self, seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        self.rng = np.random.default_rng(seed)

    @property
    @abstractmethod
    def output_len(self) -> int:
        pass

    @property
    @abstractmethod
    def classification(self) -> bool:
        pass

    def _generate_label(self) -> int:
        return self.rng.choice(self.output_len)

    @abstractmethod
    def __call__(self, image: np.ndarray) -> Union[ClassificationPretextTask, PredictionPretextTask]:
        pass

def get_pretext_task(
    task: str,
    seed: Optional[Union[int, np.random.Generator]] = None,
    **kwargs,
    ) -> PretextTaskCreator:
    match task.lower():
        case "regionduplication":
            return RegionDuplicationTaskCreator(seed=seed, **kwargs)
        case "rotation":
            return RotationTaskCreator(seed=seed, **kwargs)
        case "variance":
            return VarianceTaskCreator(seed=seed, **kwargs)
        case "standarddeviation":
            return StandardDeviationTaskCreator(seed=seed, **kwargs)
        case _:
            raise ValueError(f"Unknown task: {task}")

class RegionDuplicationTaskCreator(PretextTaskCreator):
    _QUARTERS = {
        1: (1, 2),
        2: (1, 3),
        3: (1, 4),
        4: (2, 3),
        5: (2, 4),
        6: (3, 4),
    }

    @property
    def output_len(self) -> int:
        return 7

    @property
    def classification(self) -> bool:
        return True

    def _duplicate_quarters(self, img: np.ndarray, copy_from: int, copy_to: int) -> np.ndarray:
        if img.shape[0] % 2 != 0 or img.shape[1] % 2 != 0:
            raise ValueError("Images size must be divisible by two")

        y_center, x_center = img.shape[0]//2, img.shape[1]//2

        match copy_from:
            case 1:
                copied_fragment = img[:y_center, x_center:]
            case 2:
                copied_fragment = img[:y_center, :x_center]
            case 3:
                copied_fragment = img[y_center:, :x_center]
            case 4:
                copied_fragment = img[y_center:, x_center:]

        match copy_to:
            case 1:
                img[:y_center, x_center:] = copied_fragment
            case 2:
                img[:y_center, :x_center] = copied_fragment
            case 3:
                img[y_center:, :x_center] = copied_fragment
            case 4:
                img[y_center:, x_center:] = copied_fragment

        return img

    def __call__(self, img: np.ndarray) -> ClassificationPretextTask:
        """ Given an image, generates duplicated quarters pretext task

        Args:
            img (np.ndarray): Image sample

        Returns:
            PretextTask: A modified image and associated label
        """
        label = self._generate_label()

        if label == 0:
            return ClassificationPretextTask(img, label)

        quarters = self._QUARTERS[label]

        if self.rng.uniform() < 0.5:
            return ClassificationPretextTask(self._duplicate_quarters(img, quarters[0], quarters[1]), label)
        return ClassificationPretextTask(self._duplicate_quarters(img, quarters[1], quarters[0]), label)

class RotationTaskCreator(PretextTaskCreator):
    @property
    def output_len(self) -> int:
        return 4

    @property
    def classification(self) -> bool:
        return True

    def __call__(self, image: np.ndarray) -> ClassificationPretextTask:
        label = self._generate_label()

        return ClassificationPretextTask(np.rot90(image, label), label)

class VarianceTaskCreator(PretextTaskCreator):
    def __init__(
        self,
        num_blocks: tuple[int, int],
        treshold: float = 0.00015,
        seed: Optional[Union[int, np.random.Generator]] = None,
        ) -> None:
        super().__init__(seed)
        self.num_blocks = num_blocks
        self.treshold = treshold

    @property
    def output_len(self) -> int:
        return self.num_blocks[0] * self.num_blocks[1]

    @property
    def classification(self) -> bool:
        return False

    def __call__(self, image: np.ndarray) -> PredictionPretextTask:
        blocks = _split_into_blocks(image, self.num_blocks)
        variance = np.var(blocks, 1)
        variance = variance > self.treshold
        return PredictionPretextTask(image, variance.astype(int))

class StandardDeviationTaskCreator(PretextTaskCreator):
    def __init__(
        self,
        num_blocks: tuple[int, int],
        treshold: float = 0.015,
        seed: Optional[Union[int, np.random.Generator]] = None,
        ) -> None:
        super().__init__(seed)
        self.num_blocks = num_blocks
        self.treshold = treshold

    @property
    def output_len(self) -> int:
        return self.num_blocks[0] * self.num_blocks[1]

    @property
    def classification(self) -> bool:
        return False

    def __call__(self, image: np.ndarray) -> PredictionPretextTask:
        blocks = _split_into_blocks(image, self.num_blocks)
        std = np.std(blocks, 1)
        std = std > self.treshold
        return PredictionPretextTask(image, std.astype(int))
