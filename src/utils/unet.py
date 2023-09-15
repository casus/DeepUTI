from pathlib import Path
from typing import Union

import numpy as np

from src.models.metrics import DiceCoeff
from src.utils.data import split_into_patches


def evaluate_model(model, dataset, target_height, target_width):

    dice_coeff = DiceCoeff()

    for img_idx in range(len(dataset)):
        img, mask = dataset[img_idx]
        img, mask = np.squeeze(img), np.squeeze(mask)

        img_patches = split_into_patches(img, target_height, target_width)
        mask_patches = split_into_patches(mask, target_height, target_width)
        preds = model.predict(img_patches, verbose=0)

        dice_coeff.update_state(preds, mask_patches)

    return {"avg_dice_coeff": dice_coeff.result().numpy()}

def find_best_checkpoint(log_dir: Union[Path, str]):
    log_dir = Path(log_dir)
    if (log_dir / "model.best.hdf5").exists():
        best_checkpoint = log_dir / "model.best.hdf5"
    else:
        checkpoints = list(log_dir.glob("*.hdf5"))
        best_checkpoint = max(checkpoints) # Take the latest checkpoint

    return best_checkpoint
