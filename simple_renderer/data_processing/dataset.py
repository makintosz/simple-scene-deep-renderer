import os

import cv2.cv2 as cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from simple_renderer.config import IMAGE_HEIGHT, IMAGE_WIDTH


class ImageDataset(Dataset):
    def __init__(self, x: np.ndarray, labels: pd.DataFrame) -> None:
        self._x = x.astype(np.float32)
        self._labels = labels
        self._n_samples = len(self._labels)

    def __getitem__(self, item) -> tuple[np.ndarray, np.ndarray]:
        image_id = self._labels.at[item, "id"]
        image = cv2.imread(os.path.join("data", "images", f"{image_id}.png"))
        image = image.astype(np.float32)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = np.moveaxis(image, -1, 0)
        image = np.divide(image, 255)
        return self._x[item], image

    def __len__(self) -> int:
        return self._n_samples
