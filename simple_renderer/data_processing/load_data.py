import os

import cv2.cv2 as cv2
import numpy as np

from simple_renderer.config import IMAGE_HEIGHT, IMAGE_WIDTH


def load_training_data(directory_path: str) -> tuple[np.ndarray, np.ndarray]:
    all_files = os.listdir(directory_path)
    all_image_data = []
    for image_file in all_files:
        all_image_data.append(cv2.imread(os.path.join(directory_path, image_file)))

    x = np.array(all_image_data)
    y = np.array([0, 45, 90, 135, 180, 225, 270, 315]).reshape(-1, 1)
    x = np.moveaxis(x, -1, 1)
    return x, y
