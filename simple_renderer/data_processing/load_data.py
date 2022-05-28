import os

import cv2.cv2 as cv2
import numpy as np

from simple_renderer.config import IMAGE_HEIGHT, IMAGE_WIDTH


def load_training_data(directory_path: str) -> tuple[np.ndarray, np.ndarray]:
    all_files = os.listdir(directory_path)
    all_image_data = []
    for image_file in all_files:
        image = cv2.imread(os.path.join(directory_path, image_file))
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = np.divide(image, 255)
        all_image_data.append(image)

    y = np.array(all_image_data)
    y = np.moveaxis(y, -1, 1)
    x = np.array([0, 45, 90, 135, 180, 225, 270, 315]).reshape(-1, 1)
    x = np.divide(x, 360)
    return x, y
