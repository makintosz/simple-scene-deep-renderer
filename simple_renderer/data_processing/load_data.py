import os

import cv2.cv2 as cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from simple_renderer.config import IMAGE_HEIGHT, IMAGE_WIDTH


def load_training_data(
    directory_path: str,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    labels = pd.read_csv(os.path.join(directory_path, "labels.csv"))
    all_image_data = []
    image_ids = labels["id"].values
    for image_id in image_ids:
        image = cv2.imread(os.path.join(directory_path, "images", f"{image_id}.png"))
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = np.divide(image, 255)
        all_image_data.append(image)

    y = np.array(all_image_data)
    y = np.moveaxis(y, -1, 1)

    scaler_x = MinMaxScaler((-1, 1))
    x = labels[["x", "y", "z", "yaw", "pitch"]]
    x = scaler_x.fit_transform(x)
    return x, y, scaler_x
