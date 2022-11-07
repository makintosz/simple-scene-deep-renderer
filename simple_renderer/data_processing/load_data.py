import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_training_data(
    directory_path: str,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    labels = pd.read_csv(os.path.join(directory_path, "labels.csv"))
    scaler_x = MinMaxScaler((-1, 1))
    x = labels[["x", "y", "z", "yaw", "pitch"]]
    x = scaler_x.fit_transform(x)
    return x, labels, scaler_x
