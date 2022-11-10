import os

import matplotlib.pyplot as plt
import torch

from simple_renderer.data_processing.load_data import load_training_data
from simple_renderer.model.basic import BasicSceneRenderer, Network

x, labels, scaler_x = load_training_data(os.path.join("data"))
settings = {"epochs": 101, "batch_size": 1, "learning_rate": 0.0001}

# model = Network()
# x_sample = torch.ones(5)
# y_sample = model(x_sample.view(-1, 5))
# exit()

model = BasicSceneRenderer(settings=settings, scaler_x=scaler_x)
history = model.train(x, labels)
model.save_model()
plt.plot(history["loss_train"])
plt.show()
