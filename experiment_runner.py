import os

import matplotlib.pyplot as plt
import torch

from simple_renderer.data_processing.load_data import load_training_data
from simple_renderer.model.basic import BasicSceneRenderer

x, y = load_training_data(os.path.join("data"))

settings = {"epochs": 5000, "batch_size": 4, "learning_rate": 0.001}

model = BasicSceneRenderer(settings=settings)
history = model.train(x, y)
plt.plot(history["loss_train"])
plt.show()

# model = Network()
# x = torch.ones(1)
# y = model(x.view(-1, 1))
