import os
import time

import cv2.cv2 as cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from simple_renderer.config import DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH
from simple_renderer.data_processing.dataset import ImageDataset
from simple_renderer.model.renderer_base import SceneRenderer


class BasicSceneRenderer(SceneRenderer):
    def __init__(self, settings: dict) -> None:
        self._settings = settings
        self._device = torch.device(DEVICE)
        self._model = Network().to(self._device)

    def train(self, x: np.ndarray, y: np.ndarray) -> dict:
        dataloader = DataLoader(
            dataset=ImageDataset(x, y),
            batch_size=self._settings["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        optimizer = optim.Adam(
            self._model.parameters(), lr=self._settings["learning_rate"]
        )
        self._model.train()
        history = {"loss_train": []}
        for epoch in range(self._settings["epochs"]):
            start_time = time.time()
            epoch_loss = 0
            counter = 0
            for x, y in dataloader:
                x = x.to(self._device)
                y = y.to(self._device)
                self._model.zero_grad()
                output = self._model(x.view(-1, 1))
                loss = func.mse_loss(output, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                counter += 1

            self.generate_all_frames(epoch)
            execution_time = time.time() - start_time
            history["loss_train"].append(epoch_loss / counter)
            print(f"{epoch} = {epoch_loss / counter} | {execution_time} s")

        return history

    def generate_all_frames(self, epoch: int) -> None:
        if epoch % 100 == 0 and epoch != 0:
            os.makedirs(
                os.path.join("results", "all_angels", str(epoch)), exist_ok=True
            )
            for i in range(0, 360, 5):
                preview = self.generate_frame([i])
                cv2.imwrite(
                    os.path.join("results", "all_angels", str(epoch), f"{i}.jpg"),
                    preview,
                )

        if epoch % 10 == 0:
            preview = self.generate_frame([45])
            cv2.imwrite(os.path.join("results", "45_degrees", f"{epoch}.jpg"), preview)

    def generate_frame(self, data: list) -> np.ndarray:
        data = np.divide(data, 360)
        x = torch.Tensor(data).view(-1, 1)
        x = x.to(self._device)
        output = self._model(x)
        output = output.cpu().detach().numpy()
        output = np.moveaxis(output, 1, -1).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        output *= 255
        # print(np.max(output))
        # print(np.min(output))
        output = output.clip(0, 256)
        return output

    def save_model(self) -> None:
        torch.save(
            self._model.linear.state_dict(), os.path.join("results", "basic_linear.pt")
        )
        torch.save(
            self._model.convolutional.state_dict(),
            os.path.join("results", "basic_convolutional.pt"),
        )

    def load_model(self, path_linear: str, path_convolutional: str) -> None:
        self._model.linear.load_state_dict(torch.load(path_linear))
        self._model.linear.eval()
        self._model.convolutional.load_state_dict(torch.load(path_convolutional))
        self._model.convolutional.eval()


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4096),
            nn.ReLU(),
        )

        self.unflatten = nn.Sequential(
            nn.Unflatten(dim=-1, unflattened_size=(64, 8, 8)),
        )

        self.convolutional = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                stride=(2, 2),
                kernel_size=(3, 5),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                stride=(2, 2),
                kernel_size=(3, 7),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                stride=(2, 2),
                kernel_size=(3, 5),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                stride=(2, 2),
                kernel_size=(3, 5),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=32,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=32,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=3, kernel_size=(3, 3), padding=(1, 1)
            ),
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.unflatten(x)
        x = self.convolutional(x)
        return x
