import torch
import torch.nn as nn
import torch.nn.functional as F


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2),
            nn.LeakyReLU(),
            nn.Dropout(),

            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Dropout(),

            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Dropout(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Dropout(),

            nn.Conv2d(512, 512, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Dropout(),

            nn.MaxPool2d(kernel_size=5, stride=2)
        )
        self.linear = nn.Linear(512, 2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.softmax(x)
        x = self.dropout(x)
        x = self.linear(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.fc4(x)

        return x
