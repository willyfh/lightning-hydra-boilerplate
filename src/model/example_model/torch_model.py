# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""An example of CNN model definition using PyTorch for image classification."""

import torch
from torch import nn
from torch.nn import functional


class ExampleTorchModel(nn.Module):
    """A simple CNN with two convolutional layers and two fully connected layers."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize the CNN model.

        Args:
            num_classes (int): Number of output classes. Default is 10.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = functional.relu(self.conv1(x))
        x = self.pool(functional.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = functional.relu(self.fc1(x))
        return self.fc2(x)
