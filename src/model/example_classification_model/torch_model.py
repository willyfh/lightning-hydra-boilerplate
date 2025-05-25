# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""An example of CNN model definition using PyTorch for image classification."""

import torch
from torch import nn
from torch.nn import functional


class ExampleClassificationCNN(nn.Module):
    """A simple CNN with two convolutional layers and two fully connected layers.

    Args:
        num_classes (int): Number of output classes. Default is 10.
        conv1_channels (int): Number of output channels for the first conv layer. Default is 32.
        conv2_channels (int): Number of output channels for the second conv layer. Default is 64.
        fc1_units (int): Number of units in the first fully connected layer. Default is 128.
        dropout_rate (float): Dropout probability after the first fully connected layer. Default is 0.0 (no dropout).
        kernel_size (int): Kernel size for convolutional layers. Default is 3.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        fc1_units: int = 128,
        dropout_rate: float = 0.0,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv1_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=conv1_channels,
            out_channels=conv2_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(conv2_channels * 14 * 14, fc1_units)
        self.fc2 = nn.Linear(fc1_units, num_classes)

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
        x = self.dropout(x)
        return self.fc2(x)
