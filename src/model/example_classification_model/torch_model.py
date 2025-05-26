# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""A simple CNN for image classification using one local max pool and a small adaptive grid.

Architecture:
    Conv2d → ReLU
    Conv2d → ReLU → MaxPool2d(2x2)
    AdaptiveAvgPool2d(2x2)
    Flatten → FC → ReLU → Dropout → FC

This preserves some spatial structure (2x2 grid) before flattening, giving the head
more capacity than a 1x1 global pool while still avoiding hardcoded size math.
"""

import torch
from torch import nn
from torch.nn import functional


class ExampleClassificationCNN(nn.Module):
    """CNN for image classification with local pooling + adaptive grid pooling.

    Args:
        num_classes (int): Number of output classes.
        input_channels (int): Number of channels in input images (e.g., 1 for MNIST).
        conv1_channels (int): Channels out of the first conv layer. Default 32.
        conv2_channels (int): Channels out of the second conv layer. Default 64.
        fc1_units (int): Units in the first fully connected layer. Default 128.
        dropout_rate (float): Dropout after the first FC. Default 0.0.
        kernel_size (int): Convolutional kernel size. Default 3.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        input_channels: int = 1,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        fc1_units: int = 128,
        dropout_rate: float = 0.0,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        # Two convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
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

        # One local max pool to reduce spatial size by 2x
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive average pool to a fixed 2x2 grid
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        # Fully connected head
        self.fc1 = nn.Linear(conv2_channels * 2 * 2, fc1_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc1_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = self.pool(x)
        x = self.adaptive_pool(x)  # now shape (B, conv2_channels, 2, 2)
        x = torch.flatten(x, 1)  # (B, conv2_channels*2*2)
        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
