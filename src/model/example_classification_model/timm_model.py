# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""An example of classification model wrapper using timm library."""

import timm
import torch
from torch import nn


class TimmClassificationModel(nn.Module):
    """A PyTorch wrapper for a TIMM model to be used for classification tasks.

    This class creates a timm model instance with customizable architecture,
    pretrained weights, number of output classes, and input channels.

    Args:
        model_name (str): Name of the timm model architecture to instantiate.
            Examples include "resnet18", "efficientnet_b0", etc.
            Defaults to "resnet18".
        pretrained (bool): If True, loads pretrained weights on ImageNet.
            Defaults to True.
        num_classes (int): Number of output classes for the classification head.
            Defaults to 10.
        input_channels (int): Number of input image channels.
            Must match the input channels expected by the model.
            Defaults to 1 (e.g., grayscale images like MNIST).
        **kwargs: Additional keyword arguments to pass to timm.create_model.

    Example:
        >>> model = TimmClassificationModel(model_name="resnet18", pretrained=True, num_classes=10, input_channels=3)
        >>> outputs = model(torch.randn(8, 3, 224, 224))

    Notes:
        - The input tensor shape should be (batch_size, input_channels, height, width).
        - Ensure the input images are appropriately resized and normalized to match the
          timm model's expected input size and preprocessing.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        num_classes: int = 10,
        input_channels: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=input_channels,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the timm model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output logits with shape (batch_size, num_classes).
        """
        return self.model(x)
