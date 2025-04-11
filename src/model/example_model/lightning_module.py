# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Example of LightningModule implementation for training a classification model."""

from collections.abc import Callable

import torch
from lightning import LightningModule
from torch.nn import functional

from .torch_model import ExampleTorchModel


class ExampleLightningModel(LightningModule):
    """LightningModule for training, validating, and testing a classification model."""

    def __init__(self, num_classes: int, optimizer: Callable) -> None:
        """Initialize the model.

        Args:
            num_classes (int): Number of output classes.
            lr (float): Learning rate for the optimizer.
            optimizer (Callable): A partial function that returns an optimizer when passed model parameters.
        """
        super().__init__()
        self.model = ExampleTorchModel(num_classes)
        self.optimizer = optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        """Run one training step.

        Args:
            batch (tuple): A tuple of input, label, and index tensors.
            _ (int): Unused batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        x, y, _ = batch
        logits = self.forward(x)
        loss = functional.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        """Run one validation step.

        Args:
            batch (tuple): A tuple of input, label, and index tensors.
            _ (int): Unused batch index.

        Returns:
            torch.Tensor: Validation loss.
        """
        x, y, _ = batch
        logits = self.forward(x)
        loss = functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        """Evaluate the model on the test set.

        Args:
            batch (tuple): A tuple of input, label, and index tensors.
            _ (int): Unused batch index.

        Returns:
            torch.Tensor: Test loss.
        """
        x, y, _ = batch
        logits = self.forward(x)
        loss = functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        """Generate predictions for a given batch during inference.

        This method is used during `Trainer.predict()` to produce model outputs,
        such as logits or predicted class indices.

        Args:
            batch (tuple): A tuple of input, (optional) label, and index tensors.
            _ (int): Unused batch index.

        Returns:
            torch.Tensor: Model predictions (e.g., predicted class indices).
        """
        x, _, idx = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        return {"idx": idx, "pred": preds}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Set up the optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer instance.
        """
        return self.optimizer(self.model.parameters())
