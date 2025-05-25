# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Example of LightningModule implementation for training a classification model."""

from collections.abc import Callable

import torch
from lightning import LightningModule


class ClassificationLitModule(LightningModule):
    """LightningModule for training, validating, and testing a classification model."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Callable,
        loss_fn: Callable,
        scheduler: Callable | None = None,
    ) -> None:
        """Initialize the model.

        Args:
            net (torch.nn.Module): Torch Model instance.
            optimizer (Callable): A partial function that returns an optimizer when passed model parameters.
            loss_fn (Callable): Loss function used for training, validation, and testing.
            scheduler (Callable, optional): A scheduler function to adjust the learning rate.

        """
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.net(x)

    def training_step(self, batch: dict, _: int) -> torch.Tensor:
        """Run one training step.

        Args:
            batch (tuple): A tuple of input, label, and index tensors.
            _ (int): Unused batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        x, y = batch["image"], batch["label"]
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict, _: int) -> torch.Tensor:
        """Run one validation step.

        Args:
            batch (tuple): A tuple of input, label, and index tensors.
            _ (int): Unused batch index.

        Returns:
            torch.Tensor: Validation loss.
        """
        x, y = batch["image"], batch["label"]
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: dict, _: int) -> torch.Tensor:
        """Evaluate the model on the test set.

        Args:
            batch (tuple): A tuple of input, label, and index tensors.
            _ (int): Unused batch index.

        Returns:
            torch.Tensor: Test loss.
        """
        x, y = batch["image"], batch["label"]
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch: dict, _: int) -> torch.Tensor:
        """Generate predictions for a given batch during inference.

        This method is used during `Trainer.predict()` to produce model outputs,
        such as logits or predicted class indices.

        Args:
            batch (tuple): A tuple of input, (optional) label, and index tensors.
            _ (int): Unused batch index.

        Returns:
            torch.Tensor: Model predictions (e.g., predicted class indices).
        """
        x, idx = batch["image"], batch["index"]
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        return {"idx": idx, "pred": preds}

    def configure_optimizers(self) -> torch.optim.Optimizer | dict:
        """Set up the optimizer and scheduler.

        Returns:
            torch.optim.Optimizer | dict: The optimizer instance if no scheduler is used,
            or a dictionary containing the optimizer and scheduler if a scheduler is provided.
        """
        optimizer = self.optimizer(self.net.parameters())

        # If scheduler is provided, return the optimizer and scheduler
        if self.scheduler:
            scheduler = self.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # You can also use "step" if you want the scheduler to step per batch
                    "frequency": 1,
                },
            }

        return optimizer
