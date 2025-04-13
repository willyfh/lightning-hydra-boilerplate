# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Callback for logging metrics during validation and test using PyTorch Lightning."""

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT


class MetricEvaluator(Callback):
    """Logs custom metrics during validation and test epochs.

    Args:
        metrics (Dict[str, List[torch.nn.Module]]): A dictionary mapping each stage
            ("validation", "test") to a list of metric instances.

    Example:
            {
                "validation": [Accuracy(...), F1Score(...)],
                "test": [Accuracy(...), F1Score(...)]
            }
    """

    def __init__(self, metrics: dict[str, list[torch.nn.Module]]) -> None:
        self.metrics = metrics

    def _reset(self, stage: str) -> None:
        """Reset all metrics for a given stage."""
        for metric in self.metrics.get(stage, []):
            metric.reset()

    def _update(self, stage: str, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update all metrics for a given stage with predictions and targets."""
        for metric in self.metrics.get(stage, []):
            metric.to(preds.device)
            metric.update(preds, targets)

    def _log(self, stage: str, pl_module: LightningModule) -> None:
        """Compute and log all metrics for a given stage."""
        prefix = "val" if stage == "validation" else "test"
        for metric in self.metrics.get(stage, []):
            name = metric.__class__.__name__.lower()
            value = metric.compute()
            pl_module.log(f"{prefix}_{name}", value, prog_bar=True)

    def on_validation_epoch_start(self, _trainer: Trainer, _pl_module: LightningModule) -> None:
        """Reset metrics at the start of validation epoch."""
        self._reset("validation")

    def on_validation_batch_end(
        self,
        _trainer: Trainer,
        pl_module: LightningModule,
        _outputs: STEP_OUTPUT,
        batch: dict,
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ) -> None:
        """Update validation metrics with each batch."""
        x, y = batch["image"], batch["label"]
        preds = pl_module(x).argmax(dim=1)
        self._update("validation", preds, y)

    def on_validation_epoch_end(self, _trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log validation metrics at end of epoch."""
        self._log("validation", pl_module)

    def on_test_epoch_start(self, _trainer: Trainer, _pl_module: LightningModule) -> None:
        """Reset metrics at the start of test epoch."""
        self._reset("test")

    def on_test_batch_end(
        self,
        _trainer: Trainer,
        pl_module: LightningModule,
        _outputs: STEP_OUTPUT,
        batch: dict,
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ) -> None:
        """Update test metrics with each batch."""
        x, y = batch["image"], batch["label"]
        preds = pl_module(x).argmax(dim=1)
        self._update("test", preds, y)

    def on_test_epoch_end(self, _trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log test metrics at end of epoch."""
        self._log("test", pl_module)
