# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Example of LightningDataModule implementation for managing data loading."""

from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from .torch_dataset import ExampleTorchDataset


class ExampleDataModule(LightningDataModule):
    """DataModule for training, validation, test, and prediction using ExampleTorchDataset."""

    def __init__(self, batch_size: int = 32, num_workers: int = 4) -> None:
        """Initialize the DataModule.

        Args:
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses used for data loading.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Prepare datasets for all splits.

        Args:
            stage (str | None): One of None, 'fit', 'validate', 'test', or 'predict'.
                This argument is unused in this implementation, but you could use it to conditionally
                load specific datasets based on the stage of the model training or evaluation. For example:
                - 'fit': Load the training and validation datasets.
                - 'validate': Optionally, load validation data only.
                - 'test': Load test dataset for evaluation.
                - 'predict': Load prediction dataset for inference.
        """
        full_train_dataset = ExampleTorchDataset(train=True)
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [55000, 5000],
        )
        self.test_dataset = ExampleTorchDataset(train=False)

    def train_dataloader(self) -> DataLoader:
        """Return the training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return the data loader for making predictions.

        In this simple implementation, we use the same `test_dataset` for both testing
        and prediction purposes. This can be extended in the future to use a different
        dataset for inference if needed.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
