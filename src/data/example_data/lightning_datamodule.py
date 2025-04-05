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

    def setup(self, stage: str) -> None:
        """Prepare datasets for different stages: 'fit', 'validate', 'test', 'predict'.

        Args:
            stage (str): One of 'fit', 'validate', 'test', or 'predict'.
        """
        if stage == "fit":
            full_train_dataset = ExampleTorchDataset(train=True)
            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset,
                [55000, 5000],
            )

        elif stage == "validate":
            if not hasattr(self, "val_dataset"):
                full_train_dataset = ExampleTorchDataset(train=True)
                _, self.val_dataset = random_split(
                    full_train_dataset,
                    [55000, 5000],
                )

        elif stage == "test":
            self.test_dataset = ExampleTorchDataset(train=False)

        elif stage == "predict":
            self.predict_dataset = ExampleTorchDataset(train=False)

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
        """Return the predict data loader."""
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
