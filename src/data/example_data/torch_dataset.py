# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Torch Dataset wrapper for the MNIST dataset with normalization."""

from torch.utils.data import Dataset
from torchvision import datasets, transforms


class ExampleTorchDataset(Dataset):
    """Custom Dataset for loading MNIST with preprocessing."""

    def __init__(self, train: bool = True) -> None:
        """Initialize the MNIST dataset with transforms.

        Args:
            train (bool): Whether to load the training or test split.
        """
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ],
        )
        self.dataset = datasets.MNIST(
            root="data_temp",
            train=train,
            download=True,
            transform=self.transform,
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[object, int]:
        """Retrieve a single (image, label) pair by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Tensor, int]: Tuple of image tensor and its corresponding label.
        """
        image, label = self.dataset[idx]
        return image, label
