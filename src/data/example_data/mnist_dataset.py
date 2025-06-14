# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Torch Dataset wrapper for the MNIST dataset with normalization."""

from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    """Custom Dataset for loading MNIST with preprocessing."""

    def __init__(self, train: bool = True, root: str = "data_temp") -> None:
        """Initialize the MNIST dataset with transforms.

        Args:
            train (bool): Whether to load the training or test split.
            root (str): Path to store the dataset.
        """
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ],
        )
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=self.transform,
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """Retrieve a single data sample as a dictionary.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary with keys 'image', 'label', and 'index'.
        """
        image, label = self.dataset[idx]
        return {
            "image": image,
            "label": label,
            "index": idx,
        }
