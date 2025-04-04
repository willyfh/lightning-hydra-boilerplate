import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class ExampleTorchDataset(Dataset):
    def __init__(self, train: bool = True):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = datasets.MNIST(root="data_temp", train=train, download=True, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label
