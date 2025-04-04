import lightning as L
from torch.utils.data import DataLoader, random_split
from .torch_dataset import ExampleTorchDataset

class ExampleDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        full_dataset = ExampleTorchDataset(train=True)
        self.train_dataset, self.val_dataset = random_split(full_dataset, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
