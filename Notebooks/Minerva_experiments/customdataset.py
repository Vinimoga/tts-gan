import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import lightning as L

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class CustomDataModule(L.LightningDataModule):
    def __init__(self, data, labels, batch_size=16, val_split=0.2, num_workers=8):
        super().__init__()
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = CustomDataset(self.data, self.labels)
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)