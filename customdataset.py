import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import lightning as L
import os
import pandas as pd
from sklearn.model_selection import train_test_split

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
    


def get_data(data_path, batch_size=16, val_split=0.2, num_workers=8, seed=42, device='cpu'):
    dataNames = os.listdir(data_path)

    X = []
    y = []   
    for dataName in dataNames:
        print(dataName)
        dfTr = pd.read_csv(data_path + '/' + dataName + '/train.csv')
        X_tr = dfTr.values[:,:360].reshape(-1,6,60)
        y_tr = dfTr.values[:,-1].astype(np.int32)
        X.append(X_tr)
        y.append(y_tr)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True, random_state = seed)

    X_train = torch.tensor(X_train.astype(np.float32), dtype=torch.float32, device=device).detach()
    X_test = torch.tensor(X_test.astype(np.float32), dtype=torch.float32, device=device).detach()
    y_train = torch.tensor(y_train.astype(np.float32), dtype=torch.float32, device=device).detach()
    y_test = torch.tensor(y_test.astype(np.float32), dtype=torch.float32, device=device).detach()


    X = X.astype(np.float32)
    y = y.astype(int)

    # Create the DataModule
    data_module = CustomDataModule(X, y, batch_size=batch_size, val_split=val_split, num_workers=num_workers)
    
    return data_module