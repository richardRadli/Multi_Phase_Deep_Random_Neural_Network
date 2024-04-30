import numpy as np
import torch
from torch.utils.data import Dataset


class NpyDataset(Dataset):
    def __init__(self, file_path, operation, transform=None):
        data = np.load(file_path, allow_pickle=True)
        self.transform = transform

        if operation == "train":
            self.x = torch.from_numpy(data[0]).float()
            self.y = torch.from_numpy(data[2]).float()
        elif operation == "test":
            self.x = torch.from_numpy(data[1]).float()
            self.y = torch.from_numpy(data[3]).float()
        else:
            raise Exception("Operation must be train or test")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.transform is not None:
            self.x = self.transform(self.x[idx])

        return self.x[idx], self.y[idx]
