import numpy as np
import torch

from torch.utils.data import Dataset
from typing import Tuple


class NpzDataset(Dataset):
    def __init__(self, file_path: str, operation: str):
        """
        Initializes the dataset by loading data from a .npz file based on the specified operation.

        Args:
            file_path (str): Path to the .npz file containing the dataset.
            operation (str): The operation type which determines which subset of data to load. Should be one of
                ["train", "test"].

        Raises:
            ValueError: If the provided operation is not one of ["train", "test"].
        """

        data = np.load(file_path, allow_pickle=True)

        data_keys = {
            "train": ("train_x", "train_y"),
            "test": ("test_x", "test_y")
        }

        if operation not in data_keys:
            raise ValueError("operation must be one of {}".format(data_keys.keys()))

        x_key, y_key = data_keys[operation]

        self.x = torch.from_numpy(data.get(x_key)).float()
        self.y = torch.from_numpy(data.get(y_key)).float()

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """

        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The sample and its label.
        """

        return self.x[idx], self.y[idx]
