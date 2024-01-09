import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def load_data_elm(gen_ds_cfg):
    data_file = gen_ds_cfg.get("cached_dataset_file")
    data = np.load(data_file, allow_pickle=True)
    
    train_data = data[0]
    train_labels = data[2]
    test_data = data[1]
    test_labels = data[3]

    return train_data, train_labels, test_data, test_labels


def load_data_fcnn(gen_ds_cfg, cfg):
    # Load the fcnn_data directly
    data_file = gen_ds_cfg.get("cached_dataset_file")
    data = np.load(data_file, allow_pickle=True)
    train_data = torch.tensor(data[0], dtype=torch.float32)
    train_labels = torch.tensor(data[2], dtype=torch.float32)

    # Split data into training and validation sets
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        train_data, train_labels, test_size=cfg.valid_size, random_state=42
    )

    # Create DataLoader for training data
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    # Create DataLoader for validation data
    valid_dataset = TensorDataset(valid_data, valid_labels)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)

    test_data = torch.tensor(data[1], dtype=torch.float32)
    test_labels = torch.tensor(data[3], dtype=torch.float32)
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
