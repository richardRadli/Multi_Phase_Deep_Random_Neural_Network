import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from typing import List

from config.config import ViTELMConfig
from config.dataset_config import vitelm_general_dataset_config
from nn.models.vit_elm import ViTELM
from nn.dataloader.vit_dataset_selector import create_dataset
from utils.utils import create_timestamp, create_save_dirs, use_gpu_if_available, setup_logger


class ViTELMTrain:
    def __init__(self):
        timestamp = create_timestamp()
        setup_logger()

        self.device = use_gpu_if_available()
        self.vitelm_cfg = ViTELMConfig().parse()
        dataset_config = vitelm_general_dataset_config(self.vitelm_cfg)

        if self.vitelm_cfg.seed:
            torch.manual_seed(0)

        self.dataset = create_dataset(train=True, dataset_info=dataset_config)
        train_size = int(self.vitelm_cfg.train_set_size * len(self.dataset))
        valid_size = len(self.dataset) - train_size
        self.train_dataset, valid_dataset = random_split(self.dataset, [train_size, valid_size])

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.vitelm_cfg.batch_size, shuffle=True)
        self.valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.vitelm_cfg.batch_size, shuffle=False)

        self.num_classes = dataset_config.get("num_classes")
        self.combined_model = ViTELM(self.num_classes).to(self.device)
        self.num_input_neurons = self.combined_model.get_input_neurons()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.combined_model.parameters(),
                                    lr=self.vitelm_cfg.learning_rate)

        # Create save directory for model weights
        self.save_path_vit_weights = (
            create_save_dirs(timestamp, dataset_config.get("ViT_saved_weights"))
        )
        self.save_path_combined_model_weights = (
            create_save_dirs(timestamp, dataset_config.get("combined_model_saved_weights"))
        )

        save_path_logs = (
            create_save_dirs(timestamp, dataset_config.get("logs"))
        )
        self.writer = SummaryWriter(log_dir=save_path_logs)

        self.best_valid_loss = float('inf')
        self.best_model_path = None

    def train_vit_loop(self, train_losses: List[float]):
        self.combined_model.train()

        for inputs, labels in tqdm(self.train_loader, total=len(self.train_loader), desc="Training ViT"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.combined_model(inputs)
            train_loss = self.criterion(outputs, labels)
            train_loss.backward()
            self.optimizer.step()
            train_losses.append(train_loss.item())

        return train_losses

    def valid_vit_loop(self, valid_losses: List[float]):
        self.combined_model.eval()

        with torch.no_grad():
            for inputs, labels in tqdm(self.valid_loader, total=len(self.valid_loader), desc="Validating ViT"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.combined_model(inputs)
                validation_loss = self.criterion(outputs, labels)
                valid_losses.append(validation_loss.item())

        return valid_losses

    def save_model_weights(self, epoch: int, valid_loss: float) -> None:
        """
        Saves the model and weights if the validation loss is improved.

        Args:
            epoch (int): the current epoch
            valid_loss (float): the validation loss

        Returns:
            None
        """

        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            if self.best_model_path is not None:
                os.remove(self.best_model_path)
            self.best_model_path = os.path.join(self.save_path_vit_weights, "epoch_" + str(epoch) + ".pt")
            torch.save(self.combined_model.state_dict(), self.best_model_path)
            logging.info(f"New weights have been saved at epoch {epoch} with value of {valid_loss:.5f}")
        else:
            logging.warning(f"No new weights have been saved. Best valid loss was {self.best_valid_loss:.5f},\n "
                            f"current valid loss is {valid_loss:.5f}")

    def extract_features(self):
        # Pre-allocate tensors for features and labels
        num_samples = len(self.train_dataset)
        all_features = torch.zeros((num_samples, self.num_input_neurons), device=self.device)
        all_labels = torch.zeros((num_samples, self.num_classes), device=self.device)

        # Extract features for ELM training
        self.combined_model.eval()
        with torch.no_grad():
            start_idx = 0
            for inputs, labels in tqdm(self.train_loader, total=len(self.train_loader), desc="Collecting features"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_size = inputs.size(0)
                features = self.combined_model.extract_vit_features(inputs)
                end_idx = start_idx + batch_size
                all_features[start_idx:end_idx] = features
                all_labels[start_idx:end_idx] = \
                    torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
                start_idx = end_idx

        return all_features, all_labels

    def train_elm(self, all_features, all_labels):
        logging.info("Training ELM")
        self.combined_model.train_elm(all_features, all_labels)
        torch.save({
            'vit_model_state_dict': self.combined_model.vit_model.state_dict(),
            'elm_alpha_weights': self.combined_model.elm_head.alpha_weights,
            'elm_beta_weights': self.combined_model.elm_head.beta_weights
        }, os.path.join(self.save_path_combined_model_weights, "combined_model.pth"))

    def fit(self):
        train_losses = []
        valid_losses = []

        if not self.vitelm_cfg.load_weights:
            for epoch in tqdm(range(self.vitelm_cfg.epochs), desc="Epochs"):
                train_losses = self.train_vit_loop(train_losses)
                valid_losses = self.valid_vit_loop(valid_losses)

                train_loss = np.average(train_losses)
                valid_loss = np.average(valid_losses)
                self.writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)
                logging.info(f'train_loss: {train_loss:.5f} valid_loss: {valid_loss:.5f}')

                train_losses.clear()
                valid_losses.clear()

                self.save_model_weights(epoch, valid_loss)

            # Log metrics
            self.writer.close()
            self.writer.flush()
            del train_loss, valid_loss
        else:
            pass  # TODO: load the latest .pt file

        # Collect features
        all_features, all_labels = self.extract_features()

        # Train ELM classifier
        self.train_elm(all_features, all_labels)


if __name__ == "__main__":
    vitelm_train = ViTELMTrain()
    vitelm_train.fit()
