import colorama
import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from typing import List, Tuple

from config.config import ViTELMConfig
from config.dataset_config import vitelm_general_dataset_config
from nn.models.model_selector import ModelFactory
from nn.models.vit_elm import ELM
from nn.dataloader.vit_dataset_selector import create_dataset
from utils.utils import (create_timestamp, create_save_dirs, use_gpu_if_available, setup_logger, display_config,
                         find_latest_file_in_latest_directory)


class ViTELMTrain:
    def __init__(self):
        # Set up configuration settings
        timestamp = create_timestamp()
        setup_logger()
        colorama.init()
        self.device = use_gpu_if_available()
        self.vitelm_cfg = ViTELMConfig().parse()
        display_config(self.vitelm_cfg)
        self.dataset_config = vitelm_general_dataset_config(self.vitelm_cfg)

        if self.vitelm_cfg.seed:
            torch.manual_seed(42)

        # Set up dataset
        self.dataset = create_dataset(train=True, dataset_info=self.dataset_config)
        train_size = int(self.vitelm_cfg.train_set_size * len(self.dataset))
        valid_size = len(self.dataset) - train_size
        self.train_dataset, valid_dataset = random_split(self.dataset, [train_size, valid_size])
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.vitelm_cfg.batch_size, shuffle=True)
        self.valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.vitelm_cfg.batch_size, shuffle=False)
        self.num_classes = self.dataset_config.get("num_classes")

        # Set up model
        self.model = ModelFactory.create_model(network_type=self.vitelm_cfg.network_type,
                                               vit_model_name=self.vitelm_cfg.vit_model_name,
                                               num_neurons=768,
                                               num_classes=self.num_classes,
                                               device=self.device)

        summary(model=self.model,
                input_size=(1, self.dataset_config.get("num_channels"), 224, 224),
                verbose=2)

        # Set loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.vitelm_cfg.learning_rate)

        # Create directories
        if not self.vitelm_cfg.load_weights:
            # Create save directory for model weights
            self.save_path_vit_weights = (
                create_save_dirs(
                    directory_path=self.dataset_config.get('ViT_saved_weights'),
                    timestamp=timestamp,
                    network_type=self.vitelm_cfg.network_type,
                    model_type=self.vitelm_cfg.vit_model_name
                )
            )

            # Create save directory for logging the ViT
            save_path_logs = (
                create_save_dirs(
                    directory_path=self.dataset_config.get("logs"),
                    timestamp=timestamp,
                    network_type=self.vitelm_cfg.network_type,
                    model_type=self.vitelm_cfg.vit_model_name
                )
            )
            self.writer = SummaryWriter(log_dir=save_path_logs)

        # if self.vitelm_cfg.network_type == "ViTELM":
        self.save_path_combined_model_weights = (
            create_save_dirs(directory_path=self.dataset_config.get("combined_model_saved_weights"),
                             timestamp=timestamp,
                             network_type=self.vitelm_cfg.network_type,
                             model_type=self.vitelm_cfg.vit_model_name
                             )
        )

        self.best_valid_loss = float('inf')
        self.best_model_path = None

    def train_vit_loop(self, train_losses: List[float]) -> List[float]:
        """
        This method iterates over the training DataLoader, performs forward passes,
        computes the loss, backpropagates the error, and updates the model parameters.
        It appends the training losses to the provided list.

        Args:
            train_losses (List[float]): A list to store the training loss values for each batch.

        Returns:
            List[float]: The updated list of training loss values for each batch after the training loop.

        """

        self.model.train()

        for inputs, labels in tqdm(self.train_loader,
                                   total=len(self.train_loader),
                                   desc=colorama.Fore.LIGHTRED_EX + f"Training {self.vitelm_cfg.network_type}, "
                                                                    f"with {self.vitelm_cfg.vit_model_name} model"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            train_loss = self.criterion(outputs, labels)
            train_loss.backward()
            self.optimizer.step()
            train_losses.append(train_loss.item())

        return train_losses

    def valid_vit_loop(self, valid_losses: List[float]) -> List[float]:
        """
        This method iterates over the validation DataLoader, performs forward passes,
        computes the loss, and appends the validation losses to the provided list.
        The model is set to evaluation mode, and gradients are not calculated.

        Args:
            valid_losses (List[float]): A list to store the validation loss values for each batch.

        Returns:
            List[float]: The updated list of validation loss values for each batch after the validation loop.

        """

        self.model.eval()

        with torch.no_grad():
            for inputs, labels in tqdm(self.valid_loader,
                                       total=len(self.valid_loader),
                                       desc=colorama.Fore.CYAN + f"Validating {self.vitelm_cfg.network_type}, "
                                                                 f"with {self.vitelm_cfg.vit_model_name} model"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
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
            torch.save(self.model.state_dict(), self.best_model_path)
            logging.info(f"New weights have been saved at epoch {epoch} with value of {valid_loss:.5f}")
        else:
            logging.warning(f"No new weights have been saved. Best valid loss was {self.best_valid_loss:.5f},\n "
                            f"current valid loss is {valid_loss:.5f}")

    def extract_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from the Vision Transformer (ViT) model for training the ELM.

        This method iterates over the training DataLoader, extracts features using the ViT model,
        and stores these features along with the one-hot encoded labels in pre-allocated tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - all_features: A tensor of shape (num_samples, num_input_neurons) with the extracted features.
                - all_labels: A tensor of shape (num_samples, num_classes) with the one-hot encoded labels.
        """

        # Pre-allocate tensors for features and labels
        num_input_neurons = 768  # self.model.get_input_neurons()
        num_samples = len(self.train_dataset)
        all_features = torch.zeros((num_samples, num_input_neurons), device=self.device)
        all_labels = torch.zeros((num_samples, self.num_classes), device=self.device)

        # Extract features for ELM training
        model = self.model
        model.eval()
        model.model.heads = nn.Sequential(nn.Identity())

        with torch.no_grad():
            start_idx = 0
            for inputs, labels in tqdm(self.train_loader,
                                       total=len(self.train_loader),
                                       desc=colorama.Fore.MAGENTA + "Extracting features"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_size = inputs.size(0)
                features = model.forward(inputs)
                end_idx = start_idx + batch_size
                all_features[start_idx:end_idx] = features
                all_labels[start_idx:end_idx] = \
                    torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
                start_idx = end_idx

        return all_features, all_labels

    def train_elm(self, all_features: torch.Tensor, all_labels: torch.Tensor) -> None:
        """
        Train the Extreme Learning Machine (ELM) using the extracted features and labels.

        This method trains the ELM model using the extracted features and one-hot encoded labels
        obtained from the Vision Transformer (ViT) model.

        Args:
            all_features (torch.Tensor): A tensor containing the extracted features from the ViT model.
            all_labels (torch.Tensor): A tensor containing the one-hot encoded labels.

        Returns:
            None
        """

        logging.info("Training ELM")
        self.model.train_elm(all_features, all_labels)
        torch.save(self.model.state_dict(),
                   os.path.join(self.save_path_combined_model_weights, "combined_model.pth"))

    def fit(self):
        """

        Returns:

        """

        train_losses = []
        valid_losses = []

        if not self.vitelm_cfg.load_weights:
            for epoch in tqdm(range(self.vitelm_cfg.epochs), desc=colorama.Fore.BLUE + "Epochs"):
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
            latest_vit_weight_file = (
                find_latest_file_in_latest_directory(self.dataset_config.get("ViT_saved_weights"))
            )
            state_dict = torch.load(latest_vit_weight_file)
            self.model.load_state_dict(state_dict)

        # if self.vitelm_cfg.network_type == "ViTELM":
            # Collect features
        all_features, all_labels = self.extract_features()

        all_features = all_features.to("cpu")
        all_labels = all_labels.to("cpu")

        elm = ELM(768, 768, 10)
        elm.fit(all_features, all_labels)

        # Train ELM classifier
        # self.train_elm(all_features, all_labels)


if __name__ == "__main__":
    try:
        vitelm_train = ViTELMTrain()
        vitelm_train.fit()
    except KeyboardInterrupt as kie:
        logging.error(kie)
