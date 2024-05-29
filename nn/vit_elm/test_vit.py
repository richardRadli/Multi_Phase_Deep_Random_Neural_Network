import logging
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader

from config.config import ViTELMConfig
from config.dataset_config import vitelm_general_dataset_config
from nn.models.model_selector import ModelFactory
from nn.dataloader.vit_dataset_selector import create_dataset
from utils.utils import (display_config, find_latest_file_in_latest_directory, setup_logger, use_gpu_if_available,
                         pretty_print_results_vit, create_timestamp, create_save_dirs, plot_confusion_matrix_vit)


class TestViTELM:
    def __init__(self):
        timestamp = create_timestamp()
        setup_logger()
        self.cfg = ViTELMConfig().parse()
        display_config(self.cfg)
        dataset_info = vitelm_general_dataset_config(self.cfg)

        self.num_classes = dataset_info.get("num_classes")
        self.class_labels = dataset_info.get("class_labels")
        self.device = use_gpu_if_available()
        self.model = self.load_model(dataset_info)

        test_dataset = create_dataset(train=False, dataset_info=dataset_info)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.cfg.batch_size, shuffle=False)

        self.save_results_path = (
            create_save_dirs(
                dataset_info.get("path_to_results"), timestamp, self.cfg.network_type, self.cfg.vit_model_name
            )
        )

        self.cm_path = (
            create_save_dirs(
                dataset_info.get("path_to_cm"), timestamp, self.cfg.network_type, self.cfg.vit_model_name
            )
        )

    def load_model(self, dataset_info):
        model = ModelFactory.create_model(network_type=self.cfg.network_type,
                                          vit_model_name=self.cfg.vit_model_name,
                                          num_neurons=768,
                                          num_classes=self.num_classes,
                                          device=self.device)

        if self.cfg.network_type == "ViTELM":
            latest_combined_model_path = (
                find_latest_file_in_latest_directory(dataset_info.get("combined_model_saved_weights"))
            )
            checkpoint = torch.load(latest_combined_model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
        else:
            latest_model_path = find_latest_file_in_latest_directory(dataset_info.get("ViT_saved_weights"))
            model.load_state_dict(torch.load(latest_model_path))

        return model

    def test_metrics(self):
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_dataloader,
                                       total=len(self.test_dataloader),
                                       desc=f'Evaluating {self.cfg.network_type} with {self.cfg.vit_model_name} model '
                                            f'on the {self.cfg.dataset_name} dataset'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, fscore, _ = (
            precision_recall_fscore_support(all_labels, all_predictions, average='macro')
        )
        cm = confusion_matrix(all_labels, all_predictions)

        pretty_print_results_vit(acc=accuracy,
                                 precision=precision,
                                 recall=recall,
                                 fscore=fscore,
                                 root_dir=self.save_results_path)

        plot_confusion_matrix_vit(cm, self.cm_path, labels=self.class_labels, cfg=self.cfg)


if __name__ == '__main__':
    try:
        test_vitelm = TestViTELM()
        test_vitelm.test_metrics()
    except KeyboardInterrupt as kie:
        logging.error(kie)
