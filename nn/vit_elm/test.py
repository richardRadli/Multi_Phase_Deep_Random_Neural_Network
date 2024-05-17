import logging
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from config.config import ViTELMConfig
from config.dataset_config import vitelm_general_dataset_config
from nn.models.vit_elm import ViTELM
from nn.dataloader.vit_dataset_selector import create_dataset
from utils.utils import use_gpu_if_available, setup_logger


class TestViTELM:
    def __init__(self):
        torch.manual_seed(0)
        setup_logger()

        cfg = ViTELMConfig().parse()
        dataset_info = vitelm_general_dataset_config(cfg)
        self.device = use_gpu_if_available()
        self.combined_model = self.load_model()
        test_dataset = create_dataset(train=False, dataset_info=dataset_info)
        self.test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    def load_model(self):
        combined_model = ViTELM(10)
        checkpoint = torch.load('C:/Users/ricsi/Desktop/combined_model.pth', map_location=self.device)
        combined_model.vit_model.load_state_dict(checkpoint['vit_model_state_dict'])
        combined_model.elm_head.alpha_weights.data = checkpoint['elm_alpha_weights']
        combined_model.elm_head.beta_weights = checkpoint['elm_beta_weights']
        return combined_model

    def test_accuracy(self):
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_dataloader, total=len(self.test_dataloader), desc='Test'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.combined_model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        logging.info(f'Accuracy of the combined model on the test images: {100 * correct / total} %')


if __name__ == '__main__':
    test_vitelm = TestViTELM()
    test_vitelm.test_accuracy()
