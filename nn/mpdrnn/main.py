import logging

from elm.src.config.config import DatasetConfig, MPDRNNConfig
from elm.src.config.dataset_config import general_dataset_configs
from elm.src.dataset_operations.load_dataset import load_data
from elm.src.utils.utils import setup_logger, display_dataset_info
from mpdrnn_wrapper import ELMModelWrapper


class MultiPhaseDeepRandomizedNeuralNetwork:
    def __init__(self):
        # Initialize paths and settings
        setup_logger()
        cfg_training = MPDRNNConfig().parse()
        cfg_data_preprocessing = DatasetConfig().parse()
        gen_ds_cfg = general_dataset_configs(cfg_training)

        display_dataset_info(gen_ds_cfg)

        # Load data
        self.train_data, self.train_labels, self.test_data, self.test_labels = (
            load_data(gen_ds_cfg, cfg_data_preprocessing))

    def main(self):
        elm_wrapper = (
            ELMModelWrapper(train_data=self.train_data, train_labels=self.train_labels, test_data=self.test_data,
                            test_labels=self.test_labels))

        # Train the model
        elm_wrapper.train_first_layer()
        elm_wrapper.evaluate_first_layer()

        elm_wrapper.train_second_layer()
        elm_wrapper.evaluate_second_layer()
        #
        # elm_wrapper.train_third_layer()
        # elm_wrapper.evaluate_third_layer()

        total_time = elm_wrapper.get_total_execution_time()
        logging.info("Total training time: %.4f seconds", total_time)


if __name__ == '__main__':
    try:
        mp_d_rnn = MultiPhaseDeepRandomizedNeuralNetwork()
        mp_d_rnn.main()
    except KeyboardInterrupt as kie:
        logging.error(f'{kie}')
