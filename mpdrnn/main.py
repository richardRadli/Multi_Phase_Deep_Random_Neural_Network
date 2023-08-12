import logging
import numpy as np

from scipy import stats
from sklearn import preprocessing

from config.config import DatasetConfig, MPDRNNConfig
from config.dataset_config import general_dataset_configs
from mpdrnn_wrapper import ELMModelWrapper
from utils.utils import setup_logger, display_dataset_info


class MultiPhaseDeepRandomizedNeuralNetwork:
    def __init__(self):
        setup_logger()
        cfg_training = MPDRNNConfig().parse()
        cfg_dataset = DatasetConfig().parse()
        gen_ds_cfg = general_dataset_configs(cfg_training)

        display_dataset_info(gen_ds_cfg)

        data_file = gen_ds_cfg.get("cached_dataset_file")
        data = np.load(data_file, allow_pickle=True)

        self.train_data = data[0]
        self.train_labels = data[2]
        self.test_data = data[1]
        self.test_labels = data[3]

        if cfg_dataset.normalize:
            if cfg_dataset.type_of_normalization == "zscore":
                self.train_data = stats.zscore(self.train_data)
                self.test_data = stats.zscore(self.test_data)
            elif cfg_dataset.type_of_normalization == "minmax":
                scaler = preprocessing.MinMaxScaler()
                self.train_data = scaler.fit_transform(self.train_data)
                self.test_data = scaler.fit_transform(self.test_data)
            else:
                raise ValueError("Wrong type of normalization!")

    def main(self):
        elm_wrapper = (
            ELMModelWrapper(train_data=self.train_data, train_labels=self.train_labels, test_data=self.test_data,
                            test_labels=self.test_labels))

        # Train the model
        elm_wrapper.train_first_layer()
        elm_wrapper.evaluate_first_layer()

        elm_wrapper.train_second_layer()
        elm_wrapper.evaluate_second_layer()

        elm_wrapper.train_third_layer()
        elm_wrapper.evaluate_third_layer()

        total_time = elm_wrapper.get_total_execution_time()
        logging.info("Total training time: %.4f seconds", total_time)


if __name__ == '__main__':
    try:
        mp_d_rnn = MultiPhaseDeepRandomizedNeuralNetwork()
        mp_d_rnn.main()
    except KeyboardInterrupt as kie:
        logging.error(f'{kie}')
