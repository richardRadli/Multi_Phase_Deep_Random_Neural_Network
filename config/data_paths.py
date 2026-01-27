import logging
import os

from pathlib import Path

from utils.utils import setup_logger


class _Const(object):
    setup_logger()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    STORAGE_ROOT = PROJECT_ROOT / "storage"
    DATASET_ROOT = STORAGE_ROOT / "datasets"

    for dir_root in [STORAGE_ROOT, DATASET_ROOT]:
        os.makedirs(dir_root, exist_ok=True)

    @classmethod
    def create_directories(cls, dirs, root_type) -> None:
        """
        Class method that creates the missing directories.
        :param dirs: These are the directories that the function checks.
        :param root_type: Either PROJECT or DATASET or STORAGE.
        :return: None
        """

        for _, path in dirs.items():
            if root_type == "STORAGE":
                dir_path = os.path.join(cls.STORAGE_ROOT, path)
            elif root_type == "DATASET":
                dir_path = os.path.join(cls.DATASET_ROOT, path)
            elif root_type == "PROJECT":
                dir_path = os.path.join(cls.PROJECT_ROOT, path)
            else:
                raise ValueError("Wrong root type!")

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"Directory {dir_path} has been created")



class ConfigFilePaths(_Const):
    dirs_config_paths = {
        
        "config_dev_drnn":
            "config/json_files/DEV_DRNN_config.json",
        "config_schema_dev_drnn":
            "config/json_files/DEV_DRNN_config_schema.json"
    }

    def __init__(self):
        super().__init__()

    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.dirs_config_paths.get(key, ""))


class DevDRNNPaths(_Const):
    dirs_dataset_paths = {
        "results_adult":
            "networks/dev_drnn/data/results/adult",
        "results_cifar10":
            "networks/dev_drnn/data/results/cifar10",
        "results_connect4":
            "networks/dev_drnn/data/results/connect4",
        "results_isolete":
            "networks/dev_drnn/data/results/isolete",
        "results_letter":
            "networks/dev_drnn/data/results/letter",
        "results_mnist":
            "networks/dev_drnn/data/results/mnist",
        "results_mnist_fashion":
            "networks/dev_drnn/data/results/mnist_fashion",
        "results_musk2":
            "networks/dev_drnn/data/results/musk2",
        "results_optdigits":
            "networks/dev_drnn/data/results/optdigits",
        "results_page_blocks":
            "networks/dev_drnn/data/results/page_blocks",
        "results_segment":
            "networks/dev_drnn/data/results/segment",
        "results_shuttle":
            "networks/dev_drnn/data/results/shuttle",
        "results_spambase":
            "networks/dev_drnn/data/results/spambase",
        "results_usps":
            "networks/dev_drnn/data/results/usps",
        "results_satimages":
            "networks/dev_drnn/data/results/satimages",
        "results_wall":
            "networks/dev_drnn/data/results/wall",
        "results_waveform":
            "networks/dev_drnn/data/results/waveform",

        "hyperparam_adult":
            "networks/dev_drnn/data/hyperparam/adult",
        "hyperparam_cifar10":
            "networks/dev_drnn/data/hyperparam/cifar10",
        "hyperparam_connect4":
            "networks/dev_drnn/data/hyperparam/connect4",
        "hyperparam_isolete":
            "networks/dev_drnn/data/hyperparam/isolete",
        "hyperparam_letter":
            "networks/dev_drnn/data/hyperparam/letter",
        "hyperparam_mnist":
            "networks/dev_drnn/data/hyperparam/mnist",
        "hyperparam_mnist_fashion":
            "networks/dev_drnn/data/hyperparam/mnist_fashion",
        "hyperparam_musk2":
            "networks/dev_drnn/data/hyperparam/musk2",
        "hyperparam_optdigits":
            "networks/dev_drnn/data/hyperparam/optdigits",
        "hyperparam_page_blocks":
            "networks/dev_drnn/data/hyperparam/page_blocks",
        "hyperparam_segment":
            "networks/dev_drnn/data/hyperparam/segment",
        "hyperparam_shuttle":
            "networks/dev_drnn/data/hyperparam/shuttle",
        "hyperparam_spambase":
            "networks/dev_drnn/data/hyperparam/spambase",
        "hyperparam_usps":
            "networks/dev_drnn/data/hyperparam/usps",
        "hyperparam_satimages":
            "networks/dev_drnn/data/hyperparam/satimages",
        "hyperparam_wall":
            "networks/dev_drnn/data/hyperparam/wall",
        "hyperparam_waveform":
            "networks/dev_drnn/data/hyperparam/waveform"
    }

    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset_paths, "STORAGE")

    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.dirs_dataset_paths.get(key, ""))


class DatasetFilesPaths(_Const):
    dirs_dataset_paths = {
        "dataset_path_adult":
            "adult",
        "dataset_path_cifar10":
            "cifar10",
        "dataset_path_connect4":
            "connect4",
        "dataset_path_isolete":
            "isolete",
        "dataset_path_letter":
            "letter",
        "dataset_path_mnist":
            "mnist",
        "dataset_path_mnist_fashion":
            "mnist_fashion",
        "dataset_path_musk2":
            "musk2",
        "dataset_path_optdigits":
            "optdigits",
        "dataset_path_page_blocks":
            "page_blocks",
        "dataset_path_segment":
            "segment",
        "dataset_path_shuttle":
            "shuttle",
        "dataset_path_spambase":
            "spambase",
        "dataset_path_usps":
            "usps",
        "dataset_path_satimages":
            "satimages",
        "dataset_path_wall":
            "wall",
        "dataset_path_waveform":
            "waveform",
    }

    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset_paths, "DATASET")

    def get_data_path(self, key):
        return os.path.join(self.DATASET_ROOT, self.dirs_dataset_paths.get(key, ""))


CONST: _Const = _Const()

JSON_FILES_PATHS: ConfigFilePaths = ConfigFilePaths()
DEV_DRNN_PATHS: DevDRNNPaths = DevDRNNPaths()
DATASET_FILES_PATHS: DatasetFilesPaths = DatasetFilesPaths()
