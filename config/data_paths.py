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
        "config_mpdrnn":
            "config/json_files/MPDRNN_config.json",
        "config_schema_mpdrnn":
            "config/json_files/MPDRNN_config_schema.json",

        "config_dev_drnn":
            "config/json_files/DEV_DRNN_config.json",
        "config_schema_dev_drnn":
            "config/json_files/DEV_DRNN_config_schema.json"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.dirs_config_paths.get(key, ""))


class MPDRNNPaths(_Const):
    dirs_dataset_paths = {
        "results_connect4":
            "mpdrnn/data/results/connect4",
        "results_isolete":
            "mpdrnn/data/results/isolete",
        "results_letter":
            "mpdrnn/data/results/letter",
        "results_mnist":
            "mpdrnn/data/results/mnist",
        "results_mnist_fashion":
            "mpdrnn/data/results/mnist_fashion",
        "results_musk2":
            "mpdrnn/data/results/musk2",
        "results_optdigits":
            "mpdrnn/data/results/optdigits",
        "results_page_blocks":
            "mpdrnn/data/results/page_blocks",
        "results_segment":
            "mpdrnn/data/results/segment",
        "results_shuttle":
            "mpdrnn/data/results/shuttle",
        "results_spambase":
            "mpdrnn/data/results/spambase",
        "results_usps":
            "mpdrnn/data/results/usps",
        "results_satimages":
            "mpdrnn/data/results/satimages",
        "results_wall":
            "mpdrnn/data/results/wall",
        "results_waveform":
            "mpdrnn/data/results/waveform",

        "hyperparam_connect4":
            "mpdrnn/data/hyperparam/connect4",
        "hyperparam_isolete":
            "mpdrnn/data/hyperparam/isolete",
        "hyperparam_letter":
            "mpdrnn/data/hyperparam/letter",
        "hyperparam_mnist":
            "mpdrnn/data/hyperparam/mnist",
        "hyperparam_mnist_fashion":
            "mpdrnn/data/hyperparam/mnist_fashion",
        "hyperparam_musk2":
            "mpdrnn/data/hyperparam/musk2",
        "hyperparam_optdigits":
            "mpdrnn/data/hyperparam/optdigits",
        "hyperparam_page_blocks":
            "mpdrnn/data/hyperparam/page_blocks",
        "hyperparam_segment":
            "mpdrnn/data/hyperparam/segment",
        "hyperparam_shuttle":
            "mpdrnn/data/hyperparam/shuttle",
        "hyperparam_spambase":
            "mpdrnn/data/hyperparam/spambase",
        "hyperparam_usps":
            "mpdrnn/data/hyperparam/usps",
        "hyperparam_satimages":
            "mpdrnn/data/hyperparam/satimages",
        "hyperparam_wall":
            "mpdrnn/data/hyperparam/wall",
        "hyperparam_waveform":
            "mpdrnn/data/hyperparam/waveform"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset_paths, "STORAGE")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.dirs_dataset_paths.get(key, ""))


class DEVDRNNPaths(_Const):
    dirs_dataset_paths = {
        "results_connect4":
            "dev_drnn/data/results/connect4",
        "results_isolete":
            "dev_drnn/data/results/isolete",
        "results_letter":
            "dev_drnn/data/results/letter",
        "results_mnist":
            "dev_drnn/data/results/mnist",
        "results_mnist_fashion":
            "dev_drnn/data/results/mnist_fashion",
        "results_musk2":
            "dev_drnn/data/results/musk2",
        "results_optdigits":
            "dev_drnn/data/results/optdigits",
        "results_page_blocks":
            "dev_drnn/data/results/page_blocks",
        "results_segment":
            "dev_drnn/data/results/segment",
        "results_shuttle":
            "dev_drnn/data/results/shuttle",
        "results_spambase":
            "dev_drnn/data/results/spambase",
        "results_usps":
            "dev_drnn/data/results/usps",
        "results_satimages":
            "dev_drnn/data/results/satimages",
        "results_wall":
            "dev_drnn/data/results/wall",
        "results_waveform":
            "dev_drnn/data/results/waveform",

        "hyperparam_connect4":
            "dev_drnn/data/hyperparam_tuning/connect4",
        "hyperparam_isolete":
            "dev_drnn/data/hyperparam_tuning/isolete",
        "hyperparam_letter":
            "dev_drnn/data/hyperparam_tuning/letter",
        "hyperparam_mnist":
            "dev_drnn/data/hyperparam_tuning/mnist",
        "hyperparam_mnist_fashion":
            "dev_drnn/data/hyperparam_tuning/mnist_fashion",
        "hyperparam_musk2":
            "dev_drnn/data/hyperparam_tuning/musk2",
        "hyperparam_optdigits":
            "dev_drnn/data/hyperparam_tuning/optdigits",
        "hyperparam_page_blocks":
            "dev_drnn/data/hyperparam_tuning/page_blocks",
        "hyperparam_segment":
            "dev_drnn/data/hyperparam_tuning/segment",
        "hyperparam_satimages":
            "dev_drnn/data/hyperparam_tuning/satimages",
        "hyperparam_shuttle":
            "dev_drnn/data/hyperparam_tuning/shuttle",
        "hyperparam_spambase":
            "dev_drnn/data/hyperparam_tuning/spambase",
        "hyperparam_usps":
            "dev_drnn/data/hyperparam_tuning/usps",
        "hyperparam_wall":
            "dev_drnn/data/hyperparam_tuning/wall",
        "hyperparam_waveform":
            "dev_drnn/data/hyperparam_tuning/waveform"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset_paths, "STORAGE")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.dirs_dataset_paths.get(key, ""))


class DatasetFilesPaths(_Const):
    dirs_dataset_paths = {
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

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset_paths, "DATASET")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.DATASET_ROOT, self.dirs_dataset_paths.get(key, ""))


CONST: _Const = _Const()

JSON_FILES_PATHS: ConfigFilePaths = ConfigFilePaths()
MPDRNN_PATHS: MPDRNNPaths = MPDRNNPaths()
DEV_DRNN_PATHS: DEVDRNNPaths = DEVDRNNPaths()
DATASET_FILES_PATHS: DatasetFilesPaths = DatasetFilesPaths()
