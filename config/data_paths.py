import logging
import os

from utils.utils import setup_logger


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++ C O N S T ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _Const(object):
    setup_logger()

    # Select user and according paths
    user = os.getlogin()
    root_mapping = {
        "ricsi": {
            "STORAGE_ROOT":
                "D:/storage/Journal2",
            "DATASET_ROOT":
                "D:/storage/Journal2/datasets",
            "PROJECT_ROOT":
                "C:/Users/ricsi/Documents/research/Multi_Phase_Deep_Random_Neural_Network",
        }
    }

    if user in root_mapping:
        root_info = root_mapping[user]
        STORAGE_ROOT = root_info["STORAGE_ROOT"]
        DATASET_ROOT = root_info["DATASET_ROOT"]
        PROJECT_ROOT = root_info["PROJECT_ROOT"]
    else:
        raise ValueError("Wrong user!")

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   D I R C T O R I E S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
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
            "config/MPDRNN_config.json",
        "config_ipmdrnn":
            "config/IPMPDRNN_config.json",
        "config_fcnn":
            "config/FCNN_config.json",
        "config_schema_mpdrnn":
            "config/MPDRNN_config_schema.json",
        "config_schema_ipmdrnn":
            "config/IPMPDRNN_config_schema.json",
        "config_schema_fcnn":
            "config/FCNN_config_schema.json"
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
        # Confusion matrices
        "cm_connect4":
            "mpdrnn/images/confusion_matrix/connect4",
        "cm_isolete":
            "mpdrnn/images/confusion_matrix/isolete",
        "cm_letter":
            "mpdrnn/images/confusion_matrix/letter",
        "cm_mnist":
            "mpdrnn/images/confusion_matrix/mnist",
        "cm_mnist_fashion":
            "mpdrnn/images/confusion_matrix/mnist_fashion",
        "cm_musk2":
            "mpdrnn/images/confusion_matrix/musk2",
        "cm_optdigits":
            "mpdrnn/images/confusion_matrix/optdigits",
        "cm_page_blocks":
            "mpdrnn/images/confusion_matrix/page_blocks",
        "cm_segment":
            "mpdrnn/images/confusion_matrix/segment",
        "cm_shuttle":
            "mpdrnn/images/confusion_matrix/shuttle",
        "cm_spambase":
            "mpdrnn/images/confusion_matrix/spambase",
        "cm_usps":
            "mpdrnn/images/confusion_matrix/usps",
        "cm_satimages":
            "mpdrnn/images/confusion_matrix/satimages",

        "metrics_connect4":
            "mpdrnn/images/metrics/connect4",
        "metrics_isolete":
            "mpdrnn/images/metrics/isolete",
        "metrics_letter":
            "mpdrnn/images/metrics/letter",
        "metrics_mnist":
            "mpdrnn/images/metrics/mnist",
        "metrics_mnist_fashion":
            "mpdrnn/images/metrics/mnist_fashion",
        "metrics_musk2":
            "mpdrnn/images/metrics/musk2",
        "metrics_optdigits":
            "mpdrnn/images/metrics/optdigits",
        "metrics_page_blocks":
            "mpdrnn/images/metrics/page_blocks",
        "metrics_segment":
            "mpdrnn/images/metrics/segment",
        "metrics_shuttle":
            "mpdrnn/images/metrics/shuttle",
        "metrics_spambase":
            "mpdrnn/images/metrics/spambase",
        "metrics_usps":
            "mpdrnn/images/metrics/usps",
        "metrics_satimages":
            "mpdrnn/images/metrics/satimages",

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
            "mpdrnn/data/results/satimages"
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


class IPMPDRNNPaths(_Const):
    dirs_dataset_paths = {
        # Confusion matrices
        "cm_connect4":
            "ipmpdrnn/images/confusion_matrix/connect4",
        "cm_isolete":
            "ipmpdrnn/images/confusion_matrix/isolete",
        "cm_letter":
            "ipmpdrnn/images/confusion_matrix/letter",
        "cm_mnist":
            "ipmpdrnn/images/confusion_matrix/mnist",
        "cm_mnist_fashion":
            "ipmpdrnn/images/confusion_matrix/mnist_fashion",
        "cm_musk2":
            "ipmpdrnn/images/confusion_matrix/musk2",
        "cm_optdigits":
            "ipmpdrnn/images/confusion_matrix/optdigits",
        "cm_page_blocks":
            "ipmpdrnn/images/confusion_matrix/page_blocks",
        "cm_segment":
            "ipmpdrnn/images/confusion_matrix/segment",
        "cm_shuttle":
            "ipmpdrnn/images/confusion_matrix/shuttle",
        "cm_spambase":
            "ipmpdrnn/images/confusion_matrix/spambase",
        "cm_usps":
            "ipmpdrnn/images/confusion_matrix/usps",
        "cm_satimages":
            "ipmpdrnn/images/confusion_matrix/satimages",

        "metrics_connect4":
            "ipmpdrnn/images/metrics/connect4",
        "metrics_isolete":
            "ipmpdrnn/images/metrics/isolete",
        "metrics_letter":
            "ipmpdrnn/images/metrics/letter",
        "metrics_mnist":
            "ipmpdrnn/images/metrics/mnist",
        "metrics_mnist_fashion":
            "ipmpdrnn/images/metrics/mnist_fashion",
        "metrics_musk2":
            "ipmpdrnn/images/metrics/musk2",
        "metrics_optdigits":
            "ipmpdrnn/images/metrics/optdigits",
        "metrics_page_blocks":
            "ipmpdrnn/images/metrics/page_blocks",
        "metrics_segment":
            "ipmpdrnn/images/metrics/segment",
        "metrics_shuttle":
            "ipmpdrnn/images/metrics/shuttle",
        "metrics_spambase":
            "ipmpdrnn/images/metrics/spambase",
        "metrics_usps":
            "ipmpdrnn/images/metrics/usps",
        "metrics_satimages":
            "ipmpdrnn/images/metrics/satimages",

        "results_connect4":
            "ipmpdrnn/data/results/connect4",
        "results_isolete":
            "ipmpdrnn/data/results/isolete",
        "results_letter":
            "ipmpdrnn/data/results/letter",
        "results_mnist":
            "ipmpdrnn/data/results/mnist",
        "results_mnist_fashion":
            "ipmpdrnn/data/results/mnist_fashion",
        "results_musk2":
            "ipmpdrnn/data/results/musk2",
        "results_optdigits":
            "ipmpdrnn/data/results/optdigits",
        "results_page_blocks":
            "ipmpdrnn/data/results/page_blocks",
        "results_segment":
            "ipmpdrnn/data/results/segment",
        "results_shuttle":
            "ipmpdrnn/data/results/shuttle",
        "results_spambase":
            "ipmpdrnn/data/results/spambase",
        "results_usps":
            "ipmpdrnn/data/results/usps",
        "results_satimages":
            "ipmpdrnn/data/results/satimages"
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


class FCNNPaths(_Const):
    dirs_dataset_paths = {
        # FCNN saved weights
        "sw_connect4":
            "fcnn_data/saved_weights_fcnn/connect4",
        "sw_isolete":
            "fcnn_data/saved_weights_fcnn/isolete",
        "sw_letter":
            "fcnn_data/saved_weights_fcnn/letter",
        "sw_mnist":
            "fcnn_data/saved_weights_fcnn/mnist",
        "sw_mnist_fashion":
            "fcnn_data/saved_weights_fcnn/mnist_fashion",
        "sw_musk2":
            "fcnn_data/saved_weights_fcnn/musk2",
        "sw_optdigits":
            "fcnn_data/saved_weights_fcnn/optdigits",
        "sw_page_blocks":
            "fcnn_data/saved_weights_fcnn/page_blocks",
        "sw_segment":
            "fcnn_data/saved_weights_fcnn/segment",
        "sw_shuttle":
            "fcnn_data/saved_weights_fcnn/shuttle",
        "sw_spambase":
            "fcnn_data/saved_weights_fcnn/spambase",
        "sw_usps":
            "fcnn_data/saved_weights_fcnn/usps",
        "sw_satimages":
            "fcnn_data/saved_weights_fcnn/satimages",

        # FCNN logs
        "logs_connect4":
            "fcnn_data/logs_fcnn/connect4",
        "logs_isolete":
            "fcnn_data/logs_fcnn/isolete",
        "logs_letter":
            "fcnn_data/logs_fcnn/letter",
        "logs_mnist":
            "fcnn_data/logs_fcnn/mnist",
        "logs_mnist_fashion":
            "fcnn_data/logs_fcnn/mnist_fashion",
        "logs_musk2":
            "fcnn_data/logs_fcnn/musk2",
        "logs_optdigits":
            "fcnn_data/logs_fcnn/optdigits",
        "logs_page_blocks":
            "fcnn_data/logs_fcnn/page_blocks",
        "logs_segment":
            "fcnn_data/logs_fcnn/segment",
        "logs_shuttle":
            "fcnn_data/logs_fcnn/shuttle",
        "logs_spambase":
            "fcnn_data/logs_fcnn/spambase",
        "logs_usps":
            "fcnn_data/logs_fcnn/usps",
        "logs_satimages":
            "fcnn_data/logs_fcnn/satiamges",

        "results_connect4":
            "fcnn_data/results_fcnn/connect4",
        "results_isolete":
            "fcnn_data/results_fcnn/isolete",
        "results_letter":
            "fcnn_data/results_fcnn/letter",
        "results_mnist":
            "fcnn_data/results_fcnn/mnist",
        "results_mnist_fashion":
            "fcnn_data/results_fcnn/mnist_fashion",
        "results_musk2":
            "fcnn_data/results_fcnn/musk2",
        "results_optdigits":
            "fcnn_data/results_fcnn/optdigits",
        "results_page_blocks":
            "fcnn_data/results_fcnn/page_blocks",
        "results_segment":
            "fcnn_data/results_fcnn/segment",
        "results_shuttle":
            "fcnn_data/results_fcnn/shuttle",
        "results_spambase":
            "fcnn_data/results_fcnn/spambase",
        "results_usps":
            "fcnn_data/results_fcnn/usps",

        "hyperparam_tuning_connect4":
            "fcnn_data/hyperparam_tuning/connect4",
        "hyperparam_tuning_isolete":
            "fcnn_data/hyperparam_tuning/isolete",
        "hyperparam_tuning_letter":
            "fcnn_data/hyperparam_tuning/letter",
        "hyperparam_tuning_mnist":
            "fcnn_data/hyperparam_tuning/mnist",
        "hyperparam_tuning_mnist_fashion":
            "fcnn_data/hyperparam_tuning/mnist_fashion",
        "hyperparam_tuning_musk2":
            "fcnn_data/hyperparam_tuning/musk2",
        "hyperparam_tuning_optdigits":
            "fcnn_data/hyperparam_tuning/optdigits",
        "hyperparam_tuning_page_blocks":
            "fcnn_data/hyperparam_tuning/page_blocks",
        "hyperparam_tuning_segment":
            "fcnn_data/hyperparam_tuning/segment",
        "hyperparam_tuning_shuttle":
            "fcnn_data/hyperparam_tuning/shuttle",
        "hyperparam_tuning_spambase":
            "fcnn_data/hyperparam_tuning/spambase",
        "hyperparam_tuning_usps":
            "fcnn_data/hyperparam_tuning/usps"
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
        # Confusion matrices
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
IPMPDRNN_PATHS: IPMPDRNNPaths = IPMPDRNNPaths()
FCNN_PATHS: FCNNPaths = FCNNPaths()
DATASET_FILES_PATHS: DatasetFilesPaths = DatasetFilesPaths()
