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
                "D:/storage/lion17",
            "DATASET_ROOT":
                "D:/storage/lion17/datasets",
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
            "config/json_files/MPDRNN_config.json",
        "config_schema_mpdrnn":
            "config/json_files/MPDRNN_config_schema.json",

        "config_fcnn":
            "config/json_files/FCNN_config.json",
        "config_schema_fcnn":
            "config/json_files/FCNN_config_schema.json",

        "config_helm":
            "config/json_files/HELM_config.json",
        "config_schema_helm":
            "config/json_files/HELM_config_schema.json",
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
    dirs_data_paths = {
        "results_connect4":
            "networks/mpdrnn/data/results/connect4",
        "results_isolete":
            "networks/mpdrnn/data/results/isolete",
        "results_letter":
            "networks/mpdrnn/data/results/letter",
        "results_mnist":
            "networks/mpdrnn/data/results/mnist",
        "results_mnist_fashion":
            "networks/mpdrnn/data/results/mnist_fashion",
        "results_musk2":
            "networks/mpdrnn/data/results/musk2",
        "results_optdigits":
            "networks/mpdrnn/data/results/optdigits",
        "results_page_blocks":
            "networks/mpdrnn/data/results/page_blocks",
        "results_segment":
            "networks/mpdrnn/data/results/segment",
        "results_shuttle":
            "networks/mpdrnn/data/results/shuttle",
        "results_usps":
            "networks/mpdrnn/data/results/usps",

        "hyperparam_connect4":
            "networks/mpdrnn/data/hyperparam/connect4",
        "hyperparam_isolete":
            "networks/mpdrnn/data/hyperparam/isolete",
        "hyperparam_letter":
            "networks/mpdrnn/data/hyperparam/letter",
        "hyperparam_mnist":
            "networks/mpdrnn/data/hyperparam/mnist",
        "hyperparam_mnist_fashion":
            "networks/mpdrnn/data/hyperparam/mnist_fashion",
        "hyperparam_musk2":
            "networks/mpdrnn/data/hyperparam/musk2",
        "hyperparam_optdigits":
            "networks/mpdrnn/data/hyperparam/optdigits",
        "hyperparam_page_blocks":
            "networks/mpdrnn/data/hyperparam/page_blocks",
        "hyperparam_segment":
            "networks/mpdrnn/data/hyperparam/segment",
        "hyperparam_shuttle":
            "networks/mpdrnn/data/hyperparam/shuttle",
        "hyperparam_usps":
            "networks/mpdrnn/data/hyperparam/usps"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_data_paths, "STORAGE")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.dirs_data_paths.get(key, ""))


class FCNNPaths(_Const):
    dirs_data_paths = {
        # FCNN saved weights
        "sw_connect4":
            "networks/fcnn/saved_weights_fcnn/connect4",
        "sw_isolete":
            "networks/fcnn/saved_weights_fcnn/isolete",
        "sw_letter":
            "networks/fcnn/saved_weights_fcnn/letter",
        "sw_mnist":
            "networks/fcnn/saved_weights_fcnn/mnist",
        "sw_mnist_fashion":
            "networks/fcnn/saved_weights_fcnn/mnist_fashion",
        "sw_musk2":
            "networks/fcnn/saved_weights_fcnn/musk2",
        "sw_optdigits":
            "networks/fcnn/saved_weights_fcnn/optdigits",
        "sw_page_blocks":
            "networks/fcnn/saved_weights_fcnn/page_blocks",
        "sw_segment":
            "networks/fcnn/saved_weights_fcnn/segment",
        "sw_shuttle":
            "networks/fcnn/saved_weights_fcnn/shuttle",
        "sw_usps":
            "networks/fcnn/saved_weights_fcnn/usps",

        # FCNN logs
        "logs_connect4":
            "networks/fcnn/logs_fcnn/connect4",
        "logs_isolete":
            "networks/fcnn/logs_fcnn/isolete",
        "logs_letter":
            "networks/fcnn/logs_fcnn/letter",
        "logs_mnist":
            "networks/fcnn/logs_fcnn/mnist",
        "logs_mnist_fashion":
            "networks/fcnn/logs_fcnn/mnist_fashion",
        "logs_musk2":
            "networks/fcnn/logs_fcnn/musk2",
        "logs_optdigits":
            "networks/fcnn/logs_fcnn/optdigits",
        "logs_page_blocks":
            "networks/fcnn/logs_fcnn/page_blocks",
        "logs_segment":
            "networks/fcnn/logs_fcnn/segment",
        "logs_shuttle":
            "networks/fcnn/logs_fcnn/shuttle",
        "logs_usps":
            "networks/fcnn/logs_fcnn/usps",

        "results_connect4":
            "networks/fcnn/results_fcnn/connect4",
        "results_isolete":
            "networks/fcnn/results_fcnn/isolete",
        "results_letter":
            "networks/fcnn/results_fcnn/letter",
        "results_mnist":
            "networks/fcnn/results_fcnn/mnist",
        "results_mnist_fashion":
            "networks/fcnn/results_fcnn/mnist_fashion",
        "results_musk2":
            "networks/fcnn/results_fcnn/musk2",
        "results_optdigits":
            "networks/fcnn/results_fcnn/optdigits",
        "results_page_blocks":
            "networks/fcnn/results_fcnn/page_blocks",
        "results_segment":
            "networks/fcnn/results_fcnn/segment",
        "results_shuttle":
            "networks/fcnn/results_fcnn/shuttle",
        "results_usps":
            "networks/fcnn/results_fcnn/usps",

        "hyperparam_tuning_connect4":
            "networks/fcnn/hyperparam_tuning/connect4",
        "hyperparam_tuning_isolete":
            "networks/fcnn/hyperparam_tuning/isolete",
        "hyperparam_tuning_letter":
            "networks/fcnn/hyperparam_tuning/letter",
        "hyperparam_tuning_mnist":
            "networks/fcnn/hyperparam_tuning/mnist",
        "hyperparam_tuning_mnist_fashion":
            "networks/fcnn/hyperparam_tuning/mnist_fashion",
        "hyperparam_tuning_musk2":
            "networks/fcnn/hyperparam_tuning/musk2",
        "hyperparam_tuning_optdigits":
            "networks/fcnn/hyperparam_tuning/optdigits",
        "hyperparam_tuning_page_blocks":
            "networks/fcnn/hyperparam_tuning/page_blocks",
        "hyperparam_tuning_segment":
            "networks/fcnn/hyperparam_tuning/segment",
        "hyperparam_tuning_shuttle":
            "networks/fcnn/hyperparam_tuning/shuttle",
        "hyperparam_tuning_usps":
            "networks/fcnn/hyperparam_tuning/usps",
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_data_paths, "STORAGE")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.dirs_data_paths.get(key, ""))


class HELMPaths(_Const):
    dirs_data_paths = {
        "helm_connect4":
            "networks/helm/images/confusion_matrix/connect4",
        "helm_isolete":
            "networks/helm/images/confusion_matrix/isolete",
        "helm_letter":
            "networks/helm/images/confusion_matrix/letter",
        "helm_mnist":
            "networks/helm/images/confusion_matrix/mnist",
        "helm_mnist_fashion":
            "networks/helm/images/confusion_matrix/mnist_fashion",
        "helm_musk2":
            "networks/helm/images/confusion_matrix/musk2",
        "helm_optdigits":
            "networks/helm/images/confusion_matrix/optdigits",
        "helm_page_blocks":
            "networks/helm/images/confusion_matrix/page_blocks",
        "helm_segment":
            "networks/helm/images/confusion_matrix/segment",
        "helm_shuttle":
            "networks/helm/images/confusion_matrix/shuttle",
        "helm_usps":
            "networks/helm/images/confusion_matrix/usps",

        "results_connect4":
            "networks/helm/data/results/connect4",
        "results_isolete":
            "networks/helm/data/results/isolete",
        "results_letter":
            "networks/helm/data/results/letter",
        "results_mnist":
            "networks/helm/data/results/mnist",
        "results_mnist_fashion":
            "networks/helm/data/results/mnist_fashion",
        "results_musk2":
            "networks/helm/data/results/musk2",
        "results_optdigits":
            "networks/helm/data/results/optdigits",
        "results_page_blocks":
            "networks/helm/data/results/page_blocks",
        "results_segment":
            "networks/helm/data/results/segment",
        "results_shuttle":
            "networks/helm/data/results/shuttle",
        "results_usps":
            "networks/helm/data/results/usps",

        "hyperparam_tuning_connect4":
            "networks/helm/data/hyperparam_tuning/connect4",
        "hyperparam_tuning_isolete":
            "networks/helm/data/hyperparam_tuning/isolete",
        "hyperparam_tuning_letter":
            "networks/helm/data/hyperparam_tuning/letter",
        "hyperparam_tuning_mnist":
            "networks/helm/data/hyperparam_tuning/mnist",
        "hyperparam_tuning_mnist_fashion":
            "networks/helm/data/hyperparam_tuning/mnist_fashion",
        "hyperparam_tuning_musk2":
            "networks/helm/data/hyperparam_tuning/musk2",
        "hyperparam_tuning_optdigits":
            "networks/helm/data/hyperparam_tuning/optdigits",
        "hyperparam_tuning_page_blocks":
            "networks/helm/data/hyperparam_tuning/page_blocks",
        "hyperparam_tuning_segment":
            "networks/helm/data/hyperparam_tuning/segment",
        "hyperparam_tuning_shuttle":
            "networks/helm/data/hyperparam_tuning/shuttle",
        "hyperparam_tuning_usps":
            "networks/helm/data/hyperparam_tuning/usps"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_data_paths, "STORAGE")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.dirs_data_paths.get(key, ""))


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
        "dataset_path_usps":
            "usps"
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
FCNN_PATHS: FCNNPaths = FCNNPaths()
HELM_PATHS: HELMPaths = HELMPaths()
DATASET_FILES_PATHS: DatasetFilesPaths = DatasetFilesPaths()
