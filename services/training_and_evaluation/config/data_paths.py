import logging
import os

from utils.common import setup_logger


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++ C O N S T ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _Const(object):
    setup_logger()

    # Select user and according paths
    STORAGE_ROOT = os.getenv('STORAGE_ROOT', "D:/storage/lion17")
    DATASET_ROOT = os.getenv("DATASET_ROOT", "C:/Users/hmark/Documents/Datasets/datasets")
    PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/")


    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   D I R C T O R I E S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def create_directories(cls, dir_path) -> None:
        """
        Class method that creates the missing directory path.
        :param dir_path: This is the specific directory path that the function checks and creates.
        :return: None
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
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
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.dirs_config_paths.get(key, ""))


class MPDRNNPaths(_Const):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key: str) -> str:
        """
        Method that dynamically generates and returns the absolute path for MPDRNN results and hyperparams.
        :param key: The dictionary key representing the dataset and data type (results/hyperparam).
        :return: Full absolute path.
        """
        for prefix in ["results_", "hyperparam_"]:
            if key.startswith(prefix):
                dataset = key.replace(prefix, "")
                folder_type = "hyperparam" if prefix == "hyperparam_" else "results"

                full_path = os.path.join(
                    self.STORAGE_ROOT,
                    f"networks/mpdrnn/data/{folder_type}/{dataset}",
                )
                self.create_directories(full_path)
                return full_path
        return os.path.join(self.STORAGE_ROOT, key)


class FCNNPaths(_Const):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key: str) -> str:
        """
        Method that dynamically generates and returns the absolute path for FCNN weights, logs, results and hyperparams.
        :param key: The dictionary key representing the dataset and data type (sw/logs/results/hyperparam).
        :return: Full absolute path.
        """
        prefixes = {
            "sw_": "networks/fcnn/saved_weights_fcnn",
            "logs_": "networks/fcnn/logs_fcnn",
            "results_": "networks/fcnn/results_fcnn",
            "hyperparam_tuning_": "networks/fcnn/hyperparam_tuning"
        }

        for prefix, sub_path in prefixes.items():
            if key.startswith(prefix):
                dataset = key.replace(prefix, "")
                full_path = os.path.join(self.STORAGE_ROOT, f"{sub_path}/{dataset}")
                self.create_directories(full_path)
                return full_path
        return os.path.join(self.STORAGE_ROOT, key)


class HELMPaths(_Const):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key: str) -> str:
        """
        Method that dynamically generates and returns the absolute path for HELM confusion matrices, results and hyperparams.
        :param key: The dictionary key representing the dataset and data type (helm/results/hyperparam).
        :return: Full absolute path.
        """
        prefixes = {
            "helm_": "networks/helm/images/confusion_matrix",
            "results_": "networks/helm/data/results",
            "hyperparam_tuning_": "networks/helm/data/hyperparam_tuning"
        }

        for prefix, sub_path in prefixes.items():
            if key.startswith(prefix):
                dataset = key.replace(prefix, "")
                full_path = os.path.join(self.STORAGE_ROOT, f"{sub_path}/{dataset}")
                self.create_directories(full_path)
                return full_path
        return os.path.join(self.STORAGE_ROOT, key)


class DatasetFilesPaths(_Const):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key: str) -> str:
        """
        Method that dynamically generates and returns the absolute path for the raw datasets.
        :param key: The dictionary key representing the dataset name.
        :return: Full absolute path.
        """
        prefix = "dataset_path_"
        if key.startswith(prefix):
            dataset = key.replace(prefix, "")
            full_path = os.path.join(self.DATASET_ROOT, dataset)
            self.create_directories(full_path)
            return full_path
        return os.path.join(self.DATASET_ROOT, key)


CONST: _Const = _Const()

JSON_FILES_PATHS: ConfigFilePaths = ConfigFilePaths()
MPDRNN_PATHS: MPDRNNPaths = MPDRNNPaths()
FCNN_PATHS: FCNNPaths = FCNNPaths()
HELM_PATHS: HELMPaths = HELMPaths()
DATASET_FILES_PATHS: DatasetFilesPaths = DatasetFilesPaths()
