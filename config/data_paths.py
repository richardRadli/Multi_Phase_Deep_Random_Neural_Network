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
            "config/json_files/MPDRNN_config.json",
        "config_schema_mpdrnn":
            "config/json_files/MPDRNN_config_schema.json",

        "config_ipmpdrnn":
            "config/json_files/IPMPDRNN_config.json",
        "config_schema_ipmpdrnn":
            "config/json_files/IPMPDRNN_config_schema.json",

        "config_fcnn":
            "config/json_files/FCNN_config.json",
        "config_schema_fcnn":
            "config/json_files/FCNN_config_schema.json",

        "config_helm":
            "config/json_files/HELM_config.json",
        "config_schema_helm":
            "config/json_files/HELM_config_schema.json"
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
        "results_adult":
            "mpdrnn/data/results/adult",
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

        "hyperparam_adult":
            "mpdrnn/data/hyperparam/adult",
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


class IPMPDRNNPaths(_Const):
    dirs_dataset_paths = {
        "results_adult":
            "ipmpdrnn/data/results/adult",
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
            "ipmpdrnn/data/results/satimages",
        "results_wall":
            "ipmpdrnn/data/results/wall",
        "results_waveform":
            "ipmpdrnn/data/results/waveform",

        "hyperparam_adult":
            "ipmpdrnn/data/hyperparam_tuning/adult",
        "hyperparam_connect4":
            "ipmpdrnn/data/hyperparam_tuning/connect4",
        "hyperparam_isolete":
            "ipmpdrnn/data/hyperparam_tuning/isolete",
        "hyperparam_letter":
            "ipmpdrnn/data/hyperparam_tuning/letter",
        "hyperparam_mnist":
            "ipmpdrnn/data/hyperparam_tuning/mnist",
        "hyperparam_mnist_fashion":
            "ipmpdrnn/data/hyperparam_tuning/mnist_fashion",
        "hyperparam_musk2":
            "ipmpdrnn/data/hyperparam_tuning/musk2",
        "hyperparam_optdigits":
            "ipmpdrnn/data/hyperparam_tuning/optdigits",
        "hyperparam_page_blocks":
            "ipmpdrnn/data/hyperparam_tuning/page_blocks",
        "hyperparam_segment":
            "ipmpdrnn/data/hyperparam_tuning/segment",
        "hyperparam_satimages":
            "ipmpdrnn/data/hyperparam_tuning/satimages",
        "hyperparam_shuttle":
            "ipmpdrnn/data/hyperparam_tuning/shuttle",
        "hyperparam_spambase":
            "ipmpdrnn/data/hyperparam_tuning/spambase",
        "hyperparam_usps":
            "ipmpdrnn/data/hyperparam_tuning/usps",
        "hyperparam_wall":
            "ipmpdrnn/data/hyperparam_tuning/wall",
        "hyperparam_waveform":
            "ipmpdrnn/data/hyperparam_tuning/waveform"
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
        "sw_adult":
            "fcnn/saved_weights_fcnn/adult",
        "sw_connect4":
            "fcnn/saved_weights_fcnn/connect4",
        "sw_isolete":
            "fcnn/saved_weights_fcnn/isolete",
        "sw_letter":
            "fcnn/saved_weights_fcnn/letter",
        "sw_mnist":
            "fcnn/saved_weights_fcnn/mnist",
        "sw_mnist_fashion":
            "fcnn/saved_weights_fcnn/mnist_fashion",
        "sw_musk2":
            "fcnn/saved_weights_fcnn/musk2",
        "sw_optdigits":
            "fcnn/saved_weights_fcnn/optdigits",
        "sw_page_blocks":
            "fcnn/saved_weights_fcnn/page_blocks",
        "sw_segment":
            "fcnn/saved_weights_fcnn/segment",
        "sw_shuttle":
            "fcnn/saved_weights_fcnn/shuttle",
        "sw_spambase":
            "fcnn/saved_weights_fcnn/spambase",
        "sw_usps":
            "fcnn/saved_weights_fcnn/usps",
        "sw_satimages":
            "fcnn/saved_weights_fcnn/satimages",
        "sw_wall":
            "fcnn/saved_weights_fcnn/wall",
        "sw_waveform":
            "fcnn/saved_weights_fcnn/waveform",

        # FCNN logs
        "logs_adult":
            "fcnn/logs_fcnn/adult",
        "logs_connect4":
            "fcnn/logs_fcnn/connect4",
        "logs_isolete":
            "fcnn/logs_fcnn/isolete",
        "logs_letter":
            "fcnn/logs_fcnn/letter",
        "logs_mnist":
            "fcnn/logs_fcnn/mnist",
        "logs_mnist_fashion":
            "fcnn/logs_fcnn/mnist_fashion",
        "logs_musk2":
            "fcnn/logs_fcnn/musk2",
        "logs_optdigits":
            "fcnn/logs_fcnn/optdigits",
        "logs_page_blocks":
            "fcnn/logs_fcnn/page_blocks",
        "logs_segment":
            "fcnn/logs_fcnn/segment",
        "logs_shuttle":
            "fcnn/logs_fcnn/shuttle",
        "logs_spambase":
            "fcnn/logs_fcnn/spambase",
        "logs_usps":
            "fcnn/logs_fcnn/usps",
        "logs_satimages":
            "fcnn/logs_fcnn/satiamges",
        "logs_wall":
            "fcnn/logs_fcnn/wall",
        "logs_waveform":
            "fcnn/logs_fcnn/waveform",

        "results_adult":
            "fcnn/results_fcnn/adult",
        "results_connect4":
            "fcnn/results_fcnn/connect4",
        "results_isolete":
            "fcnn/results_fcnn/isolete",
        "results_letter":
            "fcnn/results_fcnn/letter",
        "results_mnist":
            "fcnn/results_fcnn/mnist",
        "results_mnist_fashion":
            "fcnn/results_fcnn/mnist_fashion",
        "results_musk2":
            "fcnn/results_fcnn/musk2",
        "results_optdigits":
            "fcnn/results_fcnn/optdigits",
        "results_page_blocks":
            "fcnn/results_fcnn/page_blocks",
        "results_satimages":
            "fcnn/results_fcnn/satimages",
        "results_segment":
            "fcnn/results_fcnn/segment",
        "results_shuttle":
            "fcnn/results_fcnn/shuttle",
        "results_spambase":
            "fcnn/results_fcnn/spambase",
        "results_usps":
            "fcnn/results_fcnn/usps",
        "results_wall":
            "fcnn/results_fcnn/wall",
        "results_waveform":
            "fcnn/results_fcnn/waveform",

        "hyperparam_tuning_adult":
            "fcnn/hyperparam_tuning/adult",
        "hyperparam_tuning_connect4":
            "fcnn/hyperparam_tuning/connect4",
        "hyperparam_tuning_isolete":
            "fcnn/hyperparam_tuning/isolete",
        "hyperparam_tuning_letter":
            "fcnn/hyperparam_tuning/letter",
        "hyperparam_tuning_mnist":
            "fcnn/hyperparam_tuning/mnist",
        "hyperparam_tuning_mnist_fashion":
            "fcnn/hyperparam_tuning/mnist_fashion",
        "hyperparam_tuning_musk2":
            "fcnn/hyperparam_tuning/musk2",
        "hyperparam_tuning_optdigits":
            "fcnn/hyperparam_tuning/optdigits",
        "hyperparam_tuning_page_blocks":
            "fcnn/hyperparam_tuning/page_blocks",
        "hyperparam_tuning_segment":
            "fcnn/hyperparam_tuning/segment",
        "hyperparam_tuning_satimages":
            "fcnn/hyperparam_tuning/satimages",
        "hyperparam_tuning_shuttle":
            "fcnn/hyperparam_tuning/shuttle",
        "hyperparam_tuning_spambase":
            "fcnn/hyperparam_tuning/spambase",
        "hyperparam_tuning_usps":
            "fcnn/hyperparam_tuning/usps",
        "hyperparam_tuning_wall":
            "fcnn/hyperparam_tuning/wall",
        "hyperparam_tuning_waveform":
            "fcnn/hyperparam_tuning/waveform"
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


class HELMPaths(_Const):
    dirs_dataset_paths = {
        "helm_adult":
            "helm/images/confusion_matrix/adult",
        "helm_connect4":
            "helm/images/confusion_matrix/connect4",
        "helm_isolete":
            "helm/images/confusion_matrix/isolete",
        "helm_letter":
            "helm/images/confusion_matrix/letter",
        "helm_mnist":
            "helm/images/confusion_matrix/mnist",
        "helm_mnist_fashion":
            "helm/images/confusion_matrix/mnist_fashion",
        "helm_musk2":
            "helm/images/confusion_matrix/musk2",
        "helm_optdigits":
            "helm/images/confusion_matrix/optdigits",
        "helm_page_blocks":
            "helm/images/confusion_matrix/page_blocks",
        "helm_segment":
            "helm/images/confusion_matrix/segment",
        "helm_shuttle":
            "helm/images/confusion_matrix/shuttle",
        "helm_spambase":
            "helm/images/confusion_matrix/spambase",
        "helm_usps":
            "helm/images/confusion_matrix/usps",
        "helm_satimages":
            "helm/images/confusion_matrix/satimages",
        "helm_wall":
            "helm/images/confusion_matrix/wall",
        "helm_waveform":
            "helm/images/confusion_matrix/waveform",

        "results_adult":
            "helm/data/results/adult",
        "results_connect4":
            "helm/data/results/connect4",
        "results_isolete":
            "helm/data/results/isolete",
        "results_letter":
            "helm/data/results/letter",
        "results_mnist":
            "helm/data/results/mnist",
        "results_mnist_fashion":
            "helm/data/results/mnist_fashion",
        "results_musk2":
            "helm/data/results/musk2",
        "results_optdigits":
            "helm/data/results/optdigits",
        "results_page_blocks":
            "helm/data/results/page_blocks",
        "results_segment":
            "helm/data/results/segment",
        "results_shuttle":
            "helm/data/results/shuttle",
        "results_spambase":
            "helm/data/results/spambase",
        "results_usps":
            "helm/data/results/usps",
        "results_satimages":
            "helm/data/results/satimages",
        "results_wall":
            "helm/data/results/wall",
        "results_waveform":
            "helm/data/results/waveform",

        "hyperparam_tuning_adult":
            "helm/data/hyperparam_tuning/adult",
        "hyperparam_tuning_connect4":
            "helm/data/hyperparam_tuning/connect4",
        "hyperparam_tuning_isolete":
            "helm/data/hyperparam_tuning/isolete",
        "hyperparam_tuning_letter":
            "helm/data/hyperparam_tuning/letter",
        "hyperparam_tuning_mnist":
            "helm/data/hyperparam_tuning/mnist",
        "hyperparam_tuning_mnist_fashion":
            "helm/data/hyperparam_tuning/mnist_fashion",
        "hyperparam_tuning_musk2":
            "helm/data/hyperparam_tuning/musk2",
        "hyperparam_tuning_optdigits":
            "helm/data/hyperparam_tuning/optdigits",
        "hyperparam_tuning_page_blocks":
            "helm/data/hyperparam_tuning/page_blocks",
        "hyperparam_tuning_segment":
            "helm/data/hyperparam_tuning/segment",
        "hyperparam_tuning_satimages":
            "helm/data/hyperparam_tuning/satimages",
        "hyperparam_tuning_shuttle":
            "helm/data/hyperparam_tuning/shuttle",
        "hyperparam_tuning_spambase":
            "helm/data/hyperparam_tuning/spambase",
        "hyperparam_tuning_usps":
            "helm/data/hyperparam_tuning/usps",
        "hyperparam_tuning_wall":
            "helm/data/hyperparam_tuning/wall",
        "hyperparam_tuning_waveform":
            "helm/data/hyperparam_tuning/waveform"
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
        "dataset_path_adult":
            "adult",
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
IPMPDRNN_PATHS: IPMPDRNNPaths = IPMPDRNNPaths()
FCNN_PATHS: FCNNPaths = FCNNPaths()
HELM_PATHS: HELMPaths = HELMPaths()
DATASET_FILES_PATHS: DatasetFilesPaths = DatasetFilesPaths()
