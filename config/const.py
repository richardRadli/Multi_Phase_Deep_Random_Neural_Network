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
            "PROJECT_ROOT_ELM":
                "D:/storage/ELM",
            "DATASET_ROOT_ELM":
                "D:/storage/ELM/datasets",
            "PROJECT_ROOT_VIT":
                "D:/storage/ViT",
            "DATASET_ROOT_VIT":
                "D:/storage/ViT/datasets",
        }
    }

    if user in root_mapping:
        root_info = root_mapping[user]
        PROJECT_ROOT_ELM = root_info["PROJECT_ROOT_ELM"]
        DATASET_ROOT_ELM = root_info["DATASET_ROOT_ELM"]
        PROJECT_ROOT_VIT = root_info["PROJECT_ROOT_VIT"]
        DATASET_ROOT_VIT = root_info["DATASET_ROOT_VIT"]
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
        :param root_type: Either PROJECT or DATASET.
        :return: None
        """

        for _, path in dirs.items():
            if root_type == "PROJECT_ELM":
                dir_path = os.path.join(cls.PROJECT_ROOT_ELM, path)
            elif root_type == "DATASET_ELM":
                dir_path = os.path.join(cls.DATASET_ROOT_ELM, path)
            elif root_type == "PROJECT_VIT":
                dir_path = os.path.join(cls.PROJECT_ROOT_VIT, path)
            elif root_type == "DATASET_VIT":
                dir_path = os.path.join(cls.DATASET_ROOT_VIT, path)
            else:
                raise ValueError("Wrong root type!")

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"Directory {dir_path} has been created")


class FCNNPaths(_Const):
    dirs_dataset_paths = {
        # FCNN saved weights
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
        "sw_iris":
            "fcnn/saved_weights_fcnn/iris",
        "sw_forest":
            "fcnn/saved_weights_fcnn/forest",
        "sw_satimages":
            "fcnn/saved_weights_fcnn/satimages",

        # FCNN logs
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
        "logs_iris":
            "fcnn/logs_fcnn/iris",
        "logs_forest":
            "fcnn/logs_fcnn/forest",
        "logs_satimages":
            "fcnn/logs_fcnn/satiamges",

        "results_connect4":
            "fcnn/results_fcnn/connect4",
        "results_forest":
            "fcnn/results_fcnn/forest",
        "results_iris":
            "fcnn/results_fcnn/iris",
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
        "results_segment":
            "fcnn/results_fcnn/segment",
        "results_shuttle":
            "fcnn/results_fcnn/shuttle",
        "results_spambase":
            "fcnn/results_fcnn/spambase",
        "results_usps":
            "fcnn/results_fcnn/usps",
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset_paths, "PROJECT_ELM")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT_ELM, self.dirs_dataset_paths.get(key, ""))


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
        "cm_iris":
            "mpdrnn/images/confusion_matrix/iris",
        "cm_forest":
            "mpdrnn/images/confusion_matrix/forest",
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
        "metrics_iris":
            "mpdrnn/images/metrics/iris",
        "metrics_forest":
            "mpdrnn/images/metrics/forest",
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
        "results_iris":
            "mpdrnn/data/results/iris",
        "results_forest":
            "mpdrnn/data/results/forest",
        "results_satimages":
            "mpdrnn/data/results/satimages"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset_paths, "PROJECT_ELM")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT_ELM, self.dirs_dataset_paths.get(key, ""))


class BWELMPaths(_Const):
    dirs_dataset_paths = {
        # Confusion matrices
        "cm_mnist":
            "bwelm/images/confusion_matrix/mnist",
        "cm_shuttle":
            "bwelm/images/confusion_matrix/shuttle",
        "cm_iris":
            "bwelm/images/confusion_matrix/iris",
        "cm_forest":
            "bwelm/images/confusion_matrix/forest",
        "cm_satimages":
            "bwelm/images/confusion_matrix/satimages",

        "metrics_mnist":
            "bwelm/images/metrics/mnist",
        "metrics_shuttle":
            "bwelm/images/metrics/shuttle",
        "metrics_iris":
            "bwelm/images/metrics/iris",
        "metrics_forest":
            "bwelm/images/metrics/forest",
        "metrics_satimages":
            "bwelm/images/metrics/satimages",

        "results_mnist":
            "bwelm/data/results/mnist",
        "results_shuttle":
            "bwelm/data/results/shuttle",
        "results_iris":
            "bwelm/data/results/iris",
        "results_forest":
            "bwelm/data/results/forest",
        "results_satimages":
            "bwelm/data/results/satimages"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset_paths, "PROJECT_ELM")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT_ELM, self.dirs_dataset_paths.get(key, ""))


class ELMDatasetFilesPaths(_Const):
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
        "dataset_path_iris":
            "iris",
        "dataset_path_forest":
            "forest",
        "dataset_path_satimages":
            "satimages",
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset_paths, "DATASET_ELM")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.DATASET_ROOT_ELM, self.dirs_dataset_paths.get(key, ""))


class ViTELMPaths(_Const):
    dirs_dataset_paths = {
        "ViT_weights_cifar10":
            "data/ViT_weights/cifar10",
        "ViT_weights_mnist":
            "data/ViT_weights/mnist",

        "combined_weights_cifar10":
            "data/combined_weights/cifar10",
        "combined_weights_mist":
            "data/combined_weights/mnist",

        "logs_cifar10":
            "data/logs/cifar10",
        "logs_mnist":
            "data/logs/mnist",

        "results_cifar10":
            "data/results/cifar10",
        "results_mnist":
            "data/results/mnist",

        "cm_cifar10":
            "images/confusion_matrix/cifar10",
        "cm_mnist":
            "images/confusion_matrix/mnist"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset_paths, "PROJECT_VIT")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT_VIT, self.dirs_dataset_paths.get(key, ""))


class ViTELMDatasetFilesPaths(_Const):
    dirs_dataset_paths = {
        "original_files_dataset_path_cifar10":
            "cifar10/original_files",
        "original_files_dataset_path_mnist":
            "mnist/original_files"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset_paths, "DATASET_VIT")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.DATASET_ROOT_VIT, self.dirs_dataset_paths.get(key, ""))


CONST: _Const = _Const()
BWELM_PATHS: BWELMPaths = BWELMPaths()
FCNN_PATHS: FCNNPaths = FCNNPaths()
MPDRNN_PATHS: MPDRNNPaths = MPDRNNPaths()
ViTELM_PATHS: ViTELMPaths = ViTELMPaths()
ELM_DATASET_FILES_PATHS: ELMDatasetFilesPaths = ELMDatasetFilesPaths()
ViTELM_DATASET_FILES_PATHS: ViTELMDatasetFilesPaths = ViTELMDatasetFilesPaths()
