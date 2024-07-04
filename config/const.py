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
            "PROJECT_ROOT":
                "D:/storage/ELM",
            "DATASET_ROOT":
                "D:/storage/ELM/datasets",
        }
    }

    if user in root_mapping:
        root_info = root_mapping[user]
        PROJECT_ROOT = root_info["PROJECT_ROOT"]
        DATASET_ROOT = root_info["DATASET_ROOT"]
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
            if root_type == "PROJECT":
                dir_path = os.path.join(cls.PROJECT_ROOT, path)
            elif root_type == "DATASET":
                dir_path = os.path.join(cls.DATASET_ROOT, path)
            else:
                raise ValueError("Wrong root type!")

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"Directory {dir_path} has been created")


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
        self.create_directories(self.dirs_dataset_paths, "PROJECT")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.dirs_dataset_paths.get(key, ""))


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
        self.create_directories(self.dirs_dataset_paths, "DATASET")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.DATASET_ROOT, self.dirs_dataset_paths.get(key, ""))


CONST: _Const = _Const()

MPDRNN_PATHS: MPDRNNPaths = MPDRNNPaths()
ELM_DATASET_FILES_PATHS: ELMDatasetFilesPaths = ELMDatasetFilesPaths()
