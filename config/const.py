import logging
import os


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++ C O N S T ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _Const(object):
    # Select user and according paths
    user = os.getlogin()
    root_mapping = {
        "rrb12": {
            "PROJECT_ROOT":
                "C:/Users/rrb12/Documents/research/elm",
            "DATASET_ROOT":
                "C:/Users/rrb12/Documents/research/elm/datasets"
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
        "sw_yaleb":
            "fcnn_data/saved_weights_fcnn/yaleb",
        "sw_iris":
            "fcnn_data/saved_weights_fcnn/iris",
        "sw_forest":
            "fcnn_data/saved_weights_fcnn/forest",
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
        "logs_yaleb":
            "fcnn_data/logs_fcnn/yaleb",
        "logs_iris":
            "fcnn_data/logs_fcnn/iris",
        "logs_forest":
            "fcnn_data/logs_fcnn/forest",
        "logs_satimages":
            "fcnn_data/logs_fcnn/satiamges"
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


class MPDRNNPaths(_Const):
    dirs_dataset_paths = {
        # Confusion matrices
        "cm_connect4":
            "images/confusion_matrix/connect4",
        "cm_isolete":
            "images/confusion_matrix/isolete",
        "cm_letter":
            "images/confusion_matrix/letter",
        "cm_mnist":
            "images/confusion_matrix/mnist",
        "cm_mnist_fashion":
            "images/confusion_matrix/mnist_fashion",
        "cm_musk2":
            "images/confusion_matrix/musk2",
        "cm_optdigits":
            "images/confusion_matrix/optdigits",
        "cm_page_blocks":
            "images/confusion_matrix/page_blocks",
        "cm_segment":
            "images/confusion_matrix/segment",
        "cm_shuttle":
            "images/confusion_matrix/shuttle",
        "cm_spambase":
            "images/confusion_matrix/spambase",
        "cm_usps":
            "images/confusion_matrix/usps",
        "cm_yaleb":
            "images/confusion_matrix/yaleb",
        "cm_iris":
            "images/confusion_matrix/iris",
        "cm_forest":
            "images/confusion_matrix/forest",
        "cm_satimages":
            "images/confusion_matrix/satimages",

        "metrics_connect4":
            "images/metrics_plot/connect4",
        "metrics_isolete":
            "images/metrics_plot/isolete",
        "metrics_letter":
            "images/metrics_plot/letter",
        "metrics_mnist":
            "images/metrics_plot/mnist",
        "metrics_mnist_fashion":
            "images/metrics_plot/mnist_fashion",
        "metrics_musk2":
            "images/metrics_plot/musk2",
        "metrics_optdigits":
            "images/metrics_plot/optdigits",
        "metrics_page_blocks":
            "images/metrics_plot/page_blocks",
        "metrics_segment":
            "images/metrics_plot/segment",
        "metrics_shuttle":
            "images/metrics_plot/shuttle",
        "metrics_spambase":
            "images/metrics_plot/spambase",
        "metrics_usps":
            "images/metrics_plot/usps",
        "metrics_yaleb":
            "images/metrics_plot/yaleb",
        "metrics_iris":
            "images/metrics_plot/iris",
        "metrics_forest":
            "images/metrics_plot/forest",
        "metrics_satimages":
            "images/metrics_plot/satimages"
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


class DatasetFilesPaths(_Const):
    dirs_dataset_paths = {
        # Confusion matrices
        "dataset_path_connect4":
            "connect4/connect4.npy",
        "dataset_path_isolete":
            "isolete/isolete.npy",
        "dataset_path_letter":
            "letter/letter.npy",
        "dataset_path_mnist":
            "mnist/mnist.npy",
        "dataset_path_mnist_fashion":
            "mnist_fashion/mnist_fashion.npy",
        "dataset_path_musk2":
            "musk2/musk2.npy",
        "dataset_path_optdigits":
            "optdigits/optdigits.npy",
        "dataset_path_page_blocks":
            "page_blocks/page_blocks.npy",
        "dataset_path_segment":
            "segment/segment.npy",
        "dataset_path_shuttle":
            "shuttle/shuttle.npy",
        "dataset_path_spambase":
            "spambase/spambase.npy",
        "dataset_path_usps":
            "usps/usps.npy",
        "dataset_path_yaleb":
            "yaleb/yaleb.npy",
        "dataset_path_iris":
            "iris/dataset_iris.npy",
        "dataset_path_forest":
            "forest/dataset_forest.npy",
        "dataset_path_satimages":
            "satimages/dataset_satimages.npy",
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
DATASET_FCNN_PATHS: FCNNPaths = FCNNPaths()
DATASET_MPDRNN_PATHS: MPDRNNPaths = MPDRNNPaths()
DATASET_FILES_PATHS: DatasetFilesPaths = DatasetFilesPaths()
