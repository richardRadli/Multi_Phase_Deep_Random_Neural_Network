import colorlog
import gc
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import torch

from datetime import datetime
from functools import wraps
from typing import Any, Callable


def setup_logger():
    """
    Set up a colorized logger with the following log levels and colors:

    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red on a white background

    Returns:
        The configured logger instance.
    """

    # Check if logger has already been set up
    logger = logging.getLogger()
    if logger.hasHandlers():
        return logger

    # Set up logging
    logger.setLevel(logging.INFO)

    # Create a colorized formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        })

    # Create a console handler and add the formatter to it
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- C R E A T E   T I M E S T A M P ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def create_timestamp() -> str:
    """
    Creates a timestamp in the format of '%Y-%m-%d_%H-%M-%S', representing the current date and time.

    :return: The timestamp string.
    """

    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def create_dir(root_dir: str, method: str):
    timestamp = create_timestamp()
    output_dir = os.path.join(root_dir, f"{timestamp}_{method}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------- P R E T T Y   P R I N T   R E S U L T S --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def pretty_print_results(acc, precision, recall, fscore, loss, root_dir: str, operation: str, name: str,
                         exe_time=None) -> None:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    columns = [
        f"{operation} loss",
        f"{operation} acc",
        f"{operation} precision",
        f"{operation} recall",
        f"{operation} fscore",
    ]

    data = [
        np.round(loss, 4),
        np.round(acc, 4),
        np.round(precision, 4),
        np.round(recall, 4),
        np.round(fscore, 4)
    ]

    if exe_time is not None:
        columns.append(f"{operation} exe time")
        data.append(np.round(exe_time, 4))

    df = pd.DataFrame([data],
                      index=pd.Index(['Score']),
                      columns=pd.MultiIndex.from_product([columns]))

    path_to_save = os.path.join(root_dir, f"{operation}.txt")
    try:
        with open(path_to_save, 'a'):
            pass
    except FileNotFoundError:
        df.to_csv(path_to_save, sep='\t', index=True)
    else:
        df.to_csv(path_to_save, sep='\t', index=True, mode='a', header=True)

    upper_separator = "-" * 34  # Customize the separator as needed
    lower_separator = "-" * 83

    logging.info("\n%s %s %s %s \n%s\n%s", upper_separator, name, operation, upper_separator, df, lower_separator)


def measure_execution_time(func: Callable) -> Callable:
    """
    Decorator to measure the execution time of a function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        """
        Wrapper function to measure execution time.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Any: The result of the function.
        """

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        wrapper.execution_time = end_time - start_time
        # logging.info(f"Execution time of {func.__name__}: {wrapper.execution_time:.4f} seconds")
        return result

    wrapper.execution_time = None
    return wrapper


def calc_exp_neurons(total_neurons: int, n_layers: int):
    # Calculate the geometric mean of the number of neurons based on the number of layers
    geom_mean = total_neurons ** (1 / n_layers)

    # Calculate the number of neurons for each layer using the geometric mean
    neurons_per_layer = [int(geom_mean ** i) for i in range(n_layers)]

    # Adjust the first layer to match the total number of neurons
    neurons_per_layer[0] += total_neurons - sum(neurons_per_layer)

    # Make sure the last layer has more than 1 neuron
    if neurons_per_layer[-1] <= 1:
        neurons_per_layer[-2] += neurons_per_layer[-1] - 1
        neurons_per_layer.pop()

    return sorted(neurons_per_layer, reverse=True)


def display_dataset_info(gen_ds_cfg):
    df = pd.DataFrame.from_dict(gen_ds_cfg, orient='index')
    df = df.drop("cached_dataset_file", axis=0)
    df = df.drop("path_to_cm", axis=0)
    logging.info(df)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- P L O T   C O N F   M T X ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_confusion_matrix(cm, path_to_plot, name_of_dataset: str, operation: str, method: str, labels=None) -> None:
    fig, axis = plt.subplots(1, 3, figsize=(15, 5))

    for i, cm in enumerate(cm):
        ax = axis[i]
        sns.heatmap(cm, annot=True, fmt='.0f', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title('%s, %s, %s, %s' % (f"Phase {i + 1}", name_of_dataset, method, operation))
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    filename = os.path.join(path_to_plot, f"{name_of_dataset}_{method}_{operation}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    gc.collect()


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------- P L O T   T R A I N   A N D   V A L I D   D A T A ----------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_metrics(train, test, metric_type: str, path_to_plot: str, name_of_dataset: str, method: str) \
        -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    x_values = np.arange(1, len(train) + 1)

    filename = os.path.join(path_to_plot, f"{name_of_dataset}_{method}_{metric_type}.png")

    plt.plot(x_values, train, marker='o', label='Train')
    plt.plot(x_values, test, marker='o', label='Test')
    plt.xlabel('Phases')
    plt.ylabel(metric_type)
    plt.title('%s, %s, %s' % (name_of_dataset, method, metric_type))
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    gc.collect()


def use_gpu_if_available() -> torch.device:
    """
    Provides information about the currently available GPUs and returns a torch device for training and inference.

    :return: A torch device for either "cuda" or "cpu".
    """

    if torch.cuda.is_available():
        cuda_info = {
            'CUDA Available': [torch.cuda.is_available()],
            'CUDA Device Count': [torch.cuda.device_count()],
            'Current CUDA Device': [torch.cuda.current_device()],
            'CUDA Device Name': [torch.cuda.get_device_name(0)]
        }

        df = pd.DataFrame(cuda_info)
        logging.info(df)
    else:
        logging.info("Only CPU is available!")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
