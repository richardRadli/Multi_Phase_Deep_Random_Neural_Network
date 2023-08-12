import colorlog
import gc
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

from datetime import datetime


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


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------- P R E T T Y   P R I N T   R E S U L T S --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def pretty_print_results(metric_results: dict, operation: str, name: str, training_time: float = None) -> None:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    loss = metric_results.get("loss")
    loss = np.round(loss, 4)
    acc = metric_results.get("accuracy")
    precision = metric_results.get("precision_recall_fscore")[0]
    recall = metric_results.get("precision_recall_fscore")[1]
    fscore = metric_results.get("precision_recall_fscore")[2]

    if operation == "train":
        df = pd.DataFrame([[loss, acc, precision, recall, fscore, training_time]],
                          index=pd.Index(['Score']),
                          columns=pd.MultiIndex.from_product([['%s loss' % operation, '%s acc' % operation,
                                                               '%s precision' % operation, '%s recall' % operation,
                                                               '%s fscore' % operation, 'training time']]))

    elif operation == "test":
        df = pd.DataFrame([[loss, acc, precision, recall, fscore]],
                          index=pd.Index(['Score']),
                          columns=pd.MultiIndex.from_product([['%s loss' % operation, '%s acc' % operation,
                                                               '%s precision' % operation, '%s recall' % operation,
                                                               '%s fscore' % operation]]))
    else:
        raise ValueError('An unknown operation \'%s\'.' % operation)

    upper_separator = "-" * 34  # Customize the separator as needed
    lower_separator = "-" * 83

    logging.info("\n%s %s %s %s \n%s\n%s", upper_separator, name, operation, upper_separator, df, lower_separator)


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        instance = args[0]  # Get the instance of the class
        method_name = func.__name__

        # Update the total execution time for the method
        if method_name not in instance.total_execution_time:
            instance.total_execution_time[method_name] = 0.0
        instance.total_execution_time[method_name] += elapsed_time

        return result
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


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- P L O T   C O N F   M T X ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_confusion_matrix(metrics: dict, path_to_plot, name_of_dataset: str, operation: str, method: str) -> None:
    cm = metrics.get("confusion_matrix")
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.title('%s dataset, %s values, %s method' % (name_of_dataset, operation, method))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    plt.savefig(path_to_plot, dpi=300)
    plt.close()

    plt.close("all")
    plt.close()
    gc.collect()


def display_dataset_info(gen_ds_cfg):
    # Create a DataFrame from the dataset_config dictionary
    df = pd.DataFrame.from_dict(gen_ds_cfg, orient='index')
    # Drop the "cached_dataset_file" row
    df = df.drop("cached_dataset_file", axis=0)
    # Display the DataFrame
    logging.info(df)
