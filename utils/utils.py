import colorlog
import gc
import logging
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import os
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
def plot_confusion_matrix(metrics: dict, path_to_plot, labels, name_of_dataset: str, operation: str, method: str,
                          phase_name: str) -> None:
    cm = metrics.get("confusion_matrix")
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
    plt.title('%s, %s, %s, %s' % (phase_name, name_of_dataset, method, operation))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    filename = os.path.join(path_to_plot, f"{phase_name}_{name_of_dataset}_{method}_{operation}.png")
    plt.savefig(filename, dpi=300)
    plt.close()

    plt.close("all")
    plt.close()
    gc.collect()


def display_dataset_info(gen_ds_cfg):
    df = pd.DataFrame.from_dict(gen_ds_cfg, orient='index')
    df = df.drop("cached_dataset_file", axis=0)
    df = df.drop("path_to_cm", axis=0)
    logging.info(df)


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------- P L O T   T R A I N   A N D   V A L I D   D A T A ----------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_metrics(path_to_plot: str, train_accuracy: list, test_accuracy: list, name_of_dataset: str, operation: str)\
        -> None:
    # Create background
    plt.style.use("dark_background")
    for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
        plt.rcParams[param] = '0.9'  # very light grey
    for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
        plt.rcParams[param] = '#212946'  # bluish dark grey 212946

    color_mapping = {
        "accuracy": [
            '#FF62B2',  # pink
            '#F5D300',  # yellow
        ]
    }

    colors = color_mapping.get(operation, None)

    if colors is None:
        raise ValueError('An unknown operation \'%s\'.' % operation)

    # Transform fcnn_data to plot
    df = pd.DataFrame({'Train': train_accuracy,
                       'Test': test_accuracy})
    df['Train'] = df['Train'].astype(float)
    df['Test'] = df['Test'].astype(float)
    fig, ax = plt.subplots()

    # Redraw the fcnn_data with low alpha and slightly increased line width:
    n_shades = 9
    diff_line_width = 0.5
    alpha_value = 0.9 / n_shades

    for n in range(1, n_shades + 1):
        df.plot(marker='o',
                linewidth=2 + (diff_line_width * n),
                alpha=alpha_value,
                legend=False,
                ax=ax,
                color=colors)

    # Color the areas below the lines:
    for column, color in zip(df, colors):
        ax.fill_between(x=df.index,
                        y1=df[column].values,
                        y2=[0] * len(df),
                        color=color,
                        alpha=0.1)

    ax.grid(color='#2A3459')
    ax.set_xlim([ax.get_xlim()[0] - 0.2, ax.get_xlim()[1] + 0.2])  # to not have the markers cut off
    ax.set_ylim(np.min(test_accuracy) - 0.025, np.max(train_accuracy) + 0.025)

    ax.set_ylabel("Score")
    ax.set_xlabel("Iterations")
    ax.set_title('Train and Test %s on %s' % (operation, name_of_dataset))

    train_legend = mp.Patch(color='#FE53BB', label='Train')
    valid_legend = mp.Patch(color='yellow', label='Test')

    ax.legend(handles=[train_legend, valid_legend])

    filename = os.path.join(path_to_plot, f"{name_of_dataset}_{operation}.png")
    plt.savefig(filename, dpi=300)
    plt.close()

    plt.close("all")
    plt.close()
    gc.collect()
