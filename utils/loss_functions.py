import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- M S E -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes the mean of squares of errors between labels and predictions.
    param y_true: Ground truth values.
    param y_pred: Predicted values.
    :return: Weighted loss float.
    """

    return np.mean(np.power(np.asarray(y_true) - np.asarray(y_pred), 2))


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- M A E -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes the mean of absolute difference between labels and predictions.
    param y_true: Ground truth values.
    param y_pred: The predicted values.
    :return: Weighted loss float.
    """

    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))
