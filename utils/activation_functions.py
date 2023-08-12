import numpy as np

from elm.src.config.config import MPDRNNConfig

cfg = MPDRNNConfig().parse()


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- I D E N T I T Y --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def identity(x: np.ndarray) -> np.ndarray:
    """
    Linear activation function.
    param x: Input array.
    :return: A numpy array representing the input array, transformed by the activation function.
    """

    return x


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- S I G M O I D ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Applies the sigmoid activation function.
    param x: Input array
    :return: A numpy array representing the input array, transformed by the activation function.
    """

    return 1 / (1 + np.exp(-x))


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- R E L U ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def ReLU(x: np.ndarray) -> np.ndarray:
    """
    Applies the Rectified Linear Unit activation function.
    param x: Input array.
    :return: A numpy array representing the input array, transformed by the activation function.
    """

    return np.maximum(0, x)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ L E A K Y   R E L U -------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def leaky_ReLU(x: np.ndarray, alpha: float = cfg.slope) -> np.ndarray:
    """
    Leaky Rectified Linear Unit, or Leaky ReLU, is a type of activation function based on a ReLU, but it has a small
    slope for negative values instead of a flat slope.
    param x: Input array.
    param alpha: Value of the slope.
    :return: A numpy array representing the input array, transformed by the activation function.
    """

    return np.maximum(alpha * x, x)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- T A N S I G ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def tanh(x: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent activation function.
    param x: Input array.
    :return: A numpy array representing the input array, transformed by the activation function.
    """

    return (np.exp(2*x)-1)/(np.exp(2*x)+1)
