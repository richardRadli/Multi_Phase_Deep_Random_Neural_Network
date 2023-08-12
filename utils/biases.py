import numpy as np

from config.config import UtilsConfig

cfg = UtilsConfig().parse()


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- Z E R O   B I A S -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def zero_bias(hidden_nodes: int) -> np.ndarray:
    """
    Generates zero bias.
    param hidden_nodes: Number of hidden nodes.
    :return: Matrix with zeros.
    """

    return np.zeros(hidden_nodes)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- O N E S  B I A S ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def ones_bias(hidden_nodes: int):
    """
    Generates ones bias with constant number.
    param hidden_nodes: Number of hidden nodes.
    :return: Matrix with ones multiplied with a constant.
    """

    return np.ones(hidden_nodes, ) * cfg.constant


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ X A V I E R   B I A S -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def xavier_bias(hidden_nodes: int):
    """
    Generates Xavier bias.
    param hidden_nodes: Number of hidden nodes.
    :return: Matrix with Xavier initialization.
    """

    limit = np.sqrt(6 / float(hidden_nodes))
    weights = np.random.uniform(-limit, limit, size=hidden_nodes)
    return weights


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- U N I F O R M   B I A S ----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def uniform_bias(hidden_nodes: int, lower_limit=cfg.lower_limit, upper_limit=cfg.upper_limit):
    """
    Generates bias with uniform distribution.
    param hidden_nodes: Number of hidden nodes.
    :return: Matrix with uniformly distributed bias.
    """

    return np.random.uniform(low=lower_limit, high=upper_limit, size=(hidden_nodes,))
