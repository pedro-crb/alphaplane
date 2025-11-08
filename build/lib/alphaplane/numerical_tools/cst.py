from __future__ import annotations

import numpy as np
from scipy.special import comb
from alphaplane.numerical_tools.array_operations import wide, tall


def cst(x: np.ndarray, cst_parameters: np.ndarray, n1: float, n2: float) -> np.ndarray:
    num_parameters = len(cst_parameters)
    A = cst_matrix(x, num_parameters, n1, n2)
    y_curve = np.dot(A, cst_parameters)
    return y_curve


def cst_matrix(x: np.ndarray, num_parameters: int, n1: float, n2: float) -> np.ndarray:
    class_function = x ** n1 * (1 - x) ** n2
    order = num_parameters - 1
    coeffs = comb(order, np.arange(order + 1))
    dims = (num_parameters, len(x))
    S_matrix = (tall(coeffs, dims) * wide(x, dims) ** tall(np.arange(order + 1), dims) *
                wide(1 - x, dims) ** tall(order - np.arange(order + 1), dims))
    return class_function[:, None] * S_matrix.T


def cst_fit_line(x: np.ndarray, y: np.ndarray,
                 num_parameters: int,
                 n1: float, n2: float) -> np.ndarray:
    A = cst_matrix(x, num_parameters, n1, n2)
    cst_parameters, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return cst_parameters
