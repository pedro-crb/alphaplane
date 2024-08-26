from __future__ import annotations

import numpy as np


def cartesian_product(*arrays):
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la])
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr


def search_sorted_with_nan(array, values):
    array = np.atleast_1d(array)
    values = np.atleast_1d(values)
    indices = np.searchsorted(array, values)
    indices = np.minimum(indices, len(array) - 1)
    found_values = array[indices]
    result = [(ind if found == val else np.nan) for ind, found, val in zip(indices, found_values, values)]
    return result


def ndgrad(array: np.ndarray, var: np.ndarray | None, axis: int = 0) -> np.ndarray:
    var = np.array(range(array.shape[axis])) if var is None else var
    var = var.squeeze()
    assert len(var.shape) == 1 and len(var) == array.shape[axis], "variable and array shapes do not match"
    var = np.array(var, ndmin=array.ndim).swapaxes(0, -1)
    arr = np.array(array, copy=True).swapaxes(axis, 0)
    forward = np.full_like(arr, np.nan)
    backward = np.full_like(arr, np.nan)
    central = np.full_like(arr, np.nan)
    forward[:-1] = (arr[1:] - arr[:-1])/(var[1:] - var[:-1])
    backward[1:] = forward[:-1]
    central[1:-1] = (arr[2:] - arr[:-2])/(var[2:] - var[:-2])
    derivative = np.where(
        ~np.isnan(central), central, np.where(
            ~np.isnan(forward), forward, backward
        )
    )
    grad = np.where(~np.isnan(arr), derivative, np.nan)
    grad = np.swapaxes(grad, axis, 0)
    return grad


def monotonic_indices(x):
    last_value = -np.inf
    valid = np.zeros_like(x, dtype=bool)
    for i, val in enumerate(x):
        if val > last_value:
            valid[i] = True
            last_value = val
    return valid


def wide(vector, dims):
    return np.tile(np.reshape(vector, (1, dims[1])), (dims[0], 1))


def tall(vector, dims):
    return np.tile(np.reshape(vector, (dims[0], 1)), (1, dims[1]))
