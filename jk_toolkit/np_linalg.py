"""
Contains linear algebra utilities for numpy.

These are re-exported by jk_toolkit/__init__.py.
"""

import numpy as np
import numpy.typing as npt


def col(arr: npt.ArrayLike) -> np.ndarray:
    """
    Converts a 1D array to a column vector.
    """
    arr = np.asarray(arr)
    if arr.ndim > 1:
        raise ValueError("Input array must be 1D or scalar.")

    return np.reshape(arr, (-1, 1))


def row(arr: npt.ArrayLike) -> np.ndarray:
    """
    Converts a 1D array to a row vector.
    """
    arr = np.asarray(arr)
    if arr.ndim > 1:
        raise ValueError("Input array must be 1D or scalar.")
    return np.reshape(arr, (1, -1))


def is_col(arr: npt.ArrayLike) -> bool:
    """
    Checks if the given array is a column vector.
    """
    arr = np.asarray(arr)
    return arr.ndim == 2 and arr.shape[1] == 1


def is_row(arr: npt.ArrayLike) -> bool:
    """
    Checks if the given array is a row vector.
    """
    arr = np.asarray(arr)
    return arr.ndim == 2 and arr.shape[0] == 1
