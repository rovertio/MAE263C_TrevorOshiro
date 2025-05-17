import numpy as np
from numpy.typing import NDArray


def validate_np_array(
    arr: NDArray[np.double], arr_name: str, correct_shape: tuple[int, ...]
):
    err_msg1 = f"{arr_name} must be a numpy array, but {arr_name} has type {type(arr)}."
    err_msg2 = (
        f"{arr_name} must be a numpy array of shape {correct_shape}, but {arr_name} "
        f"has shape {arr.shape}."
    )
    assert isinstance(arr, np.ndarray), err_msg1
    assert arr.shape == correct_shape, err_msg2
