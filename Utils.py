import numpy as np


def softmax(x, temp=1.0):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x / temp) / np.sum(np.exp(x / temp), axis=0)


def generate_symmetries(array, axes=(0, 1)):
    flipped_array = np.flip(array, axes[0])
    return [
        array,
        np.rot90(array, 1, axes),
        np.rot90(array, 2, axes),
        np.rot90(array, 3, axes),
        flipped_array,
        np.rot90(flipped_array, 1, axes),
        np.rot90(flipped_array, 2, axes),
        np.rot90(flipped_array, 3, axes)
    ]
