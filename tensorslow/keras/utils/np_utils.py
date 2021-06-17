import numpy as np


def to_categorical(y, num_classes=None, dtype='float32'):
    """Convert vector to one-hot encoded matrix"""
    if num_classes is None:
        num_classes = np.max(y) + 1
    y_hat = np.zeros((y.size, num_classes))
    y_hat[np.arange(y.size), y] = 1
    return y_hat
