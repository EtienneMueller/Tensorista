import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    return np.greater(x, 0)
