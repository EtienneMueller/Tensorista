import numpy as np
from tensorslow.keras.engine.base_layer import Layer
from tensorslow.keras.backend import sigmoid, d_sigmoid  # , relu, d_relu


class Dense(Layer):
    def __init__(self, num_of_neurons, activation=None, input_shape=(64,), name=None):
        # input_shape should be None
        super(Layer, self)
        self.num_of_neurons = num_of_neurons
        self.activation = activation
        self.input_shape = input_shape
        self.name = name
        if "build" in dir(Dense):
            self.build()

    def build(self):
        self.W = np.random.randn(self.input_shape[0], self.num_of_neurons)  # / np.sqrt(2) # (784, 64)
        self.b = np.zeros((self.num_of_neurons, ))

    def call(self, x):
        Z = np.dot(x, self.W) + self.b
        A = sigmoid(Z)
        return Z, A

    def backprop(self, Aprev, ZZ, Y, batchsize, lr, dA):
        dZ = dA * d_sigmoid(ZZ)
        dW = (1/batchsize) * np.dot(Aprev.T, dZ)
        db = (1/batchsize) * np.sum(dZ, axis=0, keepdims=True)
        self.W = self.W - lr * dW
        self.b = self.b - lr * db
        dA = np.dot(dZ, self.W.T)
        return dA


class ReLU(Layer):
    def __init__(self, max_value=None, negative_slope=0, threshold=0, **kwargs):
        super(ReLU, self)

    def call(self, x):
        return np.maximum(0, x)

    def backprop(x):
        return np.greater(x, 0)
