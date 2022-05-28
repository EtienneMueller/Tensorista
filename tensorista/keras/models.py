from tensorista.keras.engine.training import Model
from tensorista.keras.utils.generic_utils import Progbar
from tensorista.keras.utils.np_utils import to_categorical
# from tensorslow.keras.backend import sigmoid, d_sigmoid  # , relu, d_relu
import numpy as np
# import time
import sys


class Functional(Model):
    def __init__(self):
        super(Functional, self)


class Sequential(Functional):
    def __init__(self, layers=None, name=None):
        super(Sequential, self)
        self.name = name
        self.layers = layers
        if "build" in dir(Sequential):
            self.build()

    def build(self):
        X = np.zeros((50000, 784))
        hidden_neurons = 64
        self.W1 = np.random.randn(X.shape[1], hidden_neurons)  # / np.sqrt(2) # (784, 64)
        self.b1 = np.zeros((hidden_neurons, ))  # (64,)
        self.W2 = np.random.randn(hidden_neurons, 10)  # / np.sqrt(4) # (64, 10)
        self.b2 = np.zeros((10, ))  # (10,)

        self.Z = list(range(len(self.layers)))
        self.A = list(range(len(self.layers)))
        self.dZ = list(range(len(self.layers)))
        self.dA = list(range(len(self.layers)))

    def summary(self):
        print('\nModel: "' + self.name + '"\n' + ("_" * 65))
        print("Layer (type)", " " * 15,
              "Output Shape", " " * 12,
              "Param #\n" + "=" * 65)
        print("(l1_name)" + "\n" + "_" * 65)
        print("(l2_name)" + "\n" + "=" * 65)
        print("Total params:")
        print("Trainable params:")
        print("Non-trainable params:")

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer,
        self.loss = loss,
        self.metrics = metrics

    def cross_entropy_loss(Y, A2):
        return np.absolute(np.mean(
            (Y * np.log(A2)) - ((1 - Y) * np.log(1 - A2)))
        )

    def softmax(x):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def compute_loss(Y, Y_hat):
        L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
        m = Y.shape[1]
        L = -(1./m) * L_sum
        return L

    def fit(self,
            x_train,
            y_train,
            epochs=None,
            batch_size=None,
            validation_data=None):

        x_val, y_val = validation_data
        y_train = to_categorical(y_train, num_classes=10)

        lr = 0.1

        batches = int(np.ceil(x_train.shape[0] / batch_size))
        print("Batches:", batches)
        for i in range(epochs):
            print("\nEpoch", str(i+1) + "/" + str(epochs))

            if sys.platform == "ios":
                progbar = Progbar(batches, width=10, stateful_metrics=None)
            else:
                progbar = Progbar(batches, width=30, stateful_metrics=None)
            for j in range(int(batches)):
                X = x_train[j*batch_size:(j+1)*batch_size, :]
                Y = y_train[j*batch_size:(j+1)*batch_size]

                # Forward
                Z = []
                A = []
                for i, e in enumerate(self.layers):
                    # print("forward", i, e)
                    if i == 0:
                        x = X
                    else:
                        x = A[i-1]
                    z, a = self.layers[i].call(x)
                    Z.append(z)
                    A.append(a)

                # Accuracy
                Y_new = np.argmax(A[len(A)-1], axis=1)
                compare = np.equal(to_categorical(Y_new, num_classes=10), Y)
                acc_test = np.sum(compare)/batch_size
                # print("ValidationNew", acc_test)
                # print(np.argmax(to_categorical(Y_new, num_classes=10)[0:10, :], axis=1))
                # print(np.argmax(Y[0:10,:], axis=1))

                # Backprop
                dA = [None] * (len(self.layers) + 1)
                dA[len(dA)-1] = (A[len(A)-1] - Y) / (A[len(A)-1] * (1 - A[len(A)-1]))
                for i, e in reversed(list(enumerate(self.layers))):
                    if i == 0:
                        x = X
                    else:
                        x = A[i-1]
                    dA[i] = e.backprop(x, Z[i], Y, batch_size, lr, dA[i+1])

                #dA[1] = self.layers[1].backprop(A[0], Z[1], Y, batch_size, lr, dA[2])
                #dA[0] = self.layers[0].backprop(X,    Z[0], Y, batch_size, lr, dA[1])
                # dA2 = (A2 - Y) / (A2 * (1 - A2))
                # dZ2 = dA2 * d_sigmoid(Z2)

                # dAA1 = np.dot(dZZ2, W2.T)
                # dZZ1 = dAA1 * d_sigmoid(ZZ1)

                # mse = 1/m * np.sum(np.square(param['tst_y'] - param['tst_a' + str(hyper['layers'])]))

                # for j in range(int(batches//batch_size)):
                # loss = (1/4) * (-np.dot(Y, np.log(A2).T) - np.dot(1 - Y, np.log(1 - A2).T))
                values = [('Acc', acc_test), ('pr', np.round(np.random.random(1), 3))]
                progbar.add(1, values=values)

    def evaluate(self, x_test, y_test, batch_size=128):
        print("Test Acc")
        ZZ1, AA1 = self.layers[0].call(x_test)
        ZZ2, AA2 = self.layers[1].call(AA1)
        Y_new = np.argmax(AA2, axis=1)
        print(Y_new.shape)
        print(y_test.shape)
        compare = np.equal(Y_new, y_test)
        print("ValidationNew", np.sum(compare)/y_test.shape[0])
        print(Y_new[0:10])
        print(y_test[0:10])
