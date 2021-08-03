import numpy as np
import gzip
import urllib.request
import os
# import scipy.misc as smp
# import math
# import time
# import tensorflow as tf

# TO-DO
# - CIFAR-10 and CIFAR-100 (see https://www.cs.toronto.edu/~kriz/cifar.html)

"""
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
32768/29515 [=================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
26427392/26421880 [==============================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
8192/5148 [===============================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
4423680/4422102 [==============================] - 0s 0us/step
Epoch 1/15"""


def load_data():
    path = 'tensorslow/keras/datasets/tmp/'
    # path = '~/files/'
    try:
        x_train = np.load(path+'train-images-idx3-ubyte.npy')
        y_train = np.load(path+'train-labels-idx1-ubyte.npy')
        x_test = np.load(path+'t10k-images-idx3-ubyte.npy')
        y_test = np.load(path+'t10k-labels-idx1-ubyte.npy')
    except FileNotFoundError:
        print("[INFO] MNIST as numpy array not found")
        try:
            os.makedirs(path)
        except Exception:  # FileExistsError:
            pass
        try:
            print("[INFO] downloading MNIST from http://yann.lecun.com/exdb/mnist")
            x_train_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
            urllib.request.urlretrieve(x_train_url, path+'train-images-idx3-ubyte.gz')
            y_train_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
            urllib.request.urlretrieve(y_train_url, path+'train-labels-idx1-ubyte.gz')
            x_test_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
            urllib.request.urlretrieve(x_test_url, path+'t10k-images-idx3-ubyte.gz')
            y_test_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
            urllib.request.urlretrieve(y_test_url, path+'t10k-labels-idx1-ubyte.gz')
        except Exception:
            print("[INFO] download failed.")

        print("[INFO] converting to numpy array and exporting")
        # Import training images
        with gzip.open(path+'train-images-idx3-ubyte.gz', 'rb') as f:
            images_content = f.read()
        x_train = np.zeros((60000, 28, 28), dtype=int)
        for image in range(60000):
            for column in range(28):
                for row in range(28):
                    x_train[image, column, row] = ord(images_content[(image*784)+(column*28)+row+16: (image*784)+(column*28)+row+17])
        np.save(path+'train-images-idx3-ubyte.npy', x_train)

        # Import training labels
        with gzip.open(path+'train-labels-idx1-ubyte.gz', 'rb') as f:
            labels_content = f.read()
        y_train = np.zeros((60000,), dtype=int)
        for label in range(60000):
            y_train[label] = ord(labels_content[label+8: label+9])
        np.save(path+'train-labels-idx1-ubyte.npy', y_train)

        # Import test images
        with gzip.open(path+'t10k-images-idx3-ubyte.gz', 'rb') as f:
            images_content = f.read()
        x_test = np.zeros((10000, 28, 28), dtype=int)
        for image in range(10000):
            for column in range(28):
                for row in range(28):
                    x_test[image, column, row] = ord(images_content[(image*784)+(column*28)+row+16: (image*784)+(column*28)+row+17])
        np.save(path+'t10k-images-idx3-ubyte.npy', x_test)

        # Import test labels
        with gzip.open(path+'t10k-labels-idx1-ubyte.gz', 'rb') as f:
            labels_content = f.read()
        y_test = np.zeros((10000,), dtype=int)
        for label in range(10000):
            y_test[label] = ord(labels_content[label+8: label+9])
        np.save(path+'t10k-labels-idx1-ubyte.npy', y_test)

    return (x_train, y_train), (x_test, y_test)
