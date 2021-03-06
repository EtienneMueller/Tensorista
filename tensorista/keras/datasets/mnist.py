import numpy as np
import urllib3
import requests
import gzip

# MNIST
# -----
# From: http://yann.lecun.com/exdb/mnist/
# References:
# [LeCun et al., 1998a]
# Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner.
# "Gradient-based learning applied to document recognition."
# Proceedings of the IEEE, 86(11):2278-2324,
# November 1998.

# CIFAR-10 and CIFAR-100
# To Do.
# https://www.cs.toronto.edu/~kriz/cifar.html


def load_data():
    #url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    #destpath = 'tensorista/keras/datasets/tmp/'
    #with open(destpath,'wb') as file:
    #    file.write(requests.get(url).content)

    #urllib.urlretrieve(url, "train.gz")

    path = 'tensorista/keras/datasets/tmp/'
    try:
        x_train = np.load(path+'train-images-idx3-ubyte.npy')
        y_train = np.load(path+'train-labels-idx1-ubyte.npy')
        x_test = np.load(path+'t10k-images-idx3-ubyte.npy')
        y_test = np.load(path+'t10k-labels-idx1-ubyte.npy')
    except FileNotFoundError:
        print("[INFO] MNIST as numpy array not found")
        http = urllib3.PoolManager()

        # Downloading and converting MNIST training images
        print("[INFO] Downloading MNIST training images")
        url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        filename = 'train-images-idx3-ubyte.gz'
        resp = http.request('GET', url)
        f = open(path+filename, 'wb')
        f.write(resp.data)
        f.close()
        resp.release_conn()

        print("[INFO] Converting MNIST training images to numpy arrays")
        with gzip.open(path+'train-images-idx3-ubyte.gz', 'rb') as f:
            images_content = f.read()
        x_train = np.zeros((60000, 28, 28), dtype=int)
        for image in range(60000):
            for column in range(28):
                for row in range(28):
                    x_train[image, column, row] = ord(
                        images_content[(image*784)+(column*28)+row+16:
                                       (image*784)+(column*28)+row+17])
        np.save(path+'train-images-idx3-ubyte.npy', x_train)

        # Downloading and converting MNIST training labeks
        print("[INFO] Downloading MNIST training labels")
        url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
        filename = 'train-labels-idx1-ubyte.gz'
        resp = http.request('GET', url)
        f = open(path+filename, 'wb')
        f.write(resp.data)
        f.close()
        resp.release_conn()

        print("[INFO] Converting MNIST training labels to numpy arrays")
        with gzip.open(path+'train-labels-idx1-ubyte.gz', 'rb') as f:
            labels_content = f.read()
        y_train = np.zeros((60000,), dtype=int)
        for label in range(60000):
            y_train[label] = ord(labels_content[label+8: label+9])
        np.save(path+'train-labels-idx1-ubyte.npy', y_train)

        # Downloading and converting MNIST test images
        print("[INFO] Downloading MNIST test images")
        url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
        filename = 't10k-images-idx3-ubyte.gz'
        resp = http.request('GET', url)
        f = open(path+filename, 'wb')
        f.write(resp.data)
        f.close()
        resp.release_conn()

        print("[INFO] Converting MNIST test images to numpy arrays")
        with gzip.open(path+'t10k-images-idx3-ubyte.gz', 'rb') as f:
            images_content = f.read()
        x_test = np.zeros((10000, 28, 28), dtype=int)
        for image in range(10000):
            for column in range(28):
                for row in range(28):
                    x_test[image, column, row] = ord(
                        images_content[(image*784)+(column*28)+row+16:
                                       (image*784)+(column*28)+row+17])
        np.save(path+'t10k-images-idx3-ubyte.npy', x_test)

        # Downloading and converting MNIST test labels
        print("[INFO] Downloading MNIST test labels")
        url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        filename = 't10k-labels-idx1-ubyte.gz'
        resp = http.request('GET', url)
        f = open(path+filename, 'wb')
        f.write(resp.data)
        f.close()
        resp.release_conn()

        print("[INFO] Converting MNIST test labels to numpy arrays")
        with gzip.open(path+'t10k-labels-idx1-ubyte.gz', 'rb') as f:
            labels_content = f.read()
        y_test = np.zeros((10000,), dtype=int)
        for label in range(10000):
            y_test[label] = ord(labels_content[label+8: label+9])
        np.save(path+'t10k-labels-idx1-ubyte.npy', y_test)

    return (x_train, y_train), (x_test, y_test)
