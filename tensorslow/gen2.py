import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import import_mnist
import os
from time import clock

# TO-DO
# - Trying number of nodes in own function
# - Epochs / Mini Batch Gradient Descent
# - Different activation function for output layer
# - dropout
# - Normalizing
# - Perceptron.py???
# - Adam


def session(param, hyper, auto_learning_rate=False):
    param = initialize(param, hyper)
    cost_old1 = 11  # number of output nodes+1

    if auto_learning_rate:
        alpha = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
        learning_rate = 1
        hyper['alpha'] = alpha[learning_rate]

    t1 = clock()
    for i in range(hyper['num_of_it']):

        param = fwd_prop(param, hyper, dataset='trn')
        param = backprop(param, hyper)

        param = fwd_prop(param, hyper, dataset='tst')
        print('%-8s' % (str(i + 1) + '/' + str(hyper['num_of_it'])), end=' ')

        [trn_acc, dev_acc] = calc_accuracy(param, hyper)

        sys.stdout.write('Dev: '
                         + str(round(100 * dev_acc, 2))
                         + '%\tTrn: ' 
                         + str(round(100 * trn_acc, 2))
                         + '%\t')

        # print('\n%-12f%-12f' % (trn_acc, dev_acc))

        mse = mean_squared_error(param, hyper)

        sys.stdout.write('MSE: ' + str(round(100 * mse, 2)) + '%\t')

        cost = cost_function(param, hyper)
        diff = cost_old1 - cost
        sys.stdout.write('Cost ' + str(round(float(cost), 5)) + '\t')
        sys.stdout.write('Diff ' + str(round(float(diff), 5)) + '\n')
        if auto_learning_rate:
            if diff < -2e-3:
                print('Overshooting')
                learning_rate += 1
                hyper['alpha'] = alpha[learning_rate]
                print('Learning rate is now ' + str(hyper['alpha']))
        # if 0 < diff < 2e-3:
        #     print('Stop Here')
        cost_old1 = cost

    t2 = clock()
    sys.stdout.write('Time:\t' + str(round(t2 - t1, 10)) + 's\n')

    return param


def initialize(param, hyper):
    # Always generate the same random numbers
    np.random.seed(1)
    # Randomly assigning weight matrix w and bias vector b
    for i in range(1, hyper['layers'] + 1):
        param['w' + str(i)] = np.random.randn(hyper['n' + str(i)],
                                              hyper['n' + str(i-1)]) * 0.01
        param['b' + str(i)] = np.zeros((hyper['n' + str(i)], 1))
    # Initialization for preventing exploding/vanishing gradients:
    # W = np.random.randn(shape) * np.sqrt(1/n[l-1]) for tanh (Xavier Init)
    # W = np.random.randn(shape) * np.sqrt(2/n[l-1]) for ReLU
    # W = np.random.randn(shape) * np.sqrt(1/(n[l-1] + n[l])
    # (sqrt-part is another hyperparameter)
    return param


def auto_nodes(param, hyper):
    for j in range(1, 4):
        print(j)
        recurse(param, hyper, 1, j)
    return


def recurse(param, hyper, layer, max_layers,
            min_nodes=10, max_nodes=784, step=200):
    if layer == max_layers:
        print('Output ' + str(layer + 1))
        hyper['n' + str(layer + 1)] = param['trn_y'].shape[0]
        hyper['layers'] = layer + 1
        for i in range(min_nodes, max_nodes, step):
            print(' ' * layer + 'n' + str(layer) + '=' + str(i))
            hyper['n' + str(layer)] = i
            session(param, hyper, auto_learning_rate=False)
        return
    else:
        for i in range(min_nodes, max_nodes, step):
            print(' ' * layer + 'n' + str(layer) + '=' + str(i))
            hyper['n' + str(layer)] = i
            recurse(param, hyper, layer + 1, max_layers)
    return


def fwd_prop(param, hyper, dataset, dropout=0):
    for i in range(1, hyper['layers'] + 1):
        param['z' + str(i)] = (
            np.dot(param['w' + str(i)],
                   param[dataset + '_a' + str(i - 1)])
            + param['b' + str(i)])
        param[dataset + '_a' + str(i)] = Activation.sigmoid(param['z'+str(i)])

        if dropout == 1:
            param[dataset + '_a' + str(i)] = dropout_regularization(
                param[dataset + '_a' + str(i)])
            param[dataset + '_a' + str(i)] /= 0.99

    return param


def backprop(param, hyper):
    # Backprop Algorithm
    param['dz' + str(hyper['layers'])] = (param['trn_a' + str(hyper['layers'])]
                                          - param['trn_y'])
    param = d_cost_function(param, hyper, hyper['layers'], weight_decay=1)
    for i in range(hyper['layers'] - 1, 0, -1):
        param['dz' + str(i)] = np.multiply(
            np.dot(param['w' + str(i + 1)].T, param['dz' + str(i + 1)]),
            (Activation.d_sigmoid(param['z' + str(i)])))
        param = d_cost_function(param, hyper, i, weight_decay=1)

    # Updating weights and biases
    for j in range(1, hyper['layers'] + 1):
        param['w' + str(j)] -= hyper['alpha'] * param['dw' + str(j)]
        param['b' + str(j)] -= hyper['alpha'] * param['db' + str(j)]
    return param


def cost_function(param, hyper):
    # Cross Entropy Loss
    m = param['trn_a0'].shape[1]
    j = (- (1/m) * np.sum(param['trn_y']
                          * (np.log(param['trn_a' + str(hyper['layers'])]))
                          + (1 - param['trn_y'])
                          * (np.log(1-param['trn_a' + str(hyper['layers'])]))))
    return j


def d_cost_function(param, hyper, i, weight_decay=0):
    m = param['trn_a0'].shape[1]
    param['dw' + str(i)] = ((1 / m)
                            * np.dot(param['dz' + str(i)],
                                     param['trn_a' + str(i - 1)].T))
    param['db' + str(i)] = ((1 / m)
                            * np.sum(param['dz' + str(i)],
                                     axis=1,
                                     keepdims=True))
    # Regularization
    if weight_decay == 1:
        param['dw' + str(i)] += hyper['lambd']/m * param['w' + str(i)]
    return param


def dropout_regularization(a):
    keep_prob = 0.99
    d = np.random.rand(a.shape[0], a.shape[1])
    d = d < keep_prob
    a = np.multiply(a, d)
    return a


def mean_squared_error(param, hyper):
    m = param['trn_a0'].shape[1]
    mse = 1/m * np.sum(np.square(param['tst_y']
                                 - param['tst_a'
                                 + str(hyper['layers'])]))
    return mse


def calc_accuracy(param, hyper):
    # training accuracy
    y_trn_val = np.argmax(param['trn_y'], axis=0)
    a_trn_val = np.argmax(param['trn_a' + str(hyper['layers'])], axis=0)
    trn_acc = (np.sum(y_trn_val == a_trn_val)) / param['trn_a0'].shape[1]

    # dev accuracy
    y_dev_val = np.argmax(param['tst_y'], axis=0)
    a_dev_val = np.argmax(param['tst_a' + str(hyper['layers'])], axis=0)
    dev_acc = (np.sum(y_dev_val == a_dev_val)) / param['tst_a0'].shape[1]

    return trn_acc, dev_acc


class Activation(object):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def tanh(z):
        # tanh for nodes, sigmoid only for output nodes (binary classification)
        return np.tanh(z)

    def relu(z):
        return np.maximum(0, z)

    def leaky_relu(z):
        return np.maximum(0.01 * z, z)

    def d_sigmoid(z):
        return (Activation.sigmoid(z)) * (1 - Activation.sigmoid(z))

    def d_tanh(z):
        return 1 - (np.tanh(z) ** 2)

    def d_relu(z):
        if z < 0:
            return 0
        else:
            return 1

    def d_leaky_relu(z):
        if z < 0:
            return 0.01
        else:
            return 1


def print_example(image, round=2):
    square = np.sqrt(len(image))
    for j in range(0, image.shape[1]):
        for i in range(0, image.shape[0]):
            if image[i, j] <= 0:
                print(".", end=('\t'))
            else:
                print(np.round(float(image[i]), round), end=' ')
            if (i+1) % square == 0:
                print('')
    return


if __name__ == "__main__":
    print("go")
    # Parameters
    trn_q = 600
    tst_q = 100
    epochs = 200
    alpha = 0.15
    param = dict()
    param = import_mnist.mnist(param, trn_qty=trn_q, tst_qty=tst_q)

    # Normalizing layer 0
    a_orig = (param['trn_a0']+10)/275
    t_orig = (param['tst_a0']+10)/275

    soll_trn = param['trn_y'].T  # 600,10
    soll_tst = param['tst_y'].T

    # OLD STUFF FOR 2G NN:
    # import_mnist.create_img(param)

    m = param['trn_a0'].shape[1]
    print('shape trn_a0 = ', param['trn_a0'].shape)

    # Hyperparameters
    hyper = dict()
    hyper['num_of_it'] = 5
    # hyper['layers'] = 2
    hyper['n0'] = param['trn_a0'].shape[0]
    # hyper['n1'] = 300
    # hyper['n2'] = param['trn_y'].shape[0]
    hyper['alpha'] = 0.1                    # learning rate
    hyper['lambd'] = 100                    # regularization parameter

    # Start Session
    # param = nn.session(param, hyper, auto_learning_rate=True)

    auto_nodes(param, hyper)
