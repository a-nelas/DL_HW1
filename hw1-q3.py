#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import standard_exponential

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Sign function.
        y_hat = np.argmax(self.W.dot(x_i))
        if y_hat != y_i:
            # Perceptron update.
            #w += (y - y_hat) * x
            # Correct class
            self.W[y_i, :] += x_i
            # Wrong Class
            self.W[y_hat, :] -= x_i
        # Q3.1a


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Label scores according to the model (num_labels x 1).
        label_scores = self.W.dot(x_i)[:, None]
        # One-hot vector with the true label (num_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1
        # Softmax function.
        # This gives the label probabilities according to the model (num_labels x 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        # SGD update. W is num_labels x num_features.
        self.W += learning_rate * (y_one_hot - label_probabilities) * x_i[None, :]
        # Q3.1b


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, n_layers):
        # Initialize an MLP with a single hidden layer.

        network_configuration = [n_features, hidden_size, n_classes]

        mean = .1
        standard_deviation = .1

        W1 = .1 * np.random.normal(mean, standard_deviation, size = (network_configuration[1], network_configuration[0]))
        W2 = .1 * np.random.normal(mean, standard_deviation, size = (network_configuration[2], network_configuration[1]))

        b1 = np.zeros(network_configuration[1])
        b2 = np.zeros(network_configuration[2])
        
        self.weights = [W1, W2]
        self.biases = [b1, b2]

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        
        predicted_labels = []
        for x in X:
            output, _ = forward(x, self.weights, self.biases)
            y_hat = predict_label(output)
            predicted_labels.append(y_hat)
        predicted_labels = np.array(predicted_labels)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        for x_i, y_i in zip(X, y):
            output, hiddens = forward(x_i, self.weights, self.biases)
            grad_weights, grad_biases = backward(x_i, y_i, output, hiddens, self.weights, loss_function='cross_entropy')
            self.update_parameters(self.weights, self.biases, grad_weights, grad_biases, learning_rate)

    def update_parameters(self, weights, biases, grad_weights, grad_biases, learning_rate):
        num_layers = len(weights)
        for i in range(num_layers):
            self.weights[i] -= learning_rate*grad_weights[i]
            self.biases[i] -= learning_rate*grad_biases[i]

def relu(x):
    return x * (x > 0)

def relu_derivetive(x):
    return 1. * (x > 0)

def forward(x, weights, biases):
    num_layers = len(weights)
    g = relu
    hiddens = []
    for i in range(num_layers):
        h = x if i == 0 else hiddens[i-1]
        z = weights[i].dot(h) + biases[i]
        if i < num_layers-1:  # Assume the output layer has no activation.
            hiddens.append(g(z))
    output = z
    # For classification this is a vector of logits (label scores).
    # For regression this is a vector of predictions.
    return output, hiddens

def backward(x, y, output, hiddens, weights, loss_function='squared'):
    num_layers = len(weights)
    g = relu
    z = output
    if loss_function == 'squared':
        grad_z = z - y  # Grad of loss wrt last z.
    elif loss_function == 'cross_entropy':
        # softmax transformation.
        probs = compute_label_probabilities(output)
        grad_z = probs - y  # Grad of loss wrt last z.
    grad_weights = []
    grad_biases = []
    for i in range(num_layers-1, -1, -1):
        # Gradient of hidden parameters.
        h = x if i == 0 else hiddens[i-1]
        grad_weights.append(grad_z[:, None].dot(h[:, None].T))
        grad_biases.append(grad_z)

        # Gradient of hidden layer below.
        grad_h = weights[i].T.dot(grad_z)

        # Gradient of hidden layer below before activation.
        assert(g == relu)
        grad_z = grad_h * relu_derivetive(h)

    grad_weights.reverse()
    grad_biases.reverse()
    return grad_weights, grad_biases

def compute_label_probabilities(output):
    # softmax transformation.
    output -= output.max()
    probs = np.exp(output) / np.sum(np.exp(output))
    return probs

def predict_label(output):
    # The most probable label is also the label with the largest logit.
    # y_hat = np.zeros_like(output)
    # y_hat[np.argmax(output)] = 1
    return np.argmax(output)

def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
