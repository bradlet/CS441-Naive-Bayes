# Bradley Thompson
# CS545 - Anthony Rhodes
# Programming Assignment 1

import numpy as np
import matplotlib.pyplot as pl
import os

ETA = .1
BIAS = 1
MOMENTUM = 0.9
LEARNING_RATE = 0.1
INPUTS = 785  # Bias node included
HIDDEN_N = 100  # Bias node not included
OUTPUTS = 10


class PcnNetwork:
    def __init__(self):
        self.input_weights = np.random.rand(HIDDEN_N, INPUTS) * 0.1 - 0.05
        self.hidden_weights = np.random.rand(HIDDEN_N + 1, OUTPUTS) * 0.1 - 0.05

    # One datapoint "inputs" dotted w/ all 10 perceptrons' weight arrays
    #   Returns: Array of size == # perceptrons w/ sum {each input(1) val * each weight(n) val}
    @staticmethod
    def inner_activations(data_vector, weights):
        return np.dot(data_vector, np.transpose(weights))

    # Include sigmoid squashing
    def activations(self, data_vector, weights):
        return sigmoid(self.inner_activations(data_vector, weights))

    # Used to get error between hidden and output layers
    @staticmethod
    def output_error(target, outputs):
        target_vector = np.full(10, LEARNING_RATE)
        target_vector[target] = MOMENTUM
        return outputs * (1 - outputs) * (target_vector - outputs)

    # Used to get error between input and hidden layers
    def hidden_error(self, hidden, error):
        return hidden * (1 - hidden) * np.dot(self.hidden_weights, error)

    # Used to calculate weight update delta
    #   arg names a little misleading because, to get the right result shape, I had to flip args for input-hidden delta
    @staticmethod
    def update_weights(error, activations):
        return ETA * np.outer(activations, error)

    def forward_pass(self, data_vector):
        hidden_activations = self.activations(data_vector, self.input_weights)
        hidden_activations = np.append(BIAS, hidden_activations)  # add bias to that layer
        output_activations = self.activations(hidden_activations, np.transpose(self.hidden_weights))

        return hidden_activations, output_activations

    # One full computation cycle on one 785 member datapoint from the set.
    # Carrying out Back-Propagation w/ SGD
    #   inputs: tuple -> (classification, vector of length 785)
    def compute(self, inputs):
        hidden_activations, output_activations = self.forward_pass(inputs[1])

        hidden_to_output_error = self.output_error(inputs[0], output_activations)
        input_to_hidden_error = self.hidden_error(hidden_activations, hidden_to_output_error)

        hidden_delta = self.update_weights(hidden_to_output_error, hidden_activations)
        input_delta = self.update_weights(inputs[1], input_to_hidden_error[1:])

        self.hidden_weights = self.hidden_weights + hidden_delta
        self.input_weights = self.input_weights + input_delta

    # Runs the back prop algorithm on each datapoint in the set.
    #   Note: Term "inputs" to denote tuple -> (class, data vector)
    def epoch(self, dataset):
        for inputs in dataset:
            self.compute(inputs)

    # Same return as compute, but it doesn't back propagate
    def guess_class(self, inputs):
        output_activations = self.forward_pass(inputs[1])[1]
        return np.where(output_activations == max(output_activations))[0][0]

    # Calculates how accurate the MLP is at guess class for this set
    def accuracy(self, dataset):
        correct_classifications = 0
        for inputs in dataset:
            if self.guess_class(inputs) == inputs[0]:
                correct_classifications = correct_classifications + 1
        return correct_classifications / len(dataset)


# Turn each datapoint in data set into a tuple, so:
#   data[0] grabs the first datapoint
#   datapoint[0] grabs that datapoint's class
#   datapoint[1] grabs that datapoint's set of inputs
def preprocess(dataset):
    processed = list()
    for datapoint in dataset:
        datapoint = (datapoint[0], datapoint / 255)
        datapoint[1][0] = BIAS  # Of the data in this tuple, make the first entry = BIAS
        processed.append(datapoint)
    return processed


def load_data(name):
    return np.loadtxt(fname=name, delimiter=",", dtype=np.dtype(np.uint8))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    os.chdir('./MNIST_CSV')
    training_data = preprocess(load_data('mnist_train.csv'))
    test_data = preprocess(load_data('mnist_test.csv'))
    pcn = PcnNetwork()
    pcn2 = PcnNetwork()  # Note: Turn-in state is as the program needs to be for experiment 3

    num_epochs = 50

    # Setup lists to collect accuracy information and preload w/ untrained accuracy
    train_accuracy = list()
    train2_accuracy = list()
    test_accuracy = list()
    test2_accuracy = list()

    first_quarter_index = len(training_data) // 4
    last_half_index = len(training_data) // 2

    for i in range(0, num_epochs):
        pcn.epoch(training_data[:first_quarter_index])  # Use the first quarter of the training data
        pcn2.epoch(training_data[last_half_index:])  # Use the last half of the training data
        train_accuracy.append(pcn.accuracy(training_data[:first_quarter_index]))
        train2_accuracy.append(pcn2.accuracy(training_data[last_half_index:]))
        test_accuracy.append(pcn.accuracy(test_data))
        test2_accuracy.append(pcn2.accuracy(test_data))

    print(train_accuracy)
    print(train2_accuracy)
    print(test_accuracy)
    print(test2_accuracy)

    pl.plot(np.array(train_accuracy), label='train 1/4')
    pl.plot(np.array(train2_accuracy), label='train 1/2')
    pl.plot(np.array(test_accuracy), linestyle='dashed', label='test 1/4')
    pl.plot(np.array(test2_accuracy), linestyle='dashed', label='test 1/2')
    pl.legend()
    pl.show()
