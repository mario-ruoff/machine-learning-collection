#!/usr/bin/env python3

"""An implementation of a simple neural network
"""

import numpy as np
import matplotlib.pyplot as plt
from data_loader import MnistDataloader


def main():

    # Load data
    mnist_dataloader = MnistDataloader()
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    x_train = np.array(x_train) # 60000 x 28 x 28
    y_train = np.array(y_train) # 60000
    x_test = np.array(x_test)   # 10000 x 28 x 28
    y_test = np.array(y_test)   # 10000

    # Show example image
    # plt.imshow(x_train[0], cmap=plt.cm.gray)
    # plt.show()

    # Convert tensor to lower dimension and normalize
    x_train = x_train.reshape(x_train.shape[0], -1) # 60000 x 784
    x_train = x_train / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1)    # 10000 x 784
    x_test = x_test / 255.0

    # Construct network
    hidden_neurons = 128
    output_neurons = 10
    weights_h = np.random.randn(x_train.shape[1], hidden_neurons) * 0.01  # 784 x 128
    biases_h = np.zeros(hidden_neurons)                                   # 128
    weights_o = np.random.randn(hidden_neurons, output_neurons) * 0.01    # 128 x 10
    biases_o = np.zeros(output_neurons)                                   # 10

    # Run network
    hidden_output = relu(np.dot(x_train[0], weights_h) + biases_h)
    output = softmax(np.dot(hidden_output, weights_o) + biases_o)

    print("End")


# ReLU activation function
def relu(x):
    return np.maximum(0, x)


# Softmax activation function
def softmax(vector):
    exp_vector = np.exp(vector)
    return exp_vector / np.sum(exp_vector)

    
if __name__ == '__main__':
    main()
