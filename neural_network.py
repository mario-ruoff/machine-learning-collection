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
    weights = np.random.rand(x_train.shape[1], hidden_neurons)  # 784 x 128
    biases = np.random.rand(hidden_neurons)                     # 128

    # Run network
    output = relu(np.dot(x_train[0], weights) + biases)

    print("End")

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

    
if __name__ == '__main__':
    main()