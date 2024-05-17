#!/usr/bin/env python3

"""An implementation of a simple neural network
"""

import random
import matplotlib.pyplot as plt
from data_loader import MnistDataloader


def main():
    mnist_dataloader = MnistDataloader()
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    
    plt.imshow(x_train[1], cmap=plt.cm.gray)
    plt.show()

if __name__ == '__main__':
    main()