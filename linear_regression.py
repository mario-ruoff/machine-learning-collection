#!/usr/bin/env python3

"""An implementation of the linear regression in machine learning
"""

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Define basic parameters
    train_test_split = 0.8
    scale = 30
    data = get_data()
    data_train = data[:int(train_test_split*len(data))]
    data_test = data[int(train_test_split*len(data)):]
    features_train, labels_train = zip(*data_train)
    features_test, labels_test = zip(*data_test)

    # Calculate the linear regression
    m, b = train_model(features_train, labels_train)
    print(get_error(features_train, labels_train, m, b))

    # Visualize the data
    plt.scatter(features_train, labels_train)
    line_x = [0, scale]
    line_y = [m * x + b for x in line_x]
    plt.plot(line_x, line_y, color='red')
    plt.xlabel('features')
    plt.ylabel('labels')
    plt.title('Data Plot')
    plt.xlim(0, scale)
    plt.ylim(0, scale)
    plt.show()


def train_model(features, labels):
    x_avg = sum(features) / len(features)
    y_avg = sum(labels) / len(labels)
    x_sum = sum([abs(x - x_avg) for x in features])
    y_sum = sum([abs(y - y_avg) for y in labels])
    m = y_sum / x_sum
    b = -m * x_avg + y_avg
    return m, b


def get_error(features, labels, m, b):
    return sum([abs(m * feature + b - label) for feature, label in zip(features, labels)])


def get_data():
    return [
        [2.4, 15.1],
        [3.1, 15.9],
        [0.5, 10.9],
        [4.6, 19.0],
        [1.4, 13.2],
        [4.0, 17.9],
        [1.2, 11.8],
        [2.1, 14.2],
        [4.0, 17.3],
    ]

if __name__ == '__main__':
    main()