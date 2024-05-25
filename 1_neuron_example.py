import numpy as np

def loss(output, label):
    return (output - label) ** 2

def d_loss_output(output, label):
    return 2 * (output - label)

def relu(z):
    return max(0, z)

def d_relu_z(z):
    return 1 if z > 0 else 0

def compute_z(data, weight, bias):
    return weight * data + bias

def d_z_weight(data, weight, bias):
    return data

def d_z_bias(data, weight, bias):
    return 1

learning_rate = 0.01
label = 1
weight = 0.2
bias = 0.1
input = 0.6

# Forward
z = compute_z(input, weight, bias)
output = relu(z)
error = loss(output, label)

# Backward
d_loss = d_loss_output(output, label)
d_output = d_relu_z(z)
d_z_w = d_z_weight(input, weight, bias)
d_z_b = d_z_bias(input, weight, bias)

d_loss_weight = d_loss * d_output * d_z_w
d_loss_bias = d_loss * d_output * d_z_b
weight -= learning_rate * d_loss_weight
bias -= learning_rate * d_loss_bias

print("End")
