import numpy as np


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
  return sigmoid(x) * (1 - sigmoid(x))

# data
x = np.array([0.1, 0.3])
y = 0.2
weights = np.array([-0.8, 0.5])

learnrate = 0.5

h = np.dot(x, weights)

y_hat = sigmoid(h)

error = y - y_hat

output_grad = sigmoid_prime(h)

error_term = error * output_grad

# Gradient Descent Step
del_w = [learnrate * error_term * x[0],
          learnrate * error_term * x[1]]

# Or del_w = learnrate * error_term * x
