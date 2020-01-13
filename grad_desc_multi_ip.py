import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(x))

def sigmoid_prime(x):
  return sigmoid(x) * (1 - sigmoid(x))

learnrate = 0.5
x = np.array([1, 2, 3, 4])
y = np.array(0.5)

w = np.array([0.5, -0.5, 0.3, 0.1])

h = np.dot(x, w)

y_hat = sigmoid(h)

error = y - y_hat

error_term = error * y_hat * (1 - y_hat)

del_w = learnrate * error_term * x

print('Neural net output: ')
print(y_hat)
print('Amount of error: ')
print(error)
print('Change in weights: ')
print(del_w)
