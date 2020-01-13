import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Network size
n_input = 4
n_hidden = 3
n_output = 2

np.random.seed(42)
# Fake data
X = np.random.randn(4)

weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(n_input, n_hidden))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(n_hidden, n_output))

# Forward pass through the network
hidden_layer_in = np.dot(X, weights_input_to_hidden)
hidden_layer_out = sigmoid(hidden_layer_in)

print('Hidden-layer Output: ')
print(hidden_layer_out)

output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
output_layer_out = sigmoid(output_layer_in)

print('Output-layer Output: ')
print(output_layer_out)