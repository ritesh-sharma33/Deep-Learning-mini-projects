import tensorflow as tf

# Output depth
k_output = 24

# Image Dimensions
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter dimensions
filter_size_width = 5
filter_size_height = 5

# Input/image
input1 = tf.placeholder(
  tf.float32,
  shape=[None, image_height, image_width, color_channels]
)

# Weights and bias
weight = tf.Variable(tf.truncated_normal(
  [filter_size_height, filter_size_width, color_channels, k_output]
))

bias = tf.Variable(tf.zeroes(k_output))

# Apply convolution
conv_layer = tf.nn.conv2d(input1, weight, strides=[1, 2, 2, 1], padding='SAME')
# add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# apply activation function
conv_layer = tf.nn.relu(conv_layer)

# Max pooling layer
conv_layer = tf.nn.max_pool(
  conv_layer,
  ksize=[1, 2, 2, 1],
  strides=[1, 2, 2, 1],
  padding='SAME'
)