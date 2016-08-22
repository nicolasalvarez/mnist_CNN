"""
MNIST classifier using CNN.
From TF tutorial:
https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_variable(shape):
    """Return weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Return slightly positive initial bias to avoid "dead neurons"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """Convolutions uses a stride of one and are zero padded so that the output is the same size as the input."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """Pooling is plain old max pooling over 2x2 blocks."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# download and read mnist dataset. The dataset is split into three parts: 55,000 data points of training data
# (mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of validation data (mnist.validation).
# The training images are mnist.train.images and the training labels are mnist.train.labels.
# Each image is 28x28 = 784 pixels.
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# First layer, it consist of convolution, followed by max pooling. The convolutional computes 32 features for each
# 5x5 patch. Its weight tensor has a shape of [5, 5, 1, 32]. The first two dimensions are the patch size,
# the next is the number of input channels, and the last is the number of output channels.
# There is also a bias vector with a component for each output channel.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding to image
# width and height, and the final dimension corresponding to the number of color channels.
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second layer with 64 features for each 5x5 patch.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing
# on the entire image. We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight
# matrix, add a bias, and apply a ReLU.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# To reduce overfitting, we apply dropout before the readout layer. We create a placeholder for the probability that
# a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during
# testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them,
# so dropout just works without any additional scaling.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Finally, we add a softmax layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train and evaluate model.
# ADAM optimizer.
# Additional parameter keep_prob in feed_dict to control the dropout rate.
# Add logging to every 100th iteration in the training process.

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

corr_pred = 0.0
test_size = 10000
BATCH_SIZE = 100

for i in xrange(int(test_size / BATCH_SIZE)):
    batch = mnist.test.next_batch(BATCH_SIZE)
    corr_pred += tf.reduce_sum(tf.cast(correct_prediction.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}),
                                       tf.float32))

test_accuracy = corr_pred.eval()/test_size
print("test accuracy %g" % test_accuracy)

sess.close()