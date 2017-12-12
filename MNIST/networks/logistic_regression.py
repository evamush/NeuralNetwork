from __future__ import print_function

import tensorflow as tf 
import numpy as np 

# Import training data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Training parameter
learning_rate = 0.01
num_training = 10
batch_size = 100

# tf graph input
x = tf.placeholder(dtype = tf.float32, shape = ([None, 784]))
y = tf.placeholder(dtype = tf.float32, shape = ([None, 10]))

# Weight
W = tf.Variable(tf.zeros([784, 10]))

# Bias
b = tf.Variable(tf.zeros([10]))
# Model
y_pred = tf.nn.softmax(tf.add(tf.matmul(x,W),b))

# Cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices = [1]))

# Using gradient descent to find the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Start training
init = tf.global_variables_initializer()

with tf.Session() as sess:

	sess.run(init)

	for i in range(num_training):

		num_batch = int(mnist.train.num_examples / batch_size)

		avg_cost = 0

		for j in range(num_batch):

			batch_x, batch_y = mnist.train.next_batch(batch_size) 

			opti, c = sess.run([optimizer,cost], feed_dict={x: batch_x, y: batch_y})

			avg_cost += c / num_batch

		print("Training %d, cost: %f"%(i + 1,avg_cost))


	# Testing
	# tf.equal return a bool type
	correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))

	# Accuracy
	# T determine what fraction are correct, we cast to floating point numbers
	# and then take the mean. 
	# For example, [True, False, True, True] would become
	# [1,0,1,1] which would become 0.75
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print("Accuracy:", sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels}))






