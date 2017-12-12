from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load the weights and biases

layer_1_weights = tf.Variable(np.loadtxt('hidden_layer_1_weights.txt', dtype='float32'))
layer_1_biases = tf.Variable(np.loadtxt('hidden_layer_1_biases.txt', dtype='float32'))
layer_2_weights = tf.Variable(np.loadtxt('hidden_layer_2_weights.txt', dtype='float32'))
layer_2_biases = tf.Variable(np.loadtxt('hidden_layer_2_biases.txt', dtype='float32'))
output_layer_weights = tf.Variable(np.loadtxt('output_layer_weights.txt', dtype='float32'))
output_layer_biases = tf.Variable(np.loadtxt('output_layer_biases.txt', dtype='float32'))

# load the mnist data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

INPUT_SIZE = mnist.test.images.shape[1]
OUTPUT_SIZE = mnist.test.labels.shape[1]

# create placeholders for inputs to the TensorFlow graph
x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

# TODO: build the actual model
learning_rate = 0.0001
epoch = 30
batch_size = 26
def nn(x):
	layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, layer_1_weights),layer_1_biases))
	layer2 = tf.nn.relu(tf.add(tf.matmul(layer1,layer_2_weights),layer_2_biases))
	y = tf.nn.softmax(tf.add(tf.matmul(layer2,output_layer_weights),output_layer_biases))
	return y

y_prediction = nn(x)
#cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = y_prediction))
cross_entropy = tf.reduce_mean(-np.sum( y * tf.log(y_prediction) + (1-y) * tf.log(1 - y_prediction)))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_prediction))
optimize = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range (epoch):
	batch_x, batch_y = mnist.train.next_batch(batch_size)
	op_,c = sess.run([optimize,cross_entropy], feed_dict = {x:batch_x, y:batch_y})


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_prediction, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)
print('accuracy of testing dataset:{}'.format(sess.run(accuracy,
                                        feed_dict={x: mnist.test.images,
                                                   y: mnist.test.labels})))
print('accuracy of training dataset:{}'.format(sess.run(accuracy,
                                        feed_dict={x: mnist.train.images,
                                                   y: mnist.train.labels})))
