from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Parameter
learning_rate = 0.0001
training_epochs = 20
batch_size = 100
display_step = 1

# load the mnist data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

INPUT_SIZE = mnist.test.images.shape[1]	#784
OUTPUT_SIZE = mnist.test.labels.shape[1]	#10
HIDDEN_1 = 100	#100	
HIDDEN_2 = 50	#50


# create placeholders for inputs to the TensorFlow graph
x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

layer_1_weights = tf.Variable(tf.random_normal([INPUT_SIZE,HIDDEN_1]))
layer_2_weights = tf.Variable(tf.random_normal([HIDDEN_1,HIDDEN_2]))
layer_1_biases = tf.Variable(tf.random_normal([HIDDEN_1]))
layer_2_biases = tf.Variable(tf.random_normal([HIDDEN_2]))
output_layer_weights = tf.Variable(tf.random_normal([HIDDEN_2,OUTPUT_SIZE]))
output_layer_biases = tf.Variable(tf.random_normal([OUTPUT_SIZE]))

def neural_network(feature):
	layer1 = tf.nn.sigmoid(tf.add(tf.matmul(feature, layer_1_weights ), layer_1_biases))
	layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, layer_2_weights), layer_2_biases))
	output = tf.add(tf.matmul(layer2, output_layer_weights), output_layer_biases)
	#output = tf.add(tf.matmul(layer2, output_layer_weights), output_layer_biases)
	#output = tf.Print(output, [output])
	return output

# Construct model
logits = neural_network(x)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.0
		total_batch = int(mnist.train.num_examples/batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([train_op, loss_op], feed_dict={x: batch_x,
                                                            y: batch_y})
            # Compute average loss
			avg_cost += c / total_batch
			# Display logs per epoch step
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
	print("Optimization Finished!")

    # Test model
	pred = logits
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))	
	# Calculate Accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("Accuracy(train):",accuracy.eval({x:mnist.train.images, y:mnist.train.labels}))
	print("Accuracy(test):", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

