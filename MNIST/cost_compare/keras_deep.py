from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adamax,SGD,Adagrad,Adam,Adadelta,RMSprop,Nadam
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels


batch_size = 126
num_classes = 10
epoches = 20
learning_rate = 0.0001

model = Sequential()
model.add(Dense(100,activation='sigmoid',input_shape=(784,)))
model.add(Dense(50,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))

model.summary()


model.compile(optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
			 loss='categorical_crossentropy',
			 metrics=['accuracy'])
#Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer=SGD(lr=learning_rate, momentum=0.9, nesterov=True)
#optimizer=Adagrad(lr=learning_rate, epsilon=1e-08, decay=0.0)
#optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer=Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-08, decay=0.0)
#optimizer=RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
#optimizer=Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

history = model.fit(x_train, y_train,
					batch_size = batch_size,
					epochs = epoches,
					validation_data = (x_test,y_test))

plt.figure(1)
plt.plot(history.history['loss'])
plt.title('change in loss at each epoch of Keras')
plt.xticks(np.arange(0,22,2))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

