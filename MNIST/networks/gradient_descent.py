# import packages
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse


def sigmoid_activation(x):
	return 1.0 / (1 + np.exp(-x))

#construct the argument parse and parse the argments 
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type = float, default = 1000, help = "# of epochs")
ap.add_argument("-a", "--alpha", type = float, default = 0.01, help = "learning rate")
args = vars(ap.parse_args([]))

# generate a 2-class classification problem with 250 data points,
# where each data point is a 2D feature vector
(X,y) = make_blobs(n_samples = 250, n_features = 2, centers = 2, cluster_std = 1.05, random_state = 20)

# insert a column of 1's as the first entry in the feature
# vector -- this is a little trick that allows us to treat
# the bias as a trainable parameter *within* the weight matrix
# rather than an entirely separate variable
X = np.c_[np.ones(X.shape[0]),X]

# X.shape[1] -->3columns
# X.shape[0] -->250
# W between 0 and 1
print("[INFO] starting training...")
W = np.random.uniform(size=(X.shape[1],))

# initialize a list to store the loss value for each epoch
lossHistory = []

#loop over the desired number of epoch
# range, xrange, np.arange time decreases
for epoch in np.arange(0, args["epochs"]):
  preds = sigmoid_activation(X.dot(W))
  error = preds - y
  loss = np.sum(error**2)  
  lossHistory.append(loss)
  print("[INFO] epoch #{}, loss={:.7f}".format(epoch + 1, loss))
  gradient = X.T.dot(error) / X.shape[0]
  W -= args["alpha"] *gradient
  
  
for i in np.random.choice(250, 10):
# randomly chose 10 rows/datasets from X
  activation = sigmoid_activation(X[i].dot(W))
  label = 0 if activation <0.5 else 1
  print("activation = {:.4f}; predicted_label = {}, true_label = {}".format(activation, label, y[i]))
  
Y = (-W[0] - (W[1] * X)) / W[2]

plt.figure()
plt.scatter(X[:, 1], X[:, 2], marker = "o", c = y)
plt.plot(X, Y, "r-")
fig = plt.figure()
plt.plot(np.arange(0, args["epochs"]), lossHistory)
plt.suptitle("training loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
print(W)