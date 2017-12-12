# SGD
# Instead of computing our gradient over the entire data  set, we instead sample our data, yielding a batch

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse

def sigmoid_activation(x):
	return 1.0 / (1 + np.exp(-x))

def next_batch(X,y,batchSize):
	#loop over dataset "X" in mini-batches of size 'batch size'
	for i in np.arange(0, X.shape[0],batchSize):
		#yield a tuple of the current batched data and labels
		yield (X[i:i + batchSize], y[i:i + batchSize])

#construct the arguments parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type = float, default = 1000, help = "# of epochs")
ap.add_argument("-a", "--alpha", type = float, default = 0.01, help = "learning rate")
ap.add_argument("-b", "--batch_size", type = int, default = 32, help = "size of SGD mini-batches")
args = vars(ap.parse_args([]))

X,y = make_blobs(n_samples = 400, n_features = 2, centers =2, cluster_std = 2.5, random_state = 95)
X = np.c_[np.ones((X.shape[0])), X]

print("[INFO] starting training...")
W = np.random.uniform(size = (X.shape[1],))

lossHistory = []


#loop over rhe desired number of epoches
for epochs in np.arange(0, args["epochs"]):
	epochLoss= []

	#loop over our data in batches
	for (batchX, batchY) in next_batch(X, y, args["batch_size"]):
		preds = sigmoid_activation(batchX @ W)
		error = preds - batchY
		loss = np.sum (error**2)
		epochLoss.append(loss)
		gradient = batchX.T @ error / batchX.shape[0]
		W -= args["alpha"] * gradient

	lossHistory.append(np.average(epochLoss))

Y = (-W[0] -(W[1] * X))/ W[2]

plt.figure()
plt.scatter(X[:, 1], X[:, 2], marker = "o", c = y)
plt.plot(X, Y, "r-")

fig = plt.figure()
plt.plot(np.arange(0, args["epochs"]), lossHistory)
fig.suptitle("Training loss")
plt.xlabel("Epoch#")
plt.ylabel("Loss")
plt.show()



















