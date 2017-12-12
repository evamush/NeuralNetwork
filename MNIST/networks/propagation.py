# forward propagation

import numpy as np

def fc_forward(x,w,b):

	"""
	computes the forward pass for an affine (fully connected layer)
	inputs:
	- x: input tensor(N, d_1,..., d_k). (N,D)
	- w: weights (D,M)
	- b: Bias (M,)

	N:  mini-batch size
	M:  no of outputs of fully connected layer
	D:  input dimension

	returns a tuple of:
	- out: output of shape(N,M)
	- cache: (x,w,b)
	"""
	out = None

	# get the batch size in first dimension
	N = x.shape[0]

	# reshape activations to [Nx(d_1,...d_l)], which will be a 2d matrix
	# [NxD]
	# -1 here is unspecified value
	reshaped_input = x.reshape(N, -1)

	# claculate output
	out = np.dot(reshaped_input, w) + b.T

	#save inputs for backwards propagation
	cache = (x,w,b)
	return out, cache

def fc_backward(dout, cache):
	"""
	computes the backwards pass for an affine layer
	inputs:
	- dout: layer partial derivative wrt loss of shape(N,M) same as output
	- cache: inputs from forward computation
	
	N:  mini-batch size
	M:  no of outputs of fully connected layer
	D:  input dimension
	d_1,...,d_k single input dimension

	return a tuple of :
	- dx: gradient with respect to x, of shape(N, d_1,...,d_k)
	- dw: gradient with respect to w, of shape(d,M)
	- db: gradient with respect to b, of shape(M,)

	"""
	x, w, b = cache
	dx, dw, db = None, None, None

	# get batch size first dimension
	N = x.shape[0]

	#get dx same format as x
	dx = np.dot(dout, w.T)
	dx = dx.reshape(x.shape)

	# get dw same format as w
	# reshape activation to [Nx[d_1,...,d_k]], which will be a 2d matrix
	# [NxD]
	reshaped_input = x.reshape(N, -1)

	# transpose then dot product with dout
	dw = reshaped_input.T.dot(dout)

	#get db same format as b
	db = np.sum(dout, axis=0)

	#return outputs
	return dx, dw, db





def relu_forward(x):
	"""
	Computes the forward pass for ReLU
	input:
	- x:inputs, of any shape

	returns a tuple of:(out, cache)
	the shape on the output is the same as input
	"""

	out = None

	# create a function that receive x and return x 
	# if x is bigger than 0 or zero if negative
	relu = lambda x: x if x >0 else 0
	out = relu(x)

	# cache input and return outputs
	cache = x
	return out, cache


def relu_backward(dout, cache):
	"""
	Computes the backward pass for ReLU
	Input:
	- dout: upstream derivatives of any shape
	- cache: Previous input (used on forward propagation)

	return:
	- dx: Gradient with respect to x
	"""

	#initialize dx with None and x with cache
	dx, x = None, cache

	# make all positive elements in x equal to dout while all the other elements become 0 
	dx = dout * (x >= 0)

	# return dx gradient with respect to x
	return dx




def dropout_forward(x, dropout_param):
	"""
	perfors the forward pass for inverted dropout
	input:
	- x: input data, of any shape
	- dropout_param: a dictionary with the following kets(p, test/train,seed)

	outputs:(out,cache)
	"""

	# get the current dropout mode, p , and seed
	p, mode = dropout_param['p'], dropout_param['mode']
	if 'seed' in dropout_param:
		np.random.seed(dropout_param['seed'])

	# initialization of the outputs and mask
	mask = None
	out = None

	if mode == "train":
		# Create an apply mask (normally p=0.5 for half of neurons), we scale all
		# by p to avoid having to multiply by p on backpropagation,
		# this is called inverted dropout
		mask = (np.random.randn[*x.shape]<p)/p
		#apply mask
		out = x * mask

	elif mode == "test":
		# during prediction no mask is used
		mask = None
		out = x

	# Save mask and dropout parameters for backpropagation
	cache = (dropout_param, mask)

	#convert "out" type and return output and cache	
	out = out.astype[x.dtype, copy=false]
	return out,cache


def dropout_backward(dout,cache):
	"""
	perform the backward pass for inverted dropout
	input:
	- dout: upstream derivatives of any shape
	- cache: (dropout_param, mask) from dropout_forward
	"""
	# recover dropout parameters(p,mask,mode)from cache
	dropout_param, mask = cache
	mode = dropout_param['mode']

	dx = None
	# back propagate (dropout layer has no parameters just input x)
	if mode == "train":
		# just back propagate dout from the neurons that were used during dropout
		dx = dout * mask

	elif mode == "test":
		# disable dropout during prediction/test
		dx = dout

	return dx




def conv_forward_naive(x,w,b,conv_param):
	"""
	compputes the forward pass for the convolution layer (naive)
	input:
	- x: input data of shape(N,C,H,W)
	- w: filter weights of shape(F,C,HH,WW)
	- b: biases of shape(F,)
	- conv_param: a dictionary wirh the following keys:
		- 'stride': how much ouxels the sliding window will travel
		- 'pad': the number of pixels that will be used to zero-pad the input

	N: mini-batch size
	C: input depth ie 3 dor RGB images
	H/W: image height/width
	F: Number of filters on convolution layer will be the output depth
	HH/WW: kernel height/width

	returns a tuple of:
	- out: output data of shape(N,F,H',W') where H' and W' are given by
	  H' = 1 + (H + 2 * pad - HH)/stride
	  W' = 1 + (W + 2 * pad - WW)/stride
	- cache: (x,w,b,conv_param)
	"""
	out = None
	N,C,H,W = x.shape
	F,C,HH,WW = w.shape

	# get parameters
	p = conv_param["pad"]
	s = conv_param["stride"]

	# calculate output size and initialize output volume
	H_R = 1 + (H + 2 * p - HH) / s
	W_R = 1 + (w + 2 * p - WW) / s
	out = np.zeros((N,F,H_R,W_R))

	# pad images with zeros on the border used to keep spatial info
	x_pad = np.lib.pad(x,((0,0),(0,0),(p,p),(p,p)), 'constant', constant_values=0)

	# apply the convolution
	for n in xrange(N): #for each elements on batch
		for depth in xrange(F):#for each input depth
			for r in xrange(0,H,s):#slide vertically taking stride into account
				for c in xrange(0,W,s):#slide horizontally taking stride into account
					out[n,depth,r/s,c/s] = np.sum(x_pad[n,:,r:r+HH,c:c+WW] * w[depth,:,:,:] + b[depth]



	# cache parameters and inputs for backpropagation and return output volume
	cache = (x,w,b,conv_param)
	return out, cache


def conv_backward_naive(dout, cache):

	"""
	computes the backward pass for convolution layer naive
	inputs:
	- dout: upstream derivatives
	- cache: a tuple of (x,w,b,cov_param) as in conv_forward_naive
	returns a tuple of : (dw,dx,db) gradients
	"""
	dx, dw, db = None, None, None
	x, w, b, conv_param= cache
	N, F, H_R, W_R = x.shape
	F, C, HH, WW = w.shape
	p = conv_param["pad"]
	s = conv_param["stride"]
	# do zero padding on x_pad
	x_pad = np.lib.pad(x,((0,0),(0,0),(p,p),(p,p)),'constant',constant_values=0)

	# initialize outputs
	dx = np.zeros(x_pad.shape)
	dw = np.zeros(w.shape)
	db = np.zeros(b.shape)

	# calculate dx with 2 extra col/row that will deleted
	for n in xrange(N):
		for depth in xrange(F):
			for r in xrange(0,H,s):
				for c in xrange(0,W,s):
					dx[n,:,r:r+HH,c:c+WW] += dout[n,depth,r/s,c/s] * w[depth,:,:,:]

	# deleting padded rows to match real dx
	delete_rows = range(p) + range(H+p,H+2*p,1)
	delete_columes = range(p) +range(W+p,W+2*p,1)
	dx = np.delete(dx, delete_rows,axis=2) #height
	dx = np.delete(dx, delete_columes,axis=3) #width

	# calculate dw
	for n in xrange(N):
		for depth in xrange(F):
			for r in xrange(H_R):
				for c in xrange(W_R):
					dw[depth,:,:,:] += dout[n,depth,r,c] * x_pad[n,:,r*s:r*s_HH,c*s:c*s+WW]

	# calculate db 1 scalar bias per filyer, so its just a matter of summing
	#all elements of dout per filter
	for depth in ranfe(F):
		db[depth] = np.sum(dout[:, depth, :, :])
	return dx,dw,db
















































