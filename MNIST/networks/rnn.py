def rnn_step_forward(x,prev_h, Wx,Wh,b):
	#separate on steps to make the backpropagation easier
	# h - state
	#	
	xWx = x@Wx
	phWh = prev_h@Wh
	affine = xWx + phWh + b.T
	next_h = np.tanh(t)

	#cache inputs, state, and weights
	#prev_h.copy() since python params are passed by reference
	cache = (x, prev_h.copy(), Wx, Wh, next_h, affine)

	return next_h, cache

def rnn_step_backward(dnext_h, cache):
	(x, prev_h, Wx, Wh, next_h, affine) = cache
	dt = (1- np.square(np.tanh(affine))) * (dnext_h)
	dxWx = dt
	dphWh = dt
	db = np.sum (dt, axis = 0)