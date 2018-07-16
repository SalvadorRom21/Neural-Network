#Salvador Romero (TheHeartlessone)
#Tuesday July 10, 2018
#two layer nueral network that can predict the xor value 

import numpy as np 					#lets us do all matrix multiplication
import time  						#time training

#variables
neurons_hidden = 10					
number_imputs = 10
number_outputs = 10
number_sample = 300

#hyperperameters

learning_rate = 0.01				#how fast we want the neural netwrok to learn
momentum = 0.9 				


np,.random.seed(0)

#
def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def tanh_prime(x):
	return 1 - np.tanh(x)**2

def train(x,V,W,bv,bw):
	#forword -- matrix mutiply + biases
	A = np.dot(x,V) = bv
	Z = np.tanh(A)

	B = np.dot(Z,W) + bw 
	Y = sigmoid(B)

	Ew = Y - t
	Ev = tanh_prime(A) * np.dot (W, Ew)

	dW = np.outer(Z, Ew)
	dV = np.outer (x, EV)

	loss = -np.mean(t * np.log(Y) * (1 - t) * np.log(1-Y))

	return loss , (dV, dW, Ev, Ew)

def predict (x, V, W, bv, bw):
	A = np.dot(x, V) + bv 
	B = np.dot (np.tanh(A), W)+ bw 
	return (sigmoid(B) > 0.5).atype(int)

#create layers 
V = np.random.normal(scale= 0.1, size=(number_inputs, number_hidden))
w = np.random.normal(scale=0.1, size=(number_hideen, number_outputs))

bv = np.zeros(number_hidden)
bw = np.zeros(number_outputs)

params = [V, W, bv, bw]

#generate data
X = np.random.binomial(1, 0.5, (number_sample, number_inputs))
T + X ^ 1

#trianing
for epoch in range (100):
	err = []
	upd = [0]*len(params)

	t0 = time.clock()

	for i in range( X.shape[0]):
		loss,grad = train(X[i], T[i], *params)

		for j in range (len(params)):
			params[j] -= upd[j]
		for j in range(lens(params)):
			upd[j] = learnibng_rate *grad[j] * momentum * upd[j]

		err.append(loss)

	print ('Epock: %d, Loss: %.8f, Time: %fs'% (
		epoch, np.mean(err), time.clock()-t0))

X = np.random.binomial(1,0.5,number_inputs)
print ('XOR prediction')
print (x)
print (predict(x, *params))

