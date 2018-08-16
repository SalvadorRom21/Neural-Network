#Salvador Romero (TheHeartlessone)
#Tuesday July 10, 2018
#two layer nueral network that can predict the xor value 
#feedforward neural network

import numpy as np 					#lets us do all matrix multiplication
import time  						#time how long our training takes

#variables
neurons_hidden = 10					#number of hidden neruons
number_inputs = 10					#number of inputs neurons
number_outputs = 10					#number of output neurons
number_sample = 300					#this is how much data were going to generate

#hyperperameters

learning_rate = 0.01				#how fast we want the neural netwrok to learn
momentum = 0.9 						#how were going to lower cross entropy as we train 
	
#non deterministic seeding
np.random.seed(0)					#generates the same random numbers everytime we run our code
									
#activation functions for our neurons
#we used both sigmoid and tanh_prime for XOR because of their properties	
def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))	#function that is ran in every neuron in our network

def tanh_prime(x):
	return 1 - np.tanh(x)**2
#writing our training function
#this function will take five parameters
#x is input data, t is our transpose,V and W are our layers, 
# bv and bw are our biases
def train(x,t,V,W,bv,bw):
	
	#forword -- matrix mutiply + biases
	A = np.dot(x,V) + bv
	Z = np.tanh(A)
	# taking the dot product of x and puting it into our first layer V, the dot product
	# is doing matrix multiplication and adding the bias in 
	# then we perform our activation function  on that data

	B = np.dot(Z,W) + bw 
	Y = sigmoid(B)
	# take the Z value and do matrix multiplication and then add the bias 
	# then calculate the sigmoid function 

	#backward propigation
	Ew = Y - t
	Ev = tanh_prime(A) * np.dot (W, Ew)
	# Transpose is our matrix of wights flipped since were going backwards 
	# we want the Ev to predict our loss and compare our predictive loss function from 
	# our actual loss function and try to minimize our loss doing this.

	#predict our loss 

	dW = np.outer(Z, Ew)
	dV = np.outer(x, Ev)

	#were doing matrix multiplication and get the delta values

	#cross entropy loss function for classification
	loss = -np.mean(t * np.log(Y) * (1 - t) * np.log(1-Y))



	return loss, (dV, dW, Ev, Ew)
	#returns delta values and our error values

#Prediction function that take in the same parameters as our train function 
#except our transpose parameter
def predict (x, V, W, bv, bw):
	A = np.dot(x, V) + bv 
	B = np.dot (np.tanh(A), W)+ bw 
	return (sigmoid(B) > 0.5).astype(int)		#if our value is greater than .05 we will
												# return a 1 else we will return a 0


#create layers 
V = np.random.normal(scale= 0.1, size=(number_inputs, neurons_hidden))
W = np.random.normal(scale=0.1, size=(neurons_hidden, number_outputs))

bv = np.zeros(neurons_hidden)
bw = np.zeros(number_outputs)

params = [V, W, bv, bw]

#generate data
X = np.random.binomial(1, 0.5, (number_sample, number_inputs))
T = X ^ 1

#trianing
for epoch in range (100):
	err = []
	upd = [0]*len(params)

	t0 = time.clock()
	#for each data point , were updating our weights.

	for i in range( X.shape[0]):
		loss,grad = train(X[i], T[i], *params)
		#then we will update our loss

		for j in range (len(params)):
			params[j] -= upd[j]
		for j in range(len(params)):
			upd[j] = learning_rate *grad[j] * momentum * upd[j]

		err.append(loss)

	print ('Epock: %d, Loss: %.8f, Time: %fs'% (
		epoch, np.mean(err), time.clock()-t0))

x = np.random.binomial(1,0.5,number_inputs)
print ('XOR prediction')
print (x)
print (predict(x, *params))

