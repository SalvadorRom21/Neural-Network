import numpy as np 

#creating a sigmoid function
#turns a number into a probability

def nolin(x, deriv=False)			
	if (deriv==True):				
		return (x*(1-x))


return 1/(1+np.exp(-x))


x = np.array([[0,0.1],
[0,1,1],
[1,0,1],
[1,1,1]])

y = np.array([[0],
[1],
[1],
[0]])