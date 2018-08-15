import numpy as np 
import numpy.linalg as LA

## https://en.wikipedia.org/wiki/Sparse_dictionary_learning 
## Suppose we have X, 
## || X - D R ||_F^2 + \lambda || R ||_F^2
## X \in R^{d * N}    
## D \in R^{d * k}, dictionary
## R \in R^{k * N}, code


## Alternative Minimize
##   0. initialize D and R

##   1. Given D,  
##   tr[(X-DR)^T(X-DR)] + \lambda || R ||_F^2 
## = -2 tr(X^T D R) + tr[R^T D^T D R] + lambda tr[ R R^T ] 
## = -2 tr(X^T D R) + tr[D^T D R R^T] + lambda tr[ R R^T ] 
## = -2 tr(X^T D R) + tr[(D^T D + lambda R R^T) R R^T]

## =>  R = inv(D^T D + lambda * I) * (X^T D)


##   2. Given R,  D = inv(R * R^T) * (R * X^T) 

def optimize_R(X, D, lamb):
	d, N = X.shape
	d2, k = D.shape
	assert d == d2 
	I = np.identity(k)
	I = np.matrix(I)
	R = LA.inv(D.T * D + lamb * I) * D.T * X
	return R 

def optimize_D(X, R):
	d, N = X.shape 
	k, N2 = R.shape 
	assert N == N2 
	D = LA.inv(R * R.T) * (R * X.T)
	D = D.T
	return D 

def evaluate_loss(X, R, D, lamb):
	#print(X.shape)
	#print(R.shape)
	#print(D.shape)
	tmp = D * R
	l1 = LA.norm(X - D * R, 'fro') ** 2 
	l2 = lamb * (LA.norm(R, 'fro') ** 2)
	return l1 + l2 

if __name__ == '__main__':
	N = 1000
	d = 100
	k = 10 
	lamb = 1
	iteration = int(1e7)
	X = np.random.random((d,N))
	X = np.matrix(X)
	D = np.random.random((d,k))
	D = X[:,:k]
	D = np.matrix(D)
	R = np.random.random((k,N))
	R = np.matrix(R)


	for i in range(iteration):
		R = optimize_R(X, D, lamb)
		D = optimize_D(X, R)
		loss = evaluate_loss(X, R, D, lamb)
		print(loss)










