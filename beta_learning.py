#/usr/bin/env python
from gibbupdate import *
from scipy import optimize
import numpy as np

def exponent(beta,adjacent,x):
	return beta*np.sum(adjacent==x);

def pseudolikelihood(x,beta):
	# we just evaluate the numerator since the denumerator is constant
	if (beta<-2.5) and (beta>2.5):
		return 0 
	likelihood=1;
	for i=1:m
		for j=1:n
			adjacent = X[i-1:i+1,j-1:j+1];
			adjacent = np.delete(adjacent,4);	
			temp = exponent(beta,adjacent,X[i,j]); 
			likelihood*=np.exp(temp)/(np.exp(temp)+np.exp(beta*np.size(adjacent)-temp);

		
	
def gradient(x,beta):
	gradient=1;
	for i=1:m
		
def learning(beta,Y):
	(n,m)=shape(Y);	
	X=csc_matrix(Y.shape+1, dtype=np.int8).toarray();
	X[0,:]=-np.ones((1,n));
	X[-1,]=-np.ones((1,n));
	X[1:-1,0]=-np.ones((m-2,1));
	X[1:-1,-1]=-np.ones((m-2,1));
	X[1:-1,1:-1]=Y>threshold;

	X=Gibbs_update(Y,beta,threshold=125)
	prev_value=pseudolikelihood(X,beta);
	
	current_value=prev_value+10;
	while abs(prev_value-current_value)>eps:
		beta = optimize(pseusolikehood(x))
		
	
