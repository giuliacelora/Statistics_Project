#/usr/bin/env python
from gibbupdate import *
from scipy import optimize
import numpy as np

def evaluate_Q(X):
	size=np.shape(X);
	n=size[0];
	m=size[1];
	Q=zeros([n*m,1]);
	for i in range(n):
		for j in range(m):
			
			adjacent = X[i:i+3,j:j+3];
				
			adjacent = np.delete(adjacent,[4]);
			Q[i*m+j] = np.sum(adjacent==X[i][j]) 
	return Q

def pseudolikelihood(x,beta):
	
	if (beta<-2.5) and (beta>2.5):
		return 0 
	q=evaluate_Q(x);
	result=0;
	gradient=0;
	for i in range(length(q)):
		result +=-np.log(1+np.exp(2*beta*(4-q[i]))));
		result_2  +=-2*(4-q[i])*(1/1+np.exp(2*beta(4-qi)))
	
	return result, result_2
		
	
		
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
		
	
