#/usr/bin/env python
from gibbsupdate_with_f import Gibbs_update
import numpy as np
from scipy.optimize import LinearConstraint, minimize,Bounds
from PIL import Image
import pdb
import scipy.misc as smp
def evaluate_Q(X):
	size=X.shape;
	n=size[0];
	m=size[1];
	
	Q=np.zeros([(n-2)*(m-2),1]);
	for i in range(1,n-1):
		for j in range(1,m-1):
			
			adjacent = X[i-1:i+2,j-1:j+2];
			adjacent = np.delete(adjacent,[4]);
			Q[(i-1)*(m-2)+(j-1)] = np.sum(adjacent==X[i][j]);
	
	return Q

def pseudolikelihood(beta,*args):
	
	q=args[0];
	result=0;
	for value in q:
		result += np.log(1+np.exp(2*beta*(4-value)));
	
	return result/(q.shape[0])

def gradient(beta,*args):
		
	q=args[0];
	gradient=0;
	for value in q:
		gradient += 2*(4-value)*np.exp(2*beta*(4-value))*(1/(1+np.exp(2*beta*(4-value))));
	
	return gradient/(q.shape[0])
	
		
	
		
def learning(beta,original,threshold):
		
	bnds = Bounds(-2.5,2.5);
	Y=Gibbs_update(original,beta,125)
	q=evaluate_Q(Y);
	current_value=pseudolikelihood(beta,q);
	prev_value=current_value+10;

	while abs(prev_value-current_value)>1e-5:
		
		print("beta %f " %beta + "with pseudo-likelihood %f " %current_value+"and gain %f" %(abs(prev_value-current_value)) )
		prev_value=current_value;
		
		result= minimize(pseudolikelihood,[beta],args=(q),method='SLSQP',jac=gradient,bounds=bnds);
		beta=result.x[0];
		q=evaluate_Q(Y);
		current_value = pseudolikelihood(beta,q);
		Y=Gibbs_update(original,beta,-1,initial=Y)
	return Y
		
		
image = Image.open("random50_logo.png")
#image.show()
gray = np.asarray(image.convert('L'))
#implementing Gibbs_update
Y=learning(0.8,gray,125)
img = smp.toimage( Y )
img.save('final.png')
