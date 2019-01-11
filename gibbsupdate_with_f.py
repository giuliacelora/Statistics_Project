#!/usr/bin/env python2.7
import numpy as np
from scipy.sparse import lil_matrix
#import matplotlib.pyplot as plt
from PIL import Image
import pdb
import scipy.misc as smp
import random
def f(y_i,x_i):					
	if x_i == 1 and y_i <127:
			probability = 0.15
	if x_i == 1 and y_i >=127:
			probability = 0.85
	if x_i == 0 and y_i <127:
			probability = 0.85
	if x_i == 0 and y_i >=127:
			probability = 0.15	
	return probability

def update(y_node,adjacent,beta,mode='update',*argv):
	if mode=='update':
		delta_0 = np.sum(adjacent==0);
		delta_1 = np.size(adjacent)-delta_0;
		num_0 = f(y_node,0)*np.exp(beta*delta_0);
		num_1 = f(y_node,1)*np.exp(beta*delta_1);
		# adjacent must be passed as a numpy array
		return num_0<num_1 # if TRUE the pixel will be 1 otherwise 0
	else:
		
		current=argv[0];
		delta_current = np.sum(adjacent==current);
		delta_change = np.size(adjacent)-delta_current;
		num_current = f(y_node,current)*np.exp(beta*delta_current);
		num_change = f(y_node,abs(1-current))*np.exp(beta*delta_change);
		ratio = num_change/num_current;
		
		if ratio >1:
			return abs(1-current)
		else: 
			alpha = random.uniform(0,1);
			if (alpha < ratio):
				return abs(1-current)
			else:
				return current
				
		

# we denote by M the matrix representing the image 

def Gibbs_update(Y,beta,threshold,algorithm='sequential'):
	#threshold = 125
	#we pass the initial image on the gray scale and we switch it to the binary one, we can add a frame of -1 to denote the border
	size=np.shape(Y)	
	n = size[0];
	m=size[1];
	X=np.zeros((n+2,m+2), dtype=np.int8);
	
	X[0,:]=-np.ones((1,m+2));
	X[-1,:]=-np.ones((1,m+2));
	

	X[1:-1,0]=-1;
	X[1:-1,-1]=-1;
	X[1:-1,1:-1]=Y>threshold;

	img = smp.toimage( X[1:-1,1:-1] )       # Create a PIL image
	img.show()  
	if algorithm=='sequential':
		for iteration in range(5):
			for i in range(1,n+1):
				for j in range(1,m+1):
					
					node= np.array([i,j])
				
					adjacent = X[node[0]-1:node[0]+2,node[1]-1:node[1]+2];
				
					adjacent = np.delete(adjacent,[4]);
					prev=X[node[0],node[1]];
					X[node[0],node[1]]=update(Y[node[0]-1,node[1]-1],adjacent[adjacent>-1],beta)
					if X[node[0],node[1]]!=prev:
						print('change')
			print('Iteration %f done' %iteration)
		
		img = smp.toimage( X[1:-1,1:-1] )       # Create a PIL image
		img.show()  
	else:
		for iteration in range(200000):
			i=random.randint(1,n);
			j=random.randint(1,m);
			
			adjacent = X[i-1:i+2,j-1:j+2];
			adjacent = np.delete(adjacent,[4]);
			try:	
				X[i,j] = update(Y[i-1,j-1],adjacent[adjacent>-1],beta,algorithm,X[i,j]);
			except:

				pdb.set_trace()	
		img = smp.toimage( X[1:-1,1:-1] )       # Create a PIL image
		img.show()  
	                    # View in default viewer
	
		
#importing image and turning into greyscale (each pixel an array of values from 0-255)

image = Image.open("noisybw_logo.png")
#image.show()
gray = np.asarray(image.convert('L'))
#implementing Gibbs_update
Gibbs_update(gray,0.7,125,'MCMC')

 

			

