from conditional import f
import numpy as np
from scipy.sparse import csc_matrix


def update(y_node,adjacent,beta):
	delta_0 = np.sum(adjacent==0);
	delta_1 = np.size(adjacent)-delta_0;
	num_0 = f(y_node,0)*np.exp(beta*delta_0);
	num_1 = f(y_node,1)*np.exp(beta*delta_1);
	# adjacent must be passed as a numpy array
	return num_0<num_1 # if TRUE the pixel will be 1 otherwise 0

# we denote by M the matrix representing the image 

def Gibbs_update(Y,beta,threshold=125):
	#we pass the initial image on the gray scale and we switch it to the binary one, we can add a frame of -1 to denote the border 
	(n,m)=shape(Y);	
	X=csc_matrix(Y.shape+1, dtype=np.int8).toarray();
	X[0,:]=-np.ones((1,n));
	X[-1,]=-np.ones((1,n));
	X[1:-1,0]=-np.ones((m-2,1));
	X[1:-1,-1]=-np.ones((m-2,1));
	X[1:-1,1:-1]=Y>threshold;
	for iteration in range(5):
		for i in range(1,n+1):
			for j in range(1,n+1):
				node= np.array([i,j],dtype=np.int8)
		
				adjacent = X[node[0]-1:node[0]+1,node[1]-1:node[1]+1];
				adjacent = np.delete(adjacent,4);		
				X[node[0],node[1]]=update(Y[node[0]-1,node[1]-1],adjacent[adjacent>-1],beta)
		
)
			


	
