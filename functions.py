#!/usr/bin/env python
import pdb 
import scipy.io as sio
import numpy as np
import numpy.matlib
from numpy.linalg import cholesky, solve, svd
from multiprocessing import Pool
import scipy.misc as smp
import scipy.cluster.vq
import os

os.path.join('Statistics_Project/para_gmm/')

class par_type(object):
    def __init__(self,sigma,**kwargs):
        for item in kwargs:
            self.__dict__.update(kwargs)
        self.gamma=0.67;
        self.sigma=sigma;
        if sigma<=20:
            self.lmbda=0.18;
            self.size=7;
            self.tau=4.7;
            self.num_iter=4;
            self.model=MODEL("model7");
        elif sigma<=40:
            self.lmbda=0.18;
            self.size=8;
            self.tau=4.8;
            self.num_iter=5;
            self.model=MODEL("model8");
        elif sigma<=60:
            self.gamma=0.8;
            self.lmbda=0.15;
            self.size=9;
            self.tau=5.0;
            self.num_iter=5;
            self.model=MODEL("model9");
        else:
            self.gamma=1.0;
            self.lmbda=0.12;
            self.size=10;
            self.tau=5.2;
            self.num_iter=6;
            self.model=MODEL("model10");
                      


class MODEL(object):
    # this class contain all the information about the model learned for the clustering
    # this must contain the centre_vector the  Mahalanobi matrix for each class
    # it also must contain the weight of the Gaussian Mixed Model
    # receive also the number of classes K
    def __init__(self,name):
        mat_data=sio.loadmat('para_gmm/'+name+'.mat');
        self.Sigma=mat_data['model'][0][0][1];
        self.weight=mat_data['model'][0][0][2];
        self.size=self.weight.shape[-1];
    def __getattr__(self,string):
        for key in self.__dict__.keys():
            if key.lower()==string.lower():
                return self.__dict__[key]
    def probability(self,Y,K,par,deviation=0):

        sigma=self.Sigma[:,:,K]+deviation**2*np.identity(par.size**2);
        try:
            prob = np.log(self.weight[0,K])-0.5*par.size**2*np.log(2*np.pi);
        except:
            pdb.set_trace()
        L=np.linalg.cholesky(sigma);
        MHdistance=np.linalg.solve(L,Y);

        prob -= sum(np.log(np.diagonal(L)))+0.5*np.sum(np.power(MHdistance,2),0);

        return prob

   

def reassembling(Y,W,par):
    n,m=par.original_size;
    restore=np.zeros([n,m]);
    weight=np.zeros([n,m]);
    
    for i in range(par.size-1):
        for j in range(par.size-1):
            k=par.size*i+j;
            restore[i:-par.size+i+1,j:-par.size+j+1]=restore[i:-par.size+i+1,j:-par.size+j+1]+np.reshape(Y[k,:],[n-par.size+1,m-par.size+1]);
            weight[i:-par.size+i+1,j:-par.size+j+1]=weight[i:-par.size+i+1,j:-par.size+j+1]++np.reshape(W[k,:],[n-par.size+1,m-par.size+1]);

    	restore[i:-par.size+i+1,j+1:]=restore[i:-par.size+i+1,j+1:]+np.reshape(Y[k+1,:],[n-par.size+1,m-par.size+1]);
    	weight[i:-par.size+i+1,j+1:]=weight[i:-par.size+i+1,j+1:]+np.reshape(W[k+1,:],[n-par.size+1,m-par.size+1]);
    i+=1;
    for j in range(par.size-1):
        k=par.size*(i)+j;
        restore[i:,j:-par.size+j+1]=restore[i:,j:-par.size+j+1]+np.reshape(Y[k,:],[n-par.size+1,m-par.size+1]);
        weight[i:,j:-par.size+j+1]=weight[i:,j:-par.size+j+1]+np.reshape(W[k,:],[n-par.size+1,m-par.size+1]);
    print(k)
    restore[i:,j+1:]=restore[i:,j+1:]+np.reshape(Y[k+1,:],[n-par.size+1,m-par.size+1]);
    weight[i:,j+1:]=weight[i:,j+1:]+np.reshape(W[k+1,:],[n-par.size+1,m-par.size+1]);
    restore=np.divide(restore,weight+1e-16);
    
    return restore 
    


def threshold_fun(a,b):
    return np.sign(a)*np.maximum(np.abs(a)-b,0)

def low_rank_approximation(Rx,K,par,sigma):
    m=np.outer(np.mean(Rx,1), np.ones([1,Rx.shape[1]]));
    Rx=Rx-m;
    try:
        U,S,Vh=svd(Rx,full_matrices=False);
    except:
        pdb.set_trace()
    sv_Z=np.sqrt(np.maximum(np.power(S,2)/Rx.shape[1]-sigma**2,0));
    threshold = par.tau*sigma**2*np.reciprocal(sv_Z+1e-16,dtype=float);
    sv_Z = threshold_fun(S,threshold);
    index = np.argwhere(sv_Z>0)[:,-1];
    
    U=U[:,index];
    Vh=Vh[index,:];
    Z=np.matmul(np.matmul(U,np.diag(sv_Z[index])),Vh)
    if index[-1]==Rx.shape[0]-1:
        weight=1/Rx.shape[0];#weight used to reassemble the picture
    else:
        weight=(Rx.shape[0]-index[-1]-1)/float(Rx.shape[0]);
   
    W=weight*np.ones(Z.shape);
    Z=weight*(Z+m);
    return Z,W
        
        


def patches(image,original,par):
    (n,m)=par.original_size;
    num_patches = (n-par.size+1)*(m-par.size+1);
    Y=np.zeros([par.size**2,num_patches]); #current patches 
    X=np.zeros(Y.shape); #patches extracted from the detection
    k=0;

    for i in range(par.size-1):
        for j in range(par.size-1):

            k=par.size*i+j;      # Create a PIL image
            Y[k,:]=np.reshape(image[i:-par.size+i+1,j:-par.size+j+1].transpose(),-1);
            X[k,:]=np.reshape(original[i:-par.size+i+1,j:-par.size+j+1].transpose(),-1);
	    
        k=par.size*i+j+1;
        Y[k,:]=np.reshape(image[i:-par.size+i+1,j+1:].transpose(),-1);
        X[k,:]=np.reshape(original[i:-par.size+i+1,j+1:].transpose(),-1);

    for j in range(par.size-1):
        k=par.size*(i+1)+j;
        Y[k,:]=np.reshape(image[i+1:,j:-par.size+j+1].transpose(),-1);
        X[k,:]=np.reshape(original[i+1:,j:-par.size+j+1].transpose(),-1);

    
    Y[-1,:]=np.reshape(image[i+1:,j+1:].transpose(),-1);
    X[-1,:]=np.reshape(original[i+1:,j+1:].transpose(),-1);
    
    Sigma = np.sqrt(par.gamma*abs(par.sigma**2-np.mean(np.power(X-Y,2),0)));
    return Y, Sigma
    



def cluster(Y,par,sigma_l):
    model=par.model;

    # for each patch we need to find the class that maximize the log-likelihood
    # the learning has been done by subtracting the group mean
    prob=np.zeros([model.size,Y.shape[-1]]);
    Y=Y/255.0; 
    My=np.mean(Y,0);
    Y=Y-My;
    
    dev=np.mean(sigma_l)/255;
    for i in range(model.size):

        prob[i,:]=model.probability(Y,i,par,dev);

    k=np.argmax(prob,axis=0); #class to which the patch belongs to 
    group=[]
    for i in range(model.size):
        if np.sum(k==i)>10:
            group.append(i)

    k=np.argmax(prob[group,:],axis=0); 
    k=np.array([group[y] for y in k]);
    R={}

    for item in group:
        R[item]=np.argwhere(k==item);
    return R,My
        



def Denoising(noise_image,sigma,par,name="image"):
    z=noise_image;
    for i in range(par.num_iter):
        z=z+par.lmbda*(noise_image-z);
        Y, Sigma_l=patches(z,noise_image,par);
        print("Patches created")
        if i==0:
            Sigma_l=sigma*np.ones(Sigma_l.shape);
        print(np.mean(Sigma_l))
        R,MY=cluster(Y,par,Sigma_l);
        print("Clustering done")
	
        Z={}
        W=np.zeros(Y.shape)
        for flag in R.keys():
            if R[flag].shape[0]>sigma*1000:
                N=int(np.ceil(R[flag].shape[0]/(sigma*1000)));
                ind=R[flag];
                data=np.zeros((R[flag].shape[0],3));
                data[:,0:1]=(np.mod(ind,par.original_size[0]-par.size)-0.5*par.original_size[1])/(0.5*par.original_size[1]);
                data[:,1:2]=(np.floor(ind/(par.original_size[0]-par.size))-0.5*par.original_size[0])/(0.5*par.original_size[0]);
          	data[:,2:]=MY[ind]
                centroid,k=scipy.cluster.vq.kmeans2(data,N,iter=50)
                for j in range(N):
                    indeces=R[flag][k==j,0];
                    if indeces.shape[0]>1:
                        Y[:,indeces],W[:,indeces]=low_rank_approximation(Y[:,indeces],flag,par,np.mean(Sigma_l[indeces]));
            else:
                indeces=R[flag][:,0];
                Y[:,indeces],W[:,indeces]=low_rank_approximation(Y[:,indeces],flag,par,np.mean(Sigma_l[indeces]));

        print("Low Rank Approximation completed")
        z=reassembling(Y,W,par);
        pdb.set_trace()
        img = smp.toimage( z )       # Create a PIL image
        img.show()
        img.save(name+'%f'%i+'.png')





