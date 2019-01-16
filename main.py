import numpy as np
from scipy.stats import multivariate_normal

from functions import *
import pdb
from PIL import Image
import scipy.io as sio
import os

os.path.join('Statistics_Project/para_gmm/')

picture = Image.open('Lenna.png');
picture=picture.convert('LA')
picture.show()
real_picture = np.asarray(picture.convert('L'))

#sigma_noise=input('Insert the variation of the noise:');
np.random.seed(0) 
sigma_noise=50;
Y = real_picture+np.random.normal(0,1,real_picture.shape)*sigma_noise;
img = smp.toimage( Y )       # Create a PIL image
img.show()
img.save('lenna_noisy50.bmp')
par=par_type(sigma_noise,original_size=Y.shape);

Denoising(Y,sigma_noise,par,name='lenna50_2_')
