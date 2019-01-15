import numpy as np
from scipy.stats import multivariate_normal

from functions import *
import pdb
from PIL import Image
import scipy.io as sio

picture = Image.open("peppers.png");
real_picture = np.asarray(picture.convert('L'))

#sigma_noise=input('Insert the variation of the noise:');
sigma_noise=30;
Y = real_picture+np.random.normal(0,sigma_noise,real_picture.shape);

par=par_type(sigma_noise,original_size=Y.shape);

Denoising(Y,sigma_noise,par)
