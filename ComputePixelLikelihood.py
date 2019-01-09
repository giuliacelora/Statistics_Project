#!/usr/bin/env python
# Function which takes as inputs a hidden satate x_i and a pixel intensity y_i and then gives the 
#probability that the observed intensity of pixel i is y_i and does not consider information from
#neigbouring particles

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
#import cv as cv2
from PIL import Image

#importing image and turning into greyscale (each pixel an array of values from 0-255)
import Image
image = Image.open("noisy_logo.png")
gray = np.asarray(image.convert('L'))
 

#defining the outputs of function
def likelihood(y_i,x_i):					
	if x_i == 1 and y_i <127:
			probability = 0.15
	if x_i == 1 and y_i >=127:
			probability = 0.85
	if x_i == 0 and y_i <127:
			probability = 0.85
	if x_i == 0 and y_i >=127:
			probability = 0.15	
	return probability


#Testing function by making plot#
x_i = 1			#value of hidden state x_i
probability = []		#array of probabilities
for y_i in range(0,255):
	probability.append(likelihood(y_i,x_i))
plt.plot(range(0,255),probability)
plt.show()

