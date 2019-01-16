from PIL import Image
import pdb
import scipy.misc as smp
from gibbsupdate_with_f import Gibbs_update
import numpy as np

image = Image.open("random50_logo.png")
image.show()
gray = np.asarray(image.convert('L'))

h=Gibbs_update(gray,0.8,125,'sequential')
img = smp.toimage( h );
img.save("Result_final.png")
