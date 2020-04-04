import pandas as pd
import idx2numpy
import numpy as np
from math import *
from random import randrange as rnd,choice

curve = idx2numpy.convert_from_file('../MNIST/curve')
images = idx2numpy.convert_from_file("../MNIST/spin_test_images")
curved = np.zeros((10000, 400), dtype = np.int8)

for n in range(10000):
	for i in range(400):
		curved[n][i] = images[n][curve[i][0]][curve[i][1]]

idx2numpy.convert_to_file("../MNIST/curved_test_images", curved) 