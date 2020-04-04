import pandas as pd
import idx2numpy
import numpy as np
from math import *
from random import randrange as rnd,choice

images = idx2numpy.convert_from_file("../MNIST/new_test_images")

spin = np.zeros((10000, 20, 20), dtype = np.int8)

for i in range(10000):
	for j in range(20):
		for k in range(20):
			if images[i][j][k] == 0:
				spin[i][j][k] = -1
			else:
				spin[i][j][k] = 1

idx2numpy.convert_to_file("../MNIST/spin_test_images", spin)