import pandas as pd
import idx2numpy
import numpy as np
from math import *
import random

images = idx2numpy.convert_from_file('../MNIST/curved_test_images')
labels = idx2numpy.convert_from_file('../MNIST/test_labels')

newi = np.ones((10000, 410), dtype = np.int8) * -1

for i in range(10000):
	if i % 1000 == 0:
		print(i)
	for j in range(400):
		newi[i][j] = images[i][j]
		newi[i][400 + labels[i]] = 1

idx2numpy.convert_to_file('../MNIST/complete_test_images', newi)