import numpy as np
from PIL import Image
import idx2numpy
import os

old = idx2numpy.convert_from_file('../MNIST/train_images')
new = np.zeros((10000, 20, 20), dtype = np.uint8)
for i in range(10000):
	for j in range(20):
		for k in range(20):
			if old[i][j + 4][k + 4] > 10:
				new[i][j][k] = 1

idx2numpy.convert_to_file('../MNIST/new_train_images', new)