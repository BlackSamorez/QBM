import pandas as pd
import idx2numpy
import numpy as np
from math import *
from random import randrange as rnd,choice

def calc(number):
	curve = idx2numpy.convert_from_file('../MNIST/curve')
	images = idx2numpy.convert_from_file("../MNIST/spin_train_images")
	labels = idx2numpy.convert_from_file("../MNIST/train_labels")
	double = np.zeros((410, 410), dtype = np.float_)
	for n in range(10000 * number, 10000 * (number + 1)):
		for i in range(400):
			for j in range(400):
				double[i][j] += images[n][curve[i][0]][curve[i][1]] * images[n][curve[j][0]][curve[j][1]]
				if i == 0 and j == 0:
					print(n)
	for i in range(400):
		for n in range(10000 * number, 10000 * (number + 1)):
			double[i][400 + labels[n]] += images[n][curve[i][0]][curve[i][1]]
			double[400 + labels[n]][i] += images[n][curve[i][0]][curve[i][1]]
	for i in range(410):
		for j in range(410):
			double[i][j] = double[i][j] / 60000
	idx2numpy.convert_to_file("../Data/double" + str(number), double)

a = input()
calc(int(a))