import pandas as pd
import idx2numpy
import numpy as np
from math import *
import random

dset = np.zeros((1000, 6), dtype = np.int32)

for i in range(1000):
	coef = random.randrange(4)
	if coef == 0:
		dset[i] = [-1, -1, 1, -1, -1, -1]
	if coef == 1:
		dset[i] = [1, -1, -1, 1, -1, -1]
	if coef == 2:
		dset[i] = [-1, 1, -1, -1, 1, -1]
	if coef == 3:
		dset[i] = [1, 1, -1, -1, -1, 1]
idx2numpy.convert_to_file("../Data/tests/data", dset)