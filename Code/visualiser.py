import numpy as np
from PIL import Image
import idx2numpy
import os
from random import randrange as rnd,choice

def vis_test(n = -1):
	if n == -1:
		n = rnd(0, 1000)
	ndarr = idx2numpy.convert_from_file("../MNIST/spin_test_images")
	pic = np.empty( (400, 400), dtype = np.uint8)
	labels = idx2numpy.convert_from_file("../MNIST/test_labels")

	for i in range(20):
		for j in range(20):
			if ndarr[n][i][j] == 1:
				for l in range(20):
					for m in range(20):
						pic[20 * i + l][20 * j + m] = 255
			else:
				for l in range(20):
					for m in range(20):
						pic[20 * i + l][20 * j + m] = 0
	new_im = Image.fromarray(pic)
	new_im.save("../Misc/new_image.png")
	print(labels[n])
	os.system('eog ../Misc/new_image.png')

def vis_train(n = -1):
	if n == -1:
		n = rnd(0, 1000)
	ndarr = idx2numpy.convert_from_file("../MNIST/spin_train_images")
	pic = np.empty( (560, 560), dtype = np.uint8)
	labels = idx2numpy.convert_from_file("../MNIST/train_labels")

	for i in range(28):
		for j in range(28):
			for l in range(20):
				for m in range(20):
					pic[20 * i + l][20 * j + m] = ndarr[n][i][j]
	new_im = Image.fromarray(pic)
	new_im.save("../Misc/image.png")
	print(labels[n])
	os.system('eog ../Misc/image.png')
