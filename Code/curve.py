import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from PIL import Image
import json
import idx2numpy

def create_curve():
	hilbert_curve = HilbertCurve(4, 2)
	d = np.dtype(np.uint8)
	curve = np.empty((400, 2), dtype = d)
	reverse_curve = np.empty((20, 20), dtype = d)
	for i in range(9):
		curve[4 * i] = [2 * i, 0]
		curve[4 * i + 1] = [2 * i, 1]
		curve[4 * i + 2] = [2 * i + 1, 1]
		curve[4 * i + 3] = [2 * i + 1, 0]
	curve[36] = [18, 0]
	curve[37] = [19, 0]
	for i in range(9):
		curve[38 + 4 * i] = [19, 2 * i + 1]
		curve[38 + 4 * i + 1] = [18, 2 * i + 1]
		curve[38 + 4 * i + 2] = [18, 2 * i + 2]
		curve[38 + 4 * i + 3] = [19, 2 * i + 2]
	curve[74] = [19, 19]
	curve[75] = [18, 19]
	for i in range(8):
		curve[76 + 4 * i] = [17 - 2 * i, 19]
		curve[76 + 4 * i + 1] = [17 - 2 * i, 18]
		curve[76 + 4 * i + 2] = [17 - 2 * i - 1, 18]
		curve[76 + 4 * i + 3] = [17 - 2 * i - 1, 19]
	curve[108] = [1, 19]
	curve[109] = [0, 19]
	for i in range(8):
		curve[110 + 4 * i] = [0, 19 - 2 * i - 1]
		curve[110 + 4 * i + 1] = [1, 19 - 2 * i - 1]
		curve[110 + 4 * i + 2] = [1, 19 - 2 * i - 2]
		curve[110 + 4 * i + 3] = [0, 19 - 2 * i - 2]
	curve[142] = [0, 2]
	curve[143] = [1, 2]

	for i in range(256):
		curve[144 + i][0] = hilbert_curve.coordinates_from_distance(i)[0] + 2
		curve[144 + i][1] = hilbert_curve.coordinates_from_distance(i)[1] + 2



	'''for i in range(399):
		if (curve[i][0] != curve[i + 1][0] and curve[i][1] != curve[i + 1][1]):
			print("error: ", i)'''

	'''pic = np.zeros([400, 400, 3], dtype=np.uint8)

	for i in range(400):
		for j in range(20):
			for k in range(20):
				pic[20 * curve[i][0] + j][20 * curve[i][1] + k][0] = int(i / 400 * 255 * 1.2)
				pic[20 * curve[i][0] + j][20 * curve[i][1] + k][1] = int(i / 500 * 255 * 1)
				pic[20 * curve[i][0] + j][20 * curve[i][1] + k][2] = int(i / 600 * 255 * 0.8)

	new_im = Image.fromarray(pic)
	new_im.save("../Misc/curve.png")'''

	idx2numpy.convert_to_file("../MNIST/curve", curve)