import pandas as pd
import idx2numpy
import numpy as np
from math import *
from dwave.system import DWaveSampler, EmbeddingComposite
import json
import dimod



class QBM:
	def __init__(self, hidlen, vislen, test = False):
		self.test = test
		if not self.test:
			self.sampler = EmbeddingComposite(DWaveSampler())	#accessing dwave

		self.hidlen = hidlen	#handling indexes
		self.vislen = vislen
		self.hind = ['h' + str(i) for i in range(self.hidlen)]
		self.vind = ['v' + str(i) for i in range(self.vislen)]
		self.ind = self.hind + self.vind

		self.coef = {x: {y: 0 for y in self.ind} for x in self.ind}	#coefs, question and bqm
		self.Q = {(i, j): self.coef[i][j] for i in self.ind for j in self.ind}
		self.bqm = dimod.BinaryQuadraticModel(self.Q, 'BINARY')

		self.response = 0	#response and data from it
		self.datalen = 0
		self.data = []
		self.data_prob = []
		self.datan = 0
		self.prob_vec = []

		dt = np.dtype(np.bool)	#image from set and its relation to vis layer coefs (curve)
		self.image = np.empty((20, 20), dtype = dt)
		d = np.dtype(np.uint16)
		self.curve = idx2numpy.convert_from_file('../MNIST/curve')
		self.label = 0


	def read_coef(self, n = 0, filename = 'test'): #not done yet
		filename = '../MNIST' + filename

	def read_image(self, n, train = True):
		if train:
			self.image = idx2numpy.convert_from_file('../MNIST/new_train_images')[n]
			self.label = idx2numpy.convert_from_file('../MNIST/train_labels')[n]
		else:
			self.image = idx2numpy.convert_from_file('../MNIST/new_test_images')[n]
			self.label = idx2numpy.convert_from_file('../MNIST/test_labels')[n]


	def make_q(self): #question from coefs
		self.Q = {(i, j): self.coef[i][j] for i in self.ind for j in self.ind}

	def make_bqm(self): #bqm from coefs
		self.Q = {(i, j): self.coef[i][j] for i in self.ind for j in self.ind}
		self.bqm = dimod.BinaryQuadraticModel(self.Q, 'BINARY')


	def run(self, n = 1): #run on dwave
		if not self.test:
			self.datan = n
			self.Q = {(i, j): self.coef[i][j] for i in self.ind for j in self.ind}
			self.response = self.sampler.sample_qubo(self.Q, num_reads=n)
			#self.response = self.response.data(fields = ['sample', 'num_occurrences'], sorted_by = 'sample')
		else:
			print("it's a test, can't run on dwave")

	def sim_run(self, n = 100): #run locally (simulation)
		self.datan = n
		self.response = dimod.SimulatedAnnealingSampler().sample(self.bqm, num_reads = self.datan)


	def fix_h(self, i, val): #fix hidden layer qubit
		self.bqm.fix_variable(self.hind[i], val)

	def fix_v(self, i, val): #fix visible layer qubit
		self.bqm.fix_variable(self.vind[i], val)

	def fix_image(self): #fix image into bqm
		for i in range(self.vislen - 10):
			self.bqm.fix_variable(self.vind[i], self.image[self.curve[i][0]][self.curve[i][1]])

	def fix_output(self, n): #fix output (0-9)
		for i in range(10):
			if i == n:
				self.bqm.fix_variable(self.vind[-(10 - i)], 1)
			else:
				self.bqm.fix_variable(self.vind[-(10 - i)], 0)


	def fetch_data(self):	#reading response
		self.datalen = 0
		for datum in self.response.data(fields = ['sample', 'energy', 'num_occurrences'], sorted_by = 'energy'):
			self.datalen += 1
		self.data = np.zeros((self.datalen, self.hidlen + self.vislen), dtype = "int8")
		self.data_prob = np.zeros(self.datalen, dtype = np.float_)
		
		i = 0
		for datum in self.response.data(fields = ['sample', 'energy', 'num_occurrences'], sorted_by = 'energy'):
			self.data_prob[i] = datum.num_occurrences / self.datan
			for j in range(self.hidlen + self.vislen):
				self.data[i][j] = datum.sample[self.ind[j]]
			i += 1

	def save_data(self, filename = "test"):
		filename1 = "../Data/" + filename + ".samples"
		filename2 = "../Data/" + filename + ".probs"
		filename3 = "../Data/" + filename + ".lens"

		lens = np.zeros((2), dtype = "int16")
		lens[0] = self.hidlen
		lens[1] = self.vislen

		idx2numpy.convert_to_file(filename1, self.data)
		idx2numpy.convert_to_file(filename2, self.data_prob)
		idx2numpy.convert_to_file(filename3, lens)


def test():
	a = QBM(2, 0, True)
	a.coef['h0']['h0'] = -1
	a.coef['h1']['h1'] = -1
	a.coef['h0']['h1'] = 2
	return a






















