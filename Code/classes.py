import pandas as pd
import idx2numpy
import numpy as np
from math import *
from dwave.system import DWaveSampler, EmbeddingComposite
import json
import dimod
from random import randrange as rnd,choice



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

		self.np_coef = np.zeros((self.vislen + self.hidlen, self.vislen + self.hidlen), dtype = np.float_)
		self.coef = {x: {y: 0 for y in self.ind} for x in self.ind}	#coefs, question and bqm
		self.Q = {(i, j): self.coef[i][j] for i in self.ind for j in self.ind}
		self.bqm = dimod.BinaryQuadraticModel(self.Q, 'SPIN')

		self.response = 0	#response and data from it
		self.datalen = 0
		self.data = []
		self.data_prob = []
		self.datan = 0
		self.prob_vec = []

		self.image = []	#images from dataset
		self.images = []
		self.label = 0
		self.labels = []

		self.single_unfixed = []
		self.double_unfixed = []
		self.single_fixed = []
		self.double_fixed = []


	def randomise_coefs(self):	#for premature testing
		for i in range(self.vislen + self.hidlen):
			for j in range(self.vislen + self.hidlen):
				self.coef[self.ind[i]][self.ind[j]] = rnd(0,99) / 100

	def read_images(self, train = True):
		if train:
			self.images = idx2numpy.convert_from_file('../MNIST/curved_train_images')
			self.labels = idx2numpy.convert_from_file('../MNIST/train_labels')
		else:
			self.images = idx2numpy.convert_from_file('../MNIST/curved_test_images')
			self.labels = idx2numpy.convert_from_file('../MNIST/test_labels')

	def read_image(self, n):
		self.image = self.images[n]
		self.label = self.labels[n]


	def make_q(self): #question from coefs
		self.Q = {(i, j): self.coef[i][j] for i in self.ind for j in self.ind}

	def make_bqm(self): #bqm from coefs
		self.Q = {(i, j): self.coef[i][j] for i in self.ind for j in self.ind}
		self.bqm = dimod.BinaryQuadraticModel(self.Q, 'SPIN')


	def run(self, n = 1): #run on dwave
		if not self.test:
			self.datan = n
			self.make_q()
			self.response = self.sampler.sample_qubo(self.Q, num_reads=n)
			#self.response = self.response.data(fields = ['sample', 'num_occurrences'], sorted_by = 'sample')
		else:
			print("it's a test, can't run on dwave")

	def sim_run(self, n = 100): #run locally (simulation)
		self.make_bqm()
		self.datan = n
		self.response = dimod.SimulatedAnnealingSampler().sample(self.bqm, num_reads = self.datan)


	def fix_h(self, i, val): #fix hidden layer qubit
		self.bqm.fix_variable(self.hind[i], val)

	def fix_v(self, i, val): #fix visible layer qubit
		self.bqm.fix_variable(self.vind[i], val)

	def fix_image(self): #fix image into bqm
		for i in range(400):
			self.bqm.fix_variable(self.vind[i], self.image[i])

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

	def save_data(self, filename = "test"): #save data as idx
		filename1 = "../Data/" + filename + ".samples"
		filename2 = "../Data/" + filename + ".probs"
		filename3 = "../Data/" + filename + ".lens"

		lens = np.zeros((2), dtype = "int16")
		lens[0] = self.hidlen
		lens[1] = self.vislen

		idx2numpy.convert_to_file(filename1, self.data)
		idx2numpy.convert_to_file(filename2, self.data_prob)
		idx2numpy.convert_to_file(filename3, lens)


	def calc_single_unfixed(self):
		self.single_unfixed = np.zeros(self.hidlen + self.vislen, dtype = np.float_)
		responses = self.data.transpose()

		self.single_unfixed = responses.dot(self.data_prob)

	def calc_double_unfixed(self):
		self.single_unfixed = np.zeros(self.hidlen + self.vislen, dtype = np.float_)
		self.double_unfixed = np.zeros((self.hidlen + self.vislen, self.hidlen + self.vislen), dtype = np.float_)

		for i in range(self.hidlen + self.vislen):
			for j in range(self.hidlen + self.vislen):
				for k in range(self.datalen):
					self.double_unfixed[i][j] += self.data_prob[k] * self.data[k][i] * self.data[k][j]
				if i == j:
					self.single_unfixed[i] = self.double_unfixed[i][j]

	def calc_sigma_v(self, v):
		state = np.zeros((self.vislen + self.hidlen), dtype = np.float_)
		for i in range(len(v)):
			state[i] = v[i]
		b = np.zeros(self.vislen + self.hidlen, dtype = np.float_)
		for i in range(self.vislen + self.hidlen):
			b[i] = self.np_coef[i][i]
		b_eff = b + self.np_coef.dot(state)
		sigma_v = np.tanh(b_eff)
		return sigma_v

	def calc_single_fixed(self):
		self.single_fixed = np.zeros(self.vislen + self.hidlen, dtype = np.float_)
		schet = 0
		for v in self.images:
			print(schet)
			schet += 1
			self.single_fixed += self.calc_sigma_v(v)
		self.single_fixed = self.single_fixed / len(self.images)

	def calc_double_fixed(self):
		self.double_fixed = np.zeros((410, self.hidlen), dtype = np.float_)
		#hid x hid is already 0
		#vis x vis:
		'''from_data = idx2numpy.convert_from_file('../MNIST/double')
		for i in range(self.vislen):
			for j in range(self.vislen):
				self.double_fixed[self.hidlen + i][self.hidlen + j] = from_data[i][j]'''
		#vis x hid
		schet = 0
		for v in self.images:
			vp = np.ones(410, dtype = np.float_) * -1
			for i in range(400):
				vp[i] = v[i]
			vp[400 + self.labels[schet]] = 1
			print(schet)
			schet += 1
			sigma_v = self.calc_sigma_v(v)
			for i in range(410):
				for j in range(self.hidlen):
					self.double_fixed[i][j] +=  vp[i] * sigma_v[j]









def test():
	a = QBM(100, 410, True)
	for i in range(510):
		for j in range(510):
			a.np_coef[i][j] = rnd(1, 100)
			if i > j:
				a.np_coef[i][j] = 0
	for i in range(100):
		for j in range(100):
			if i != j:
				a.np_coef[i][j] = 0
	a.read_images()
	return a





















