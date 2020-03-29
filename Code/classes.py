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
		self.datan = 0

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
		else:
			print("it's a test, can't run on dwave")

	def sim_run(self, n = 100): #run locally (simulation)
		self.datan = n
		self.response = dimod.SimulatedAnnealingSampler().sample(self.bqm, num_reads = self.datan)


	def fix_h(self, i, val): #fix hidden layer qubit
		self.bqm.fix_variable(self.hind[i], val)

	def fix_v(self, i, val): #fix visible layer qubit
		self.bqm.fix_variable(self.vind[i], val)

	def fix_vis(self): #fix image into qbm
		for i in range(self.vislen - 10):
			self.bqm.fix_variable(self.vind[i], self.image[self.curve[i][0]][self.curve[i][1]])


	def fetch_data(self):	#reading response
		self.data = []

		for datum in self.response.data(fields = ['sample', 'energy', 'num_occurrences'], sorted_by = 'energy'):
			sample = datum.sample
			energy = datum.energy
			percentage = round(datum.num_occurrences / self.datan * 100, 1)
			for i in sample:
				sample[i] = bool(sample[i])
			self.data += [{'sample': sample,'energy': energy,'percentage': percentage}]
			self.datalen = len(self.data)

	def save_data_txt(self, filename = 'last_data'):	#saving response as txt
		filename = '../Data/' + filename
		data_np = np.empty((self.datalen, 3), dtype = 'object')

		for i in range(self.datalen):
			index = list(self.data[i]['sample'])
			for key in index:
				data_np[i][0] = key + '=' + str(self.data[i]['sample'][key]) + ','

			data_np[i][0] = self.data[i]['sample']
			data_np[i][1] = self.data[i]['energy']
			data_np[i][2] = self.data[i]['percentage']

		np.savetxt(filename + '.txt', data_np, fmt='%s')

	def save_data_idx(self, filename = 'last_data'):	#saving response as idx
		filename = '../Data/' + 'sas'
		dt = np.dtype(bool)
		hid_np = np.empty((self.datalen, self.hidlen), dtype = dt)
		vis_np = np.empty((self.datalen, self.vislen), dtype = dt)
		ind_np = np.empty((self.datalen, 4), dtype = np.float)
		for i in range(self.datalen):
			for ind in range(self.hidlen):
				if self.data[i]['sample'][self.hind[ind]] == 1:
					k = 1
				else:
					k = 0
				hid_np[i][ind] = k
			for ind in range(self.vislen):
				if self.data[i]['sample'][self.vind[ind]] == 1:
					k = 1
				else:
					k = 0
				vis_np[i][ind] = k
			ind_np[i][0] = self.hidlen
			ind_np[i][1] = self.vislen
			ind_np[i][2] = self.data[i]['energy']
			ind_np[i][3] = self.data[i]['percentage']
			if ind_np != []:
				idx2numpy.convert_to_file(filename + '_ind.idx', ind_np)
			if hid_np != []:
				idx2numpy.convert_to_file(filename + '_hid.idx', hid_np)
			if vis_np != []:	
				idx2numpy.convert_to_file(filename + '_vis.idx', vis_np)

	def save_data_json(self, filename = 'last_data'):	#saving response as json
		filename = '../Data/' + filename
		with open(filename + '_json', 'w') as fout:
			json.dump(self.data, fout) 



def test():
	a = QBM(2, 0)
	a.coef['h0']['h0'] = -1
	a.coef['h1']['h1'] = -1
	a.coef['h0']['h1'] = 2
	return a

























