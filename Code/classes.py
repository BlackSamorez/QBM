import pandas as pd
import idx2numpy
import numpy as np
from math import *
from dwave.system import DWaveSampler, EmbeddingComposite



class QBM:
	def __init__(self, hidlen, vislen):
		self.hidlen = hidlen
		self.vislen = vislen
		self.hind = ['h' + str(i) for i in range(self.hidlen)]
		self.vind = ['v' + str(i) for i in range(self.vislen)]
		self.ind = self.hind + self.vind
		self.coef = {x: {y: 0 for y in self.ind} for x in self.ind}
		self.sampler = EmbeddingComposite(DWaveSampler())
		self.response = 0
		self.data = []
		self.datan = 0

	def start(self, n = 1):
		self.datan = n
		Q = {(i, j): self.coef[i][j] for i in self.ind for j in self.ind}

		self.response = self.sampler.sample_qubo(Q, num_reads=n)

	def fetch_data(self):
		self.data = []

		for datum in self.response.data(fields = ['sample', 'energy', 'num_occurrences'], sorted_by = 'energy'):
			sample = datum.sample
			energy = datum.energy
			percentage = round(datum.num_occurrences / self.datan, 4)
			self.data += [{'sample': sample,'energy': energy,'percentage': percentage}]

	def save_data(self, filename = 'last_data'):
		filename = '../Data/' + filename
		data_np = np.empty((len(self.data), 3), dtype = 'object')

		for i in range(len(self.data)):
			index = list(self.data[i]['sample'])
			for key in index:
				data_np[i][0] = key + '=' + str(self.data[i]['sample'][key]) + ','

			data_np[i][0] = self.data[i]['sample']
			data_np[i][1] = self.data[i]['energy']
			data_np[i][2] = self.data[i]['percentage']

		np.savetxt(filename, data_np, fmt='%s')

def test():
	a = QBM(2, 0)
	a.coef['h0']['h0'] = -1
	a.coef['h1']['h1'] = -1
	a.coef['h0']['h1'] = 2
	return a

























