import pandas as pd
import idx2numpy
import numpy as np
from math import *
from dwave.system import DWaveSampler, EmbeddingComposite
import json



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
		self.datalen = 0
		self.data = []
		self.datan = 0

	def run(self, n = 1):
		self.datan = n
		Q = {(i, j): self.coef[i][j] for i in self.ind for j in self.ind}

		self.response = self.sampler.sample_qubo(Q, num_reads=n)

	def fetch_data(self):
		self.data = []

		for datum in self.response.data(fields = ['sample', 'energy', 'num_occurrences'], sorted_by = 'energy'):
			sample = datum.sample
			energy = datum.energy
			percentage = round(datum.num_occurrences / self.datan * 100, 1)
			for i in sample:
				sample[i] = bool(sample[i])
			self.data += [{'sample': sample,'energy': energy,'percentage': percentage}]
			self.datalen = len(self.data)

	def save_data_txt(self, filename = 'last_data'):
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

	def save_data_idx(self, filename = 'last_data'):
		filename = '../Data/' + 'sas'
		dt = np.dtype(bool_)
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

	def save_data_json(self, filename = 'last_data'):
		filename = '../Data/' + filename
		with open(filename + '_json', 'w') as fout:
			json.dump(self.data, fout) 



def test():
	a = QBM(2, 0)
	a.coef['h0']['h0'] = -1
	a.coef['h1']['h1'] = -1
	a.coef['h0']['h1'] = 2
	return a

























