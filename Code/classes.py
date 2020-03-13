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

	def start(self, n = 1):
		Q = {(i, j): self.coef[i][j] for i in self.ind for j in self.ind}
		self.response = self.sampler.sample_qubo(Q, num_reads=n)



a = QBM(10,10)











