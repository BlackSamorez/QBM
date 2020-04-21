import pandas as pd
import idx2numpy
import numpy as np
from math import *
from dwave.system import LeapHybridSampler, EmbeddingComposite, DWaveSampler
import json
import dimod
import random




class QBM:
    def __init__(self, hidlen, vislen, test = False):
        self.test = test
        if not self.test:
            self.sampler = LeapHybridSampler()   #accessing dwave
        self.t_step = 20
        self.stepsize = 0.1

        self.hidlen = hidlen    #handling indexes
        self.vislen = vislen
        self.outlen = 10
        self.hind = ['h' + str(i) for i in range(self.hidlen)]
        self.vind = ['v' + str(i) for i in range(self.vislen)]
        self.ind = self.hind + self.vind

        self.coef = np.zeros((self.vislen + self.hidlen, self.vislen + self.hidlen), dtype = np.float_)
        #self.Q = {(i, j): self.coef[i][j] for i in self.ind for j in self.ind}
        self.bqm = dimod.BinaryQuadraticModel(self.coef, 'SPIN')

        self.response = 0    #response and data from it
        self.datalen = 0
        self.data = []
        self.data_prob = []
        self.datan = 0
        self.prob_vec = []

        self.image = []    #images from dataset
        self.images = []
        self.label = 0
        self.labels = []
        self.chosen_images = []

        self.single_unfixed = []
        self.double_unfixed = []
        self.single_fixed = []
        self.double_fixed = []
        self.delta = []


    def randomise_coef(self):    #for premature testing
        for i in range(self.vislen + self.hidlen):
            for j in range(self.vislen + self.hidlen):
                self.coef[i][j] = random.randrange(200) / 100 - 1
                if i > j :
                    self.coef[i][j] = 0
        for i in range(self.hidlen):
            for j in range(self.hidlen):
                if i != j:
                    self.coef[i][j] = 0
        for i in range(self.vislen):
            for j in range(self.vislen):
                self.coef[self.hidlen + i][self.hidlen + j] = 0


    def read_images(self, train = True):
        if train:
            self.images = idx2numpy.convert_from_file('../MNIST/complete_train_images')
            self.labels = idx2numpy.convert_from_file('../MNIST/train_labels')
        else:
            self.images = idx2numpy.convert_from_file('../MNIST/complete_test_images')
            self.labels = idx2numpy.convert_from_file('../MNIST/test_labels')

    def read_image(self, n):
        self.image = self.images[n]
        self.label = self.labels[n]

    def read_tests(self):
        self.images = idx2numpy.convert_from_file("../Data/tests/data")
        self.chosen_images = self.images

    def read_coef(self, filename = "last"):
        filename = "../Data/coef/" + filename + ".coef"
        self.coef = idx2numpy.convert_from_file(filename)

    def save_coef(self, filename = "last"):
        filename = "../Data/coef/" + filename + ".coef"
        idx2numpy.convert_to_file(filename, self.coef)


    #def make_q(self): #question from coefs
        #self.Q = {(i, j): self.coef[i][j] for i in self.ind for j in self.ind}

    def make_bqm(self): #bqm from coefs
        #self.make_q()
        self.bqm = dimod.BinaryQuadraticModel(self.coef, 'SPIN')


    def run(self, n = 1): #run on dwave
        if not self.test:
            self.datan = n
            self.make_bqm()
            self.response = self.sampler.sample(self.bqm, num_reads = n)
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
        for i in range(self.vislen - self.outlen):
            self.bqm.fix_variable(i + self.hidlen, self.image[i])

    '''def fix_output(self, n): #fix output (0-9)
        for i in range(self.outlen):
            if i == n:
                self.bqm.fix_variable(self.vind[-(self.outlen - i)], 1)
            else:
                self.bqm.fix_variable(self.vind[-(self.outlen - i)], 0)'''


    def fetch_data(self):    #reading response
        self.datalen = 0
        for datum in self.response.data(fields = ['sample', 'energy', 'num_occurrences'], sorted_by = 'energy'):
            self.datalen += 1
        self.data = np.zeros((self.datalen, self.hidlen + self.vislen), dtype = "int8")
        self.data_prob = np.zeros(self.datalen, dtype = np.float_)
        
        i = 0
        for datum in self.response.data(fields = ['sample', 'energy', 'num_occurrences'], sorted_by = 'energy'):
            self.data_prob[i] = datum.num_occurrences / self.datan
            for j in range(self.hidlen + self.vislen):
                self.data[i][j] = datum.sample[j]
            i += 1

    def save_data(self, filename = "test"): #save data as idx
        filename1 = "../Data/answer/" + filename + ".samples"
        filename2 = "../Data/answer/" + filename + ".probs"
        filename3 = "../Data/answer/" + filename + ".lens"

        lens = np.zeros((2), dtype = "int16")
        lens[0] = self.hidlen
        lens[1] = self.vislen

        idx2numpy.convert_to_file(filename1, self.data)
        idx2numpy.convert_to_file(filename2, self.data_prob)
        idx2numpy.convert_to_file(filename3, lens)

    def read_data(self, filename = "test"):
        filename1 = "../Data/answer/" + filename + ".samples"
        filename2 = "../Data/answer/" + filename + ".probs"
        #filename3 = "../Data/answer/" + filename + ".lens"

        '''lens = np.zeros((2), dtype = "int16")
        lens[0] = self.hidlen
        lens[1] = self.vislen'''

        self.data = idx2numpy.convert_from_file(filename1)
        self.data_prob = idx2numpy.convert_from_file(filename2)

    def calc_single_unfixed(self):
        self.single_unfixed = np.zeros(self.hidlen + self.vislen, dtype = np.float_)
        responses = self.data.transpose()

        self.single_unfixed = responses.dot(self.data_prob)

    def calc_double_unfixed(self):
        self.calc_single_unfixed()
        self.double_unfixed = np.zeros((self.hidlen + self.vislen, self.hidlen + self.vislen), dtype = np.float_)

        for i in range(self.hidlen + self.vislen):
            for j in range(self.hidlen + self.vislen):
                if (i == j or (i < self.hidlen and j >= self.hidlen)):
                    for k in range(self.datalen):
                        self.double_unfixed[i][j] += self.data_prob[k] * self.data[k][i] * self.data[k][j]
                    if i == j:
                        self.double_unfixed[i][j] = self.single_unfixed[i]

    def choose_images(self, n = 1000):
        self.chosen_images = []
        numbers = random.sample(range(60000), n)
        for i in numbers:
            self.chosen_images += [self.images[i]]    

    def calc_sigma_v(self, v):
        state = np.zeros((self.vislen + self.hidlen), dtype = np.float_)
        for i in range(self.vislen):
            state[self.hidlen + i] = v[i]
        b = np.zeros(self.vislen + self.hidlen, dtype = np.float_)
        b_eff = self.coef.dot(state)
        sigma_v = np.tanh(b_eff)
        return sigma_v

    def calc_single_fixed(self):
        self.single_fixed = np.zeros(self.vislen + self.hidlen, dtype = np.float_)
        for v in self.chosen_images:
            self.single_fixed += self.calc_sigma_v(v)
        self.single_fixed = self.single_fixed / len(self.chosen_images)

    def calc_double_fixed(self):
        self.double_fixed = np.zeros((self.hidlen + self.vislen, self.hidlen + self.vislen), dtype = np.float_)
        schet = 0
        for v in self.chosen_images:
            vp = np.ones(self.vislen, dtype = np.float_) * -1
            for i in range(self.vislen):
                vp[i] = v[i]
            if (schet % 100) == 0:
                print("Image: ", schet)
            schet += 1
            sigma_v = self.calc_sigma_v(v)
            for i in range(self.hidlen):
                for j in range(self.vislen):
                    self.double_fixed[i][self.hidlen + j] += sigma_v[i] * vp[j]

        for i in range(self.hidlen + self.vislen):
            for j in range(self.hidlen + self.vislen):
                self.double_fixed[i][j] = self.double_fixed[i][j] / len(self.chosen_images)
        self.calc_single_fixed()
        for i in range(self.hidlen + self.vislen):
            self.double_fixed[i][i] = self.single_fixed[i]

    def change_coef(self):
        self.delta = (self.double_fixed - self.double_unfixed) * self.stepsize
        self.coef = self.coef - self.delta


    def make_step(self, step):
        self.read_coef(str(step))
        self.make_bqm()
        self.run(self.t_step)
        self.fetch_data()
        self.save_data(str(step))
        self.calc_double_unfixed()
        self.choose_images()
        self.calc_double_fixed()
        self.change_coef()
        self.save_coef(str(int(step) + 1))

    def make_steps(self, n, stepsize = 0.1):
        self.stepsize = stepsize
        starting_step = idx2numpy.convert_from_file("../Data/current_step")
        for i in range(n):
            step = starting_step[0] + i
            self.make_step(step)
            print("Step " + str(step) + " complete!")
        new_step = np.zeros((1), dtype = np.int32)
        new_step[0] = starting_step[0] + n
        idx2numpy.convert_to_file("../Data/current_step", new_step)


