import pandas as pd
import idx2numpy
import numpy as np
from math import *
from dwave.system import LeapHybridSampler, EmbeddingComposite, DWaveSampler, AutoEmbeddingComposite
import json
import dimod
import random
import hybrid
from neal import SimulatedAnnealingSampler


class QBM:
    def __init__(self, hidlen, vislen, sampler = "Test"):
        self.sep = 0
        if sampler == "LeapHybridSampler":
            self.sampler = LeapHybridSampler()   #accessing dwave
            self.sep = 1
        if sampler == "DWaveSampler":
            self.sampler = AutoEmbeddingComposite(DWaveSampler())
        if sampler == "Test":
            self.sampler = SimulatedAnnealingSampler()
        self.t_step = 1000
        self.stepsize = 0.1
        self.path = ""

        self.hidlen = hidlen    #handling indexes
        self.vislen = vislen
        self.outlen = 10
        self.hind = ['h' + str(i) for i in range(self.hidlen)]
        self.vind = ['v' + str(i) for i in range(self.vislen)]
        self.ind = self.hind + self.vind

        self.cmap = []
        self.coef = np.zeros((self.vislen + self.hidlen, self.vislen + self.hidlen), dtype = np.float_)
        #self.Q = {(i, j): self.coef[i][j] for i in self.ind for j in self.ind}
        self.bqm = dimod.BinaryQuadraticModel(self.coef, 'SPIN')

        self.response = 0    #response and data from it
        self.datalen = 0
        self.data = []
        self.index = []
        for i in range(self.hidlen + self.vislen):
            self.index += [i]
        self.data_occ = []
        self.datan = 0
        self.prob = []
        self.top = [0] * 3
        self.expected = 0

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
        self.mean_single = []


    def randomize_coef(self):    #for premature testing
        self.read_cmap()
        for pair in self.cmap:
            self.coef[pair[0]][pair[1]] = random.randrange(200) / 100 - 1


    def read_data(self, train = True):
        prop = idx2numpy.convert_from_file(self.path + "Data/prop")
        self.hidlen = prop[0]
        self.vislen = prop[1]
        self.outlen = prop[2]
        self.coef = np.zeros((self.vislen + self.hidlen, self.vislen + self.hidlen), dtype = np.float_)
        self.make_bqm()
        self.read_cmap()
        if train:
            self.mean_single = idx2numpy.convert_from_file(self.path + 'Data/single')
            self.images = idx2numpy.convert_from_file(self.path + 'Data/train')
        else:
            self.images = idx2numpy.convert_from_file(self.path + 'Data/test')


    def read_image(self, n):
        self.image = self.images[n]

    '''def read_tests(self):
        self.mean_single = idx2numpy.convert_from_file("../Data/tests/single")
        self.images = idx2numpy.convert_from_file("../Data/tests/data")
        self.chosen_images = self.images'''

    def read_coef(self, filename = "last"):
        filename = self.path + "Coef/" + filename + ".coef"
        self.coef = idx2numpy.convert_from_file(filename)

    def save_coef(self, filename = "last"):
        filename = self.path + "Coef/" + filename + ".coef"
        idx2numpy.convert_to_file(filename, self.coef)

    def read_cmap(self):
        self.cmap = idx2numpy.convert_from_file(self.path + "Data/cmap")


    #def make_q(self): #question from coefs
        #self.Q = {(i, j): self.coef[i][j] for i in self.ind for j in self.ind}

    def make_bqm(self): #bqm from coefs
        #self.make_q()
        self.bqm = dimod.BinaryQuadraticModel(self.coef, 'SPIN')


    def run(self, n = 1): #run on dwave
        self.datan += n
        self.make_bqm()
        self.response = self.sampler.sample(self.bqm, num_reads = n)

    def run_test(self, n = 1):
        self.datan += n
        self.response = self.sampler.sample(self.bqm, num_reads = n)
    '''def sim_run(self, n = 100): #run locally (simulation)
        self.make_bqm()
        self.datan += n
        self.response = dimod.SimulatedAnnealingSampler().sample(self.bqm, num_reads = self.datan)'''


    def fix_h(self, i, val): #fix hidden layer qubit
        self.bqm.fix_variable(self.hind[i], val)

    def fix_v(self, i, val): #fix visible layer qubit
        self.bqm.fix_variable(self.vind[i], val)

    def fix_image(self): #fix image into bqm
        for i in range(self.vislen - self.outlen):
            self.bqm.fix_variable(i + self.hidlen, self.image[i])
        self.index = []
        for i in range(self.hidlen):
            self.index += [i]
        for i in range(self.outlen):
            self.index += [self.hidlen + self.vislen - self.outlen + i]

    '''def fix_output(self, n): #fix output (0-9)
        for i in range(self.outlen):
            if i == n:
                self.bqm.fix_variable(self.vind[-(self.outlen - i)], 1)
            else:
                self.bqm.fix_variable(self.vind[-(self.outlen - i)], 0)'''


    def fetch_answer(self):    #reading response
        self.datalen = 0
        for datum in self.response.data(fields = ['sample', 'energy', 'num_occurrences'], sorted_by = 'energy'):
            self.datalen += 1
        self.data = np.zeros((self.datalen, self.hidlen + self.vislen), dtype = "int8")
        self.data_occ = np.zeros(self.datalen, dtype = np.float_)
        
        i = 0
        for datum in self.response.data(fields = ['sample', 'energy', 'num_occurrences'], sorted_by = 'energy'):
            self.data_occ[i] = datum.num_occurrences
            for j in self.index:
                self.data[i][j] = datum.sample[j]
            i += 1

    def add_answer(self):
        for datum in self.response.data(fields = ['sample', 'energy', 'num_occurrences'], sorted_by = 'energy'):
            self.datalen += 1
        for datum in self.response.data(fields = ['sample', 'energy', 'num_occurrences'], sorted_by = 'energy'):
            self.data_occ = np.append(self.data_occ, datum.num_occurrences)
            ent = np.zeros((1, self.hidlen + self.vislen), dtype = np.int32)
            for i in self.index:
                ent[0][i] = datum.sample[i]
            self.data = np.concatenate((self.data, ent), axis = 0)

    def save_answer(self, number = "test"): #save data as idx
        filename1 = self.path + "Resp/" + number + ".samples"
        filename2 = self.path + "Resp/" + number + ".occs"
        filename3 = self.path + "Resp/" + number + ".lens"

        lens = np.zeros((2), dtype = "int16")
        lens[0] = self.hidlen
        lens[1] = self.vislen

        idx2numpy.convert_to_file(filename1, self.data)
        idx2numpy.convert_to_file(filename2, self.data_occ)
        idx2numpy.convert_to_file(filename3, lens)

    def read_answer(self, number = "test"):
        filename1 = self.path + "Resp/" + number + ".samples"
        filename2 = self.path + "Resp/" + number + ".occs"
        '''filename3 = self.path + "Resp/" + number + ".lens"

        lens = np.zeros((2), dtype = "int16")
        lens[0] = self.hidlen
        lens[1] = self.vislen'''

        self.data = idx2numpy.convert_from_file(filename1)
        self.data_occ = idx2numpy.convert_from_file(filename2)

    def analyze_answer(self):
        one_count = 0
        self.prob = np.zeros((self.outlen), dtype = np.float_)
        for instance in self.data:
            for out in range(self.outlen):
                if (instance[self.hidlen + self.vislen - self.outlen + out] == 1):
                    self.prob[out] += 1
                    one_count += 1
        self.prob = self.prob / one_count
        lp = 0
        top_prob = [0] * 3
        self.top = [0] * 3
        for i in range(self.outlen):
            if self.prob[i] > top_prob[0]:
                self.top[2] = self.top[1]
                top_prob[2] = top_prob[1]
                self.top[1] = self.top[0]
                top_prob[1] = top_prob[0]
                self.top[0] = i + 1
                top_prob[0] = self.prob[i]
            else:
                if self.prob[i] > top_prob[1]:
                    self.top[2] = self.top[1]
                    top_prob[2] = top_prob[1]
                    self.top[1] = i + 1
                    top_prob[1] = self.prob[i]
                else:
                    if self.prob[i] > top_prob[2]:
                        self.top[2] = i + 1
                        top_prob[2] = self.prob[i]
        for i in range(self.outlen):
            if self.image[-i - 1] == 1:
                self.expected = i
        self.expected = 26 - self.expected


    def calc_single_unfixed(self):
        print("Response processing has begun:")
        self.single_unfixed = np.zeros(self.hidlen + self.vislen, dtype = np.float_)
        responses = self.data.transpose()
        self.single_unfixed = responses.dot(self.data_occ / self.datan)
        print("Response processing has finished")

    def calc_double_unfixed(self):
        self.calc_single_unfixed()
        self.double_unfixed = np.zeros((self.hidlen + self.vislen, self.hidlen + self.vislen), dtype = np.float_)

        for pair in self.cmap:
            for k in range(self.datalen):
                self.double_unfixed[pair[0]][pair[1]] += self.data_occ[k] / self.datan * self.data[k][pair[0]] * self.data[k][pair[1]]
            if pair[0] == pair[1]:
                self.double_unfixed[pair[0]][pair[0]] = self.single_unfixed[pair[0]]

    def choose_images(self, n = 10000):
        self.chosen_images = []
        numbers = random.sample(range(len(self.images)), n)
        for i in numbers:
            self.chosen_images += [self.images[i]]    

    def calc_sigma_v(self, v, th = True):
        state = np.zeros((self.vislen + self.hidlen), dtype = np.float_)
        for i in range(self.vislen):
            state[self.hidlen + i] = v[i]
        b = np.zeros(self.vislen + self.hidlen, dtype = np.float_)
        b_eff = self.coef.dot(state)
        for i in range(self.hidlen):
            b_eff[i] += self.coef[i][i]
        if th:
            b_eff = np.tanh(b_eff)
        return b_eff

    def calc_single_fixed(self):
        self.single_fixed = np.zeros(self.vislen + self.hidlen, dtype = np.float_)
        for v in self.chosen_images:
            self.single_fixed += self.calc_sigma_v(v)
        self.single_fixed = self.single_fixed / len(self.chosen_images)


    def calc_double_fixed(self):
        print("Dataset processing has begun:")
        self.double_fixed = np.zeros((self.hidlen + self.vislen, self.hidlen + self.vislen), dtype = np.float_)
        temp = np.zeros((self.hidlen + self.vislen, self.hidlen + self.vislen), dtype = np.float_)
        schet = 0
        for v in self.chosen_images:
            vp = np.ones(self.vislen, dtype = np.float_) * -1
            for i in range(self.vislen):
                vp[i] = v[i]
            if schet % 4000 == 0:
                print("Image: ", schet, " out of ", len(self.chosen_images) , " processed")
            schet += 1
            sigma_v = self.calc_sigma_v(v)
            for i in range(self.hidlen):
                for j in range(self.vislen):
                    temp[i][self.hidlen + j] += sigma_v[i] * vp[j]

        for i in range(self.hidlen + self.vislen):
            for j in range(self.hidlen + self.vislen):
                temp[i][j] = temp[i][j] / len(self.chosen_images)
        self.calc_single_fixed()
        for i in range(self.hidlen):
            temp[i][i] = self.single_fixed[i]
        for i in range(self.vislen):
        	temp[self.hidlen + i][self.hidlen + i] = self.mean_single[i]
        for pair in self.cmap:
            self.double_unfixed[pair[0]][pair[1]] = temp[pair[0]][pair[1]]
        print("Dataset processing has finished")

    def change_coef(self):
        self.delta = (self.double_fixed - self.double_unfixed) * self.stepsize
        self.coef = self.coef + self.delta


    def make_step(self, step):
        print("Step " + str(step) + " began!")
        self.read_coef(str(step))
        self.make_bqm()
        if not self.sep:
            self.run(self.t_step)
            self.fetch_answer()
            print("Got full respose, n = ", self.t_step)
        else:
            self.run(1)
            self.fetch_answer()
            for i in range(self.t_step - 1):
                self.run(1)
                self.add_answer()
                if (i + 2) % 10 == 0:
                    print("Got respose ", i + 2, " out of", self.t_step)
        self.save_answer(str(step))
        self.calc_double_unfixed()
        self.choose_images(len(self.images))
        self.calc_double_fixed()
        self.change_coef()
        self.save_coef(str(int(step) + 1))
        print("Step " + str(step) + " complete!")

    def make_steps(self, n, stepsize = 0.1):
        self.stepsize = stepsize
        starting_step = idx2numpy.convert_from_file(self.path + "current_step")
        new_step = np.zeros((1), dtype = np.int32)
        for i in range(n):
            step = starting_step[0] + i
            self.make_step(step)
            new_step[0] = step
            idx2numpy.convert_to_file(self.path + "current_step", new_step)

    def calc_pv(self, v):
        b_eff = self.calc_sigma_v(v, False)
        Energy = 0
        for i in range(self.vislen):
            Energy += v[i] * self.coef[self.hidlen + i][self.hidlen + i]
        pv = e ** (-Energy)
        for i in range(self.hidlen):
            pv = pv * cosh(b_eff[i])
        return pv

    def calc_div(self):
        pvs = [0] * len(self.images)
        pvsum = 0
        pds = [1 / len(self.images)] * len(self.images)
        i = 0
        for image in self.images:
            pvs[i] = self.calc_pv(image)
            pvsum += pvs[i]
            i += 1
            if i % 2000 == 0:
                print(i, "th picture")
        pvs = pvs / pvsum
        div = 0
        i = 0
        for image in self.images:
            div += pds[i] * log(pds[i] / pvs[i])
            i += 1
        return div
    
    def save_div(self, steps):
        div = np.zeros((len(steps)), dtype = np.float_)
        for i in steps:
            self.read_coef(str(i))
            div[i] = self.calc_div()
            print("Calculated ", i, "'th divergence!")
        idx2numpy.convert_to_file(path + "Results/div")
