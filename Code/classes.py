import pandas as pd
import idx2numpy
import numpy as np
import matplotlib as plt
from math import *
from dwave.system import LeapHybridSampler, EmbeddingComposite, DWaveSampler, AutoEmbeddingComposite
import json
import dimod
import random
import hybrid
from neal import SimulatedAnnealingSampler
import matplotlib.pyplot as plt


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
        self.mute = True

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
        self.answerlen = 0
        self.answer = []
        self.index = []
        for i in range(self.hidlen + self.vislen):
            self.index += [i]
        self.answer_occ = []
        self.answern = 0
        self.prob = []
        self.top = [0] * 3
        self.expected = 0

        self.image = []    #images from dataset
        self.images = []
        self.label = 0
        self.labels = []
        self.chosen_images = []
        self.prob = []
        self.chosen_prob = []

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
            self.data_prob = idx2numpy.convert_from_file(self.path + 'Data/prob')
        else:
            self.images = idx2numpy.convert_from_file(self.path + 'Data/test')
        for i in range(self.hidlen + self.vislen):
            self.index += [i]


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
        self.answern += n
        self.make_bqm()
        self.response = self.sampler.sample(self.bqm, num_reads = n)

    def run_test(self, n = 1):
        self.answern += n
        self.response = self.sampler.sample(self.bqm, num_reads = n)
    '''def sim_run(self, n = 100): #run locally (simulation)
        self.make_bqm()
        self.answern += n
        self.response = dimod.SimulatedAnnealingSampler().sample(self.bqm, num_reads = self.answern)'''


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
        self.answerlen = 0
        for datum in self.response.data(fields = ['sample', 'energy', 'num_occurrences'], sorted_by = 'energy'):
            self.answerlen += 1
        self.answer = np.zeros((self.answerlen, self.hidlen + self.vislen), dtype = "int8")
        self.answer_occ = np.zeros(self.answerlen, dtype = np.int32)
        
        i = 0
        for datum in self.response.data(fields = ['sample', 'energy', 'num_occurrences'], sorted_by = 'energy'):
            self.answer_occ[i] = datum.num_occurrences
            for j in self.index:
                self.answer[i][j] = datum.sample[j]
            i += 1

    def add_answer(self):
        for datum in self.response.data(fields = ['sample', 'energy', 'num_occurrences'], sorted_by = 'energy'):
            self.answerlen += 1
        for datum in self.response.data(fields = ['sample', 'energy', 'num_occurrences'], sorted_by = 'energy'):
            self.answer_occ = np.append(self.answer_occ, datum.num_occurrences)
            ent = np.zeros((1, self.hidlen + self.vislen), dtype = np.int32)
            for i in self.index:
                ent[0][i] = datum.sample[i]
            self.answer = np.concatenate((self.answer, ent), axis = 0)

    def save_answer(self, number = "test"): #save data as idx
        filename1 = self.path + "Resp/" + number + ".samples"
        filename2 = self.path + "Resp/" + number + ".occs"
        filename3 = self.path + "Resp/" + number + ".lens"

        lens = np.zeros((2), dtype = "int16")
        lens[0] = self.hidlen
        lens[1] = self.vislen

        idx2numpy.convert_to_file(filename1, self.answer)
        idx2numpy.convert_to_file(filename2, self.answer_occ)
        idx2numpy.convert_to_file(filename3, lens)

    def read_answer(self, number = "test"):
        filename1 = self.path + "Resp/" + number + ".samples"
        filename2 = self.path + "Resp/" + number + ".occs"
        '''filename3 = self.path + "Resp/" + number + ".lens"

        lens = np.zeros((2), dtype = "int16")
        lens[0] = self.hidlen
        lens[1] = self.vislen'''

        self.answer = idx2numpy.convert_from_file(filename1)
        self.answer_occ = idx2numpy.convert_from_file(filename2)

    def prepare_answer(self):
        self.prob = np.zeros((self.answerlen), dtype = np.float_)
        for ans in range(self.answerlen):
            for pair in self.cmap:
                if pair[0] != pair[1]:
                    self.prob[ans] += self.coef[pair[0]][pair[1]] * self.answer[ans][pair[0]] * self.answer[ans][pair[1]]
                else:
                    self.prob[ans] += self.coef[pair[0]][pair[0]] * self.answer[ans][pair[0]]
        self.prob = e ** (-self.prob)
        self.prob = self.prob / np.sum(self.prob)


    def calc_single_unfixed(self):
        if not self.mute:
            print("Response processing has begun:")
        self.single_unfixed = np.zeros(self.hidlen + self.vislen, dtype = np.float_)
        responses = self.answer.transpose()
        self.single_unfixed = responses.dot(self.prob)
        if not self.mute:
            print("Response processing has finished")

    def calc_double_unfixed(self):
        self.calc_single_unfixed()
        self.double_unfixed = np.zeros((self.hidlen + self.vislen, self.hidlen + self.vislen), dtype = np.float_)

        for pair in self.cmap:
            for k in range(self.answerlen):
                self.double_unfixed[pair[0]][pair[1]] += self.prob[k] * self.answer[k][pair[0]] * self.answer[k][pair[1]]
            if pair[0] == pair[1]:
                self.double_unfixed[pair[0]][pair[0]] = self.single_unfixed[pair[0]]

    def choose_images(self, n = "who gives a fuck"):
        self.chosen_images = []
        self.chosen_prob = self.data_prob
        n = len(self.images)
        numbers = random.sample(range(len(self.images)), n)
        for i in numbers:
            self.chosen_images += [self.images[i]]    

    def calc_sigma_v(self, v, th = True):
        state = np.ones((self.vislen + self.hidlen), dtype = np.float_)
        for i in range(self.vislen):
            state[self.hidlen + i] = v[i]
        b_eff = np.zeros(self.vislen + self.hidlen, dtype = np.float_)
        b_eff = self.coef.dot(state)
        if th:
            b_eff = np.tanh(b_eff)
        return b_eff

    def calc_single_fixed(self):
        self.single_fixed = np.zeros(self.vislen + self.hidlen, dtype = np.float_)
        for n in range(len(self.chosen_images)):
            self.single_fixed += self.calc_sigma_v(self.chosen_images[n]) * self.chosen_prob[n]

    def calc_double_fixed(self):
        if not self.mute:
            print("Dataset processing has begun:")
        self.double_fixed = np.zeros((self.hidlen + self.vislen, self.hidlen + self.vislen), dtype = np.float_)
        temp = np.zeros((self.hidlen + self.vislen, self.hidlen + self.vislen), dtype = np.float_)
        image_number = 0
        for v in self.chosen_images:
            vp = np.ones(self.vislen, dtype = np.float_) * -1
            for i in range(self.vislen):
                vp[i] = v[i]
            if image_number % 4000 == 0:
                if not self.mute:
                    print("Image: ", image_number, " out of ", len(self.chosen_images) , " processed")
            sigma_v = self.calc_sigma_v(v)
            for i in range(self.hidlen):
                for j in range(self.vislen):
                    temp[i][self.hidlen + j] += sigma_v[i] * vp[j] * self.chosen_prob[image_number]
            image_number += 1

        self.calc_single_fixed()
        for i in range(self.hidlen):
            temp[i][i] = self.single_fixed[i]
        for i in range(self.vislen):
        	temp[self.hidlen + i][self.hidlen + i] = self.mean_single[i]
        for pair in self.cmap:
            self.double_fixed[pair[0]][pair[1]] = temp[pair[0]][pair[1]]
        if not self.mute:
            print("Dataset processing has finished")

    def change_coef(self):
        self.delta = (self.double_fixed - self.double_unfixed) * self.stepsize
        self.coef = self.coef - self.delta
        '''max_coef = 0
        for pair in self.cmap:
            if abs(self.coef[pair[0]][pair[1]]) > max_coef:
                max_coef = abs(self.coef[pair[0]][pair[1]])
        self.coef = self.coef / max_coef'''


    def make_step(self, step):
        print("Step " + str(step) + " began!")
        self.read_coef(str(step))
        self.make_bqm()
        if not self.sep:
            self.run(self.t_step)
            self.fetch_answer()
            if not self.mute:
                print("Got full respose, n = ", self.t_step)
        else:
            self.run(1)
            self.fetch_answer()
            for i in range(self.t_step - 1):
                self.run(1)
                self.add_answer()
                if (i + 2) % 10 == 0:
                    if not self.mute:
                        print("Got respose ", i + 2, " out of", self.t_step)
        self.save_answer(str(step))
        self.prepare_answer()
        self.calc_double_unfixed()
        self.choose_images(len(self.images))
        self.calc_double_fixed()
        self.change_coef()
        self.save_coef(str(int(step) + 1))
        '''div = self.calc_div()
        print("Calculated ", step, "th divergence:", div)'''
        print("Step " + str(step) + " complete!")

    def make_steps(self, n):
        starting_step = idx2numpy.convert_from_file(self.path + "current_step")
        new_step = np.zeros((1), dtype = np.int32)
        for i in range(n):
            step = starting_step[0] + i
            self.make_step(step)
            new_step[0] = step + 1
            idx2numpy.convert_to_file(self.path + "current_step", new_step)

    def calc_pv(self, v, beff = True):
        b_eff = self.calc_sigma_v(v, False)
        Energy = 0
        for i in range(self.vislen):
            Energy += v[i] * self.coef[self.hidlen + i][self.hidlen + i]
        pv = e ** (-Energy)
        if beff:
            for i in range(self.hidlen):
                pv = pv * cosh(b_eff[i])
        return pv

    def calc_div(self, beff = True):
        pvs = [0] * len(self.images)
        pvsum = 0
        pds = self.data_prob
        i = 0
        for image in self.images:
            pvs[i] = self.calc_pv(image)
            pvsum += self.calc_pv(image, beff)
            i += 1
            if not self.mute:
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
        div = np.zeros((steps), dtype = np.float_)
        for i in range(steps):
            self.read_coef(str(i))
            div[i] = self.calc_div()
            print("Calculated ", i, "th divergence:", div[i])
        idx2numpy.convert_to_file(self.path + "Results/div", div)

    def find_answer(self):
        psum = 0
        prob = np.zeros((self.outlen), dtype = np.float_)
        for i in range(self.outlen):
            vis = np.ones((self.vislen), dtype = np.int8) * -1
            for j in range(self.vislen - self.outlen):
                vis[j] = self.image[j]
            vis[self.vislen - self.outlen + i] = 1
            prob[i] = self.calc_pv(vis)
            psum += prob[i]
        prob = prob / psum
        return prob

    def analyze_answer(self):
        prob = self.find_answer()
        top_prob = [0] * 3
        self.top = [0] * 3
        for i in range(self.outlen):
            if prob[i] >= top_prob[0]:
                self.top[2] = self.top[1]
                top_prob[2] = top_prob[1]
                self.top[1] = self.top[0]
                top_prob[1] = top_prob[0]
                self.top[0] = i
                top_prob[0] = prob[i]
            else:
                if prob[i] >= top_prob[1]:
                    self.top[2] = self.top[1]
                    top_prob[2] = top_prob[1]
                    self.top[1] = i
                    top_prob[1] = prob[i]
                else:
                    if prob[i] >= top_prob[2]:
                        self.top[2] = i
                        top_prob[2] = prob[i]
        for i in range(self.outlen):
            if self.image[-i - 1] == 1:
                self.expected = i + 1
        self.expected = self.outlen - self.expected

