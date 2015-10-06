import numpy as np

from data import Data

import sys

class Perceptron:
    def __init__(self, data):
        self.data = data
        for i,y in enumerate(self.data.labels()):
            if y < 0:
                self.data.data()[i] = -self.data.data()[i]
                self.data.labels()[i] = -self.data.labels()[i]

    def train(self, r=0.001):
        W = np.random.random(len(self.data.data().T))
        total_mistake = 1
        iterations = 0
        while total_mistake > 0:
            iterations += 1
            M = []
            for x in self.data.data():
                if np.dot(x,W) < 0:
                    M.append(x)
            total_mistake = len(M)
            print "Iteration %d, total_mistake %d" % (iterations+1, total_mistake)
            s = 0.0
            for m in M:
                s += m
            W += r*s
        #print iterations 
        print "\nClassifier weights:",W
        print "\nNormalized with threshold:", -W/W[0]

if __name__=="__main__":
    data_file = "../data/perceptron/perceptronData.txt"
    data = Data(data_file)
    p = Perceptron(data)
    p.train(r=float(sys.argv[1]))
