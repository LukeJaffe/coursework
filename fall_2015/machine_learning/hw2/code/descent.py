import numpy as np

from evaluator import Evaluator
from utils import timing

class GradientDescent:
    def __init__(self, data, labels):
        self.X = data
        self.Y = labels

    def accuracy(self, W):
        total = len(self.X)
        correct = 0
        for j in range(total):
            s = np.dot(self.X[j], W)
            if s >= 0.5:
                s = 1.0
            else:
                s = 0.0
            if s == self.Y[j]:
                correct += 1
        acc = float(correct)/float(total)
        return acc

    def batch(self, r=1e-8):
        W = np.random.random(len(self.X.T))
        for i in range(10000):
            s = 0.0
            mse = 0.0
            for t in range(len(self.X)):
                h = np.dot(W, self.X[t])
                s += (h-self.Y[t])*self.X[t]
                mse += (h-self.Y[t])**2.0
            W -= r*s 
            if i%1000 == 0:
                print mse/float(len(self.X))
        return W

    def linreg_stoch1(self, r=1e-6):
        W = np.random.random(len(self.X.T))
        for i in range(20000001):
            index = np.random.randint(len(self.X))
            h = np.dot(W, self.X[index])
            d = (h-self.Y[index])*self.X[index]
            W -= r*d
            if i%100000 == 0:
                sse = 0.0
                for t in range(len(self.X)):
                    s = np.dot(W, self.X[t])
                    sse += (s-self.Y[t])**2.0
                mse = sse/float(len(self.X))
                if mse < 25:
                    print i
                    return W
        return W

    def linreg_stoch2(self, r=0.06):
        W_best = None
        acc_max = 0
        W = np.random.random(len(self.X.T))
        for i in range(100001):
            index = np.random.randint(len(self.X))
            h = np.dot(W, self.X[index])
            d = (h-self.Y[index])*self.X[index]
            W -= r*d
            if i%1000 == 0:
                acc = self.accuracy(W)
                if acc > acc_max:
                    #print acc
                    acc_max = acc
                    W_best = W.copy()
        return W_best

    def logreg_stoch(self, r=0.5):
        W_best = None
        acc_max = 0
        W = np.random.random(len(self.X.T))
        for i in range(100001):
            index = np.random.randint(len(self.X))
            s = np.dot(W, self.X[index])
            g = 1.0 / (1.0 + np.exp(-s))
            d = (self.Y[index] - g)*self.X[index]
            W += r*d
            if i%1000 == 0:
                acc = self.accuracy(W)
                if acc > acc_max:
                    #print acc
                    acc_max = acc
                    W_best = W.copy()
        return W_best
