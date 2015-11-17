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
        W_best = None
        mse_best = float("inf")
        W = np.random.random(len(self.X.T))
        for i in range(10000001):
            index = np.random.randint(len(self.X))
            h = np.dot(W, self.X[index])
            d = (h-self.Y[index])*self.X[index]
            W -= r*d
            if i%10000 == 0:
                sse = 0.0
                for t in range(len(self.X)):
                    s = np.dot(W, self.X[t])
                    sse += (s-self.Y[t])**2.0
                mse = sse/float(len(self.X))
                if mse < mse_best:
                    mse_best = mse
                    W_best = W.copy()
                    print i, mse
        return W_best

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

    def logreg_stoch(self, r=0.5, it=0):
        W_best = None
        acc_max = 0
        W = np.random.random(len(self.X.T))
        for i in range(it):
            index = np.random.randint(len(self.X))
            # wx
            s = np.dot(W, self.X[index])
            # h = g(wx)
            g = 1.0 / (1.0 + np.exp(-s))
            # (y - h(x))*x
            d = (self.Y[index] - g)*self.X[index]
            # w := r*(y - h(x))*x
            W += r*d
            if i%1000 == 0:
                acc = self.accuracy(W)
                if acc > acc_max:
                    acc_max = acc
                    W_best = W.copy()
        return W_best

    def logreg_ridge_batch(self, r=0.05, it=0, l=0.1):
        W_best = None
        acc_max = 0
        N = len(self.X)
        W = np.random.random(len(self.X.T))
        for t in range(it):
            total_ = 0.0
            for i in range(N):
                # wx
                s = np.dot(W, self.X[i])
                # h = g(wx)
                g = 1.0 / (1.0 + np.exp(-s))
                # (y - h(x))*x
                d = (g - self.Y[i])*self.X[i]
                # Penalty term
                p = (l/N)*W 
                # Combine terms
                single_ = d - p
                # Add to the total
                total_ += single_
            # Normalize total
            total_ /= N
            # Incorporate learning rate
            W -= r*total_
            # Check every 1k iterations
            if t%10 == 0:
                acc = self.accuracy(W)
                if acc > acc_max:
                    print "Iteration:",t,"Acc:",acc
                    acc_max = acc
                    W_best = W.copy()
        return W_best

    def logreg_ridge_stoch(self, r=0.05, it=0, l=0.05):
        W_best = None
        acc_max = 0
        N = len(self.X)
        W = np.random.random(len(self.X.T))
        for t in range(it):
            i = np.random.randint(len(self.X))
            # wx
            s = np.dot(W, self.X[i])
            # h = g(wx)
            g = 1.0 / (1.0 + np.exp(-s))
            # (y - h(x))*x
            d = (g - self.Y[i])*self.X[i]
            # Penalty term
            p = (l/N)*W 
            # Combine terms
            single_ = d - p
            # Incorporate learning rate
            W -= r*single_
            # Check every 1k iterations
            if t%1000 == 0:
                acc = self.accuracy(W)
                if acc > acc_max:
                    print "Iteration:",t,"Acc:",acc
                    acc_max = acc
                    W_best = W.copy()
        print "Final:",self.accuracy(W_best)
        return W_best
