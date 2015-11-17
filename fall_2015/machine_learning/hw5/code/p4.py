import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from evaluator import Evaluator

eps = 1e-5

class Bin:
    def __init__(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval
        self.vals = []

    def check(self, val):
        # There is overlap, need to catch equals case in bounds of range
        if self.minval <= val <= self.maxval:
            return True
        else:
            return False

    def put(self, val, label):
        self.vals.append(np.array([val, label]))


class Histogram:
    def __init__(self, vals):
        self.bins = []
        #if vals[0] < vals[1] < vals[2] < vals[3] < vals[4]:
        #    pass
        #else: print "bad"
        self.N0, self.N1 = 0, 0
        for i in range(len(vals)-1):
            self.bins.append(Bin(vals[i], vals[i+1]))

    def place(self, i, f, Y):
        # Iterate through each datapoint in feature
        for j,x in enumerate(f):
            # Iterate through bins
            for b in self.bins:
                if b.check(x):
                    b.put(x,Y[j])
                    if Y[j] == 0.0:
                        self.N0 += 1
                    else:
                        self.N1 += 1

    def calc(self):
        for b in self.bins: 
            b.vals = np.array(b.vals)
            if len(b.vals):
                a = b.vals.T[1]
                n0 = float(len(a[a==0.0]))
                n1 = float(len(a[a==1.0]))
                b.p0 = n0/float(self.N0)
                b.p1 = n1/float(self.N1)
            else:
                b.p0 = 0.0
                b.p1 = 0.0

    def prob(self, x):
        for b in self.bins:
            if b.check(x):
                return b.p0+eps,b.p1+eps
        else:
            #print "Value doesn't match any bins:",x
            return eps, eps


class NB:
    def __init__(self, train_file, test_file):
        # Get training data
        train_dataset = []
        f = open(train_file, "r")
        for line in f:
            data = [float(x) for x in line.split(",")]
            train_dataset.append(data)
        train_dataset = np.array(train_dataset)
        self.train_data = np.array(train_dataset).T[:-1].T
        self.train_labels = np.array(train_dataset).T[-1:].T
        # Get testing data
        test_dataset = []
        f = open(test_file, "r")
        for line in f:
            data = [float(x) for x in line.split(",")]
            test_dataset.append(data)
        test_dataset = np.array(test_dataset)
        self.test_data = np.array(test_dataset).T[:-1].T
        self.test_labels = np.array(test_dataset).T[-1:].T

    def gaussian(self, X, Y):
        C = []
        C.append(np.array([x[(Y==0.0).T[0]] for x in X.T]).T)
        C.append(np.array([x[(Y==1.0).T[0]] for x in X.T]).T)
        m = len(X)
        d = len(X.T)
        mean, var = [], []
        for i,c in enumerate(C):
            mean.append([]), var.append([])
            for f in c.T:
                mean[i].append(f.mean())
                var[i].append(f.var()+1e-1)
        return mean, var

    def bernoulli(self, X, Y):
        no = np.array([x[(Y==0.0).T[0]] for x in X.T]).T
        yes = np.array([x[(Y==1.0).T[0]] for x in X.T]).T
        N0, N1 = len(no), len(yes)
        mean, e0, e1 = [], [], []
        for i in range(len(X.T)):
            mean.append(np.nanmean(X.T[i]))
            nan0, nan1 = no.T[i], yes.T[i]
            nan0 = nan0[~np.isnan(nan0)]
            nan1 = nan1[~np.isnan(nan1)]
            n0 = len((nan0>mean[i]).nonzero()[0])
            n1 = len((nan1>mean[i]).nonzero()[0])
            e0.append(float(n0)/float(N0))
            e1.append(float(n1)/float(N1))
        return mean, e0, e1

    def hist4(self, X, Y):
        no = np.array([x[(Y==0.0).T[0]] for x in X.T]).T
        yes = np.array([x[(Y==1.0).T[0]] for x in X.T]).T
        histograms = []
        # Calculate bin boundaries
        for i in range(len(X.T)):
            minval = X.T[i].min()
            maxval = X.T[i].max()
            meanval = X.T[i].mean()
            nomean = no.T[i].mean()
            yesmean = yes.T[i].mean()
            # Set up bins
            histograms.append(Histogram([minval, 
                                        min(nomean, yesmean), 
                                        meanval, 
                                        max(nomean, yesmean), 
                                        maxval]))

            histograms[i].place(i, X.T[i].copy(), Y.copy())
            histograms[i].calc()
        return histograms
        
    def hist9(self, X, Y):
        no = np.array([x[(Y==0.0).T[0]] for x in X.T]).T
        yes = np.array([x[(Y==1.0).T[0]] for x in X.T]).T
        histograms = []
        # Calculate bin boundaries
        for i in range(len(X.T)):
            std = X.T[i].std()
            minval = X.T[i].min()
            maxval = X.T[i].max()
            meanval = X.T[i].mean()
            nomean = no.T[i].mean()
            yesmean = yes.T[i].mean()
            lowmean = min(nomean, yesmean)
            highmean = max(nomean, yesmean)
            l1, l2 = meanval-std, meanval-std/2.0
            h1, h2 = meanval+std/2.0, meanval+std
            vals = [minval, maxval, lowmean, highmean, l1, l2, h1, h2, lowmean-std, highmean+std]
            # Set up bins
            histograms.append(Histogram(sorted(vals)))

            histograms[i].place(i, X.T[i].copy(), Y.copy())
            histograms[i].calc()
        return histograms


    def train(self, method=None, bins=None):
        self.X_train, self.Y_train = [self.train_data], [self.train_labels]
        self.X_test, self.Y_test, self.P = [self.test_data], [self.test_labels], []

        # Calculate the prior
        p = [float(len(self.train_labels[self.train_labels==0.0])) / float(len(self.train_labels))]
        p.append(1.0-p[0])

        # Calculate the parameters for our naive bayes classification
        mean, e0, e1 = self.bernoulli(self.train_data, self.train_labels)
        self.P.append((mean, e0, e1, p))


    def uvg(self, x, mean, var):
        c = 1.0 / (np.sqrt((2*np.pi))*var)
        e = -((x-mean)**2)/(2*var**2)
        return c*np.exp(e)

    def predict_gaussian(self, Xi, Pi, j, k):
        mean,var,p = Pi
        e0 = self.uvg(Xi[j][k], mean[0][k], var[0][k]) 
        e1 = self.uvg(Xi[j][k], mean[1][k], var[1][k]) 
        return e0, e1

    def predict_bernoulli(self, Xi, Pi, j, k):
        mean,e0,e1,p = Pi
        e = [0, 0]
        e[0] = (((Xi[j][k]>=mean[k])*e0[k])+((Xi[j][k]<mean[k])*(1-e0[k]))) 
        e[1] = (((Xi[j][k]>=mean[k])*e1[k])+((Xi[j][k]<mean[k])*(1-e1[k]))) 
        return e[0], e[1]

    def predict_histogram(self, Xi, Pi, j, k):
        histograms,p = Pi
        return histograms[k].prob(Xi[j][k])

    def predict(self, Xi, Yi, Pi, method):
        if method == 'gaussian':
            predict_func = self.predict_gaussian
        elif method == 'bernoulli':
            predict_func = self.predict_bernoulli
        elif method == 'histogram':
            predict_func = self.predict_histogram
        c = 0
        m, d = len(Xi), len(Xi.T)
        # Calculate the prior
        p = [float(len(Yi[Yi==0.0])) / float(len(Yi))]
        p.append(1.0-p[0])
        # Calculate the likelihood
        likelihood = []
        for j in range(m):
            e = [1.0, 1.0]
            for k in range(d):
                if ~np.isnan(Xi[j][k]):
                    e0, e1 = predict_func(Xi, Pi, j, k)
                    e[0] *= e0 
                    e[1] *= e1
            likelihood.append((e[0]*p[0], e[1]*p[1]))
        # Calculate accuracy
        for i,l in enumerate(likelihood):
            y = 1 if l[1]>l[0] else 0
            c += 1 if y==Yi[i] else 0
        acc = float(c)/float(m)
        return acc

    def evaluate(self, X, Y, P, method):
        k = len(X)
        tacc = 0.0
        for i in range(k):
            acc = self.predict(X[i], Y[i], P[i], method)
            tacc += acc
        return tacc


    def test(self, method=None, accuracy=False, error=False, roc=False, label='default'):
        print "Training accuracy:", self.evaluate(self.X_train, self.Y_train, self.P, method)
        print "Testing accuracy:", self.evaluate(self.X_test, self.Y_test, self.P, method)


if __name__=="__main__":
    # Get cmdline args
    parser = argparse.ArgumentParser()
    args = parser.parse_args(sys.argv[1:])
    train_file = "/home/jaffe5/Documents/classes/fall_2015/machine_learning/hw5/data/spam_missing/20_percent_missing_train.txt" 
    test_file = "/home/jaffe5/Documents/classes/fall_2015/machine_learning/hw5/data/spam_missing/20_percent_missing_test.txt"
    # Do naive bayes estimation
    nb = NB(train_file, test_file)
    nb.train(method="bernoulli")
    nb.test(method="bernoulli")
