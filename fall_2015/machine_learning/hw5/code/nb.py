import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from evaluator import Evaluator
from polluted import PollutedSpambase

eps = 1e-5

class NB:
    def __init__(self, pca=False):
        dataset = PollutedSpambase()
        self.train_data, self.train_labels = dataset.training()
        self.test_data, self.test_labels = dataset.testing()
        if pca:
            pca = PCA(n_components=100)
            pca.fit(self.train_data)
            # Project PCA onto testing data
            #print self.train_data.shape, self.test_data.shape
            self.train_data = pca.transform(self.train_data)
            self.test_data = pca.transform(self.test_data)
            #print self.train_data.shape, self.test_data.shape

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
                var[i].append(f.var()+1e-5)
        return mean, var

    def train(self):
        # Calculate the prior
        p = [float(len(self.train_labels[self.train_labels==0.0])) / float(len(self.train_labels))]
        p.append(1.0-p[0])

        # Calculate the parameters for our naive bayes classification
        mean, var = self.gaussian(self.train_data, self.train_labels) 
        self.P = (mean, var, p)


    def uvg(self, x, mean, var):
        c = 1.0 / np.sqrt(2*np.pi*var)
        e = -((x-mean)**2)/(2*var)
        return c*np.exp(e)

    def predict_gaussian(self, Xi, Pi, j, k):
        mean,var,p = Pi
        e0 = self.uvg(Xi[j][k], mean[0][k], var[0][k]) 
        e1 = self.uvg(Xi[j][k], mean[1][k], var[1][k]) 
        return e0, e1

    def predict_histogram(self, Xi, Pi, j, k):
        histograms,p = Pi
        return histograms[k].prob(Xi[j][k])

    def predict(self, Xi, Yi, Pi):
        predict_func = self.predict_gaussian
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

    def evaluate(self, X, Y, P):
        return self.predict(X, Y, P)

    def test(self):
        # Training error
        print "Training accuracy:", self.evaluate(self.train_data, self.train_labels, self.P)
        print "Testing accuracy:", self.evaluate(self.test_data, self.test_labels, self.P)


if __name__=="__main__":
    # Get cmdline args
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='PCA or no')
    args = parser.parse_args(sys.argv[1:])
    # Do naive bayes estimation
    nb = NB(pca=bool(int(args.p)))
    nb.train()
    nb.test()
