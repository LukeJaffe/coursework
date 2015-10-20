import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from kfolder import KFolder
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
    def __init__(self, data_file):
        dmat = []
        f = open(data_file, "r")
        for line in f:
            x = line.split(',')
            x = [float(e) for e in x]
            dmat.append(x)
        self.D = np.array(dmat)

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
            mean.append(X.T[i].mean())
            n0 = len((no.T[i]>mean[i]).nonzero()[0])
            n1 = len((yes.T[i]>mean[i]).nonzero()[0])
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
        k = 10 
        kfolder = KFolder(self.D, k, normalize=True, shuffle=False)
        self.X_train, self.Y_train = [], []
        self.X_test, self.Y_test, self.P = [], [], []
        for i in range(1):
            # Get data and labels at fold k
            X,Y = kfolder.training(i)

            # Get the testing data
            Xi,Yi = kfolder.testing(i)

            # Store the results
            self.X_train.append(X), self.Y_train.append(Y)

            # Calculate the prior
            p = [float(len(Y[Y==0.0])) / float(len(Y))]
            p.append(1.0-p[0])

            # Calculate the parameters for our naive bayes classification
            if method == 'gaussian':
                mean, var = self.gaussian(X, Y) 
                self.X_test.append(Xi), self.Y_test.append(Yi), self.P.append((mean, var, p))
            elif method == 'bernoulli':
                mean, e0, e1 = self.bernoulli(X, Y)
                self.X_test.append(Xi), self.Y_test.append(Yi), self.P.append((mean, e0, e1, p))
            elif method == 'histogram':
                if bins == 4:
                    histograms = self.hist4(X, Y)
                elif bins == 9:
                    histograms = self.hist9(X, Y)
                else:
                    print "Unsupported number of bins."
                self.X_test.append(Xi), self.Y_test.append(Yi), self.P.append((histograms, p))


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

    def predict(self, Xi, Yi, Pi, method, error=True, roc=True):
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
                e0, e1 = predict_func(Xi, Pi, j, k)
                e[0] *= e0 
                e[1] *= e1
            likelihood.append((e[0]*p[0], e[1]*p[1]))
        if roc:
            thresh = []
            for i,l in enumerate(likelihood):
                thresh.append( np.log(l[1]/l[0]) )
            TPR, FPR = [], []
            for num,t in enumerate(sorted(thresh)):
                print num,"/",len(thresh)
                tp, tn, fp, fn = 0, 0, 0, 0
                for i,l in enumerate(likelihood):
                    y = 1 if np.log(l[1]/l[0])>t else 0
                    if y==Yi[i]==1.0:   tp += 1
                    elif y==Yi[i]==0.0: tn += 1
                    elif y>Yi[i]:       fp += 1
                    else:               fn += 1
                tpr = float(tp) / (float(tp)+float(fn)) 
                fpr = float(fp) / (float(fp)+float(tn)) 
                TPR.append(tpr), FPR.append(fpr)
            plt.plot(FPR, TPR, label='label')
            plt.axis([0.0,1.0,0.0,1.0])
            s = 0.0 
            for k in range(2,m):
                s += ((FPR[k]-FPR[k-1])*(TPR[k]+TPR[k-1])) 
            s *= 0.5
            print abs(s)
            return 0.0

        elif error:
            tp, tn, fp, fn = 0, 0, 0, 0
            for i,l in enumerate(likelihood):
                y = 1 if l[1]>l[0] else 0
                if y==Yi[i]==1.0:   tp += 1
                elif y==Yi[i]==0.0: tn += 1
                elif y>Yi[i]:       fp += 1
                else:               fn += 1
            tpr = float(tp) / (float(tp)+float(fn)) 
            fpr = float(fp) / (float(fp)+float(tn)) 
            err = float(fp+fn) / float(m)
            return err
        else:
            for i,l in enumerate(likelihood):
                y = 1 if l[1]>l[0] else 0
                c += 1 if y==Yi[i] else 0
            acc = float(c)/float(m)
            print c,"/",m,":",acc
            return acc

    def evaluate(self, X, Y, P, method, roc=False):
        tacc = 0.0
        for i in range(len(X)):
            acc = self.predict(X[i], Y[i], P[i], method, roc=roc)
            tacc += acc
        print "total:",tacc/float(len(X))


    def test(self, method=None):
        # Training error
        print "Training accuracy:"
        evaluator = self.evaluate(self.X_train, self.Y_train, self.P, method)
        print "Testing accuracy:"
        evaluator = self.evaluate(self.X_test, self.Y_test, self.P, method, roc=True)
        # Plot the ROC curve
        plt.title('Comparison of ROC for Regression Methods on Spam Data')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        legend = plt.legend(loc='lower right', shadow=True)
        plt.show()


if __name__=="__main__":
    # Get cmdline args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help='Select the method for performing naive bayes: {bernoulli, gaussian, histogram}')
    parser.add_argument('-b', help='Select the number of bins for histogram method: [int]')
    args = parser.parse_args(sys.argv[1:])
    if args.b is None:
        args.b = 0
    data_file = "../data/spambase/spambase.data"
    # Do naive bayes estimation
    nb = NB(data_file)
    nb.train(method=args.m, bins=int(args.b))
    nb.test(method=args.m)
