import sys
import argparse
import numpy as np

from kfolder import KFolder


class Booster:
    def __init__(self, data_file):
        dmat = []
        f = open(data_file, "r")
        for line in f:
            x = line.split(',')
            x = [float(e) for e in x]
            dmat.append(x)
        self.D = np.array(dmat)

    def prediction(self, feature, thresh):
        #print thresh, feature
        h = np.ones_like(feature)
        idx = (feature > thresh).ravel()
        h[idx] = -1.0
        return h

    def err(self, x, y, D):
        guess = np.array([], dtype=int)
        idx = np.argsort(x)
        for i,j in enumerate(idx):
            guess = np.append(guess, j)
            d1 = D[guess]
            d2 = D[np.setdiff1d(idx, guess, assume_unique=True)]
            s1 = y[guess]
            s2 = y[np.setdiff1d(idx, guess, assume_unique=True)]
            c1 = (s1==1.0).ravel()
            c2 = (s2==0.0).ravel()
            i1T = (c1==True).ravel()
            i1F = (c1==False).ravel()
            i2T = (c2==True).ravel()
            i2F = (c2==False).ravel()
            c1[i1T] = 0.0
            c1[i1F] = 1.0
            c2[i2T] = 0.0
            c2[i2F] = 1.0
            w1 = c1*d1
            w2 = c2*d2
            p = sum(w1) + sum(w2)
            yield (np.abs(0.5-p), p, x[j])
        
    def split(self, X, Y, D):
        for i,f in enumerate(X.T):
            #print i
            best = max(self.err(f, Y, D))
            yield (best[0], best[1], best[2], i)

    def hypothesis(self, classifier, X):
        H = np.zeros_like(X.T[0])
        for a,f,t in classifier:
            #print a
            x = X.T[f]
            #print self.prediction(x,t)
            H += self.prediction(x, t)*a
        H[H>=0] = 1.0
        H[H<0] = 0.0
        return H

    def boost(self, X, Y, T, Yt):
        # Initialize classifier
        classifier = []
        # Initialize weights
        m = len(X)
        D = np.ones(m)
        D /= float(m)
        for i in range(100):
            # Apply split with weight vector D
            split = self.split(X, Y, D)
            rank, err, thresh, feature = max(split)
            print rank, err, thresh, feature
            # Calculate update factor
            a = 0.5*np.log((1.0-err)/err)
            # Reproduce hypothesis vector
            h = self.prediction(X.T[feature], thresh)
            # Correct label vector 
            nY = Y.copy()
            nY[nY==0.0] = -1.0
            nY = nY.ravel()
            # Compute new weights
            Z = D*np.exp(-a*nY*h)
            D = Z/sum(Z)
            # Add new element to classifier
            #print err, a, feature, thresh
            classifier.append((a, feature, thresh))
            # Compute training error
            H_train = self.hypothesis(classifier, X)
            print "train:",len((H_train==Y.ravel()).nonzero()[0])
            # Compute testing error
            H_test = self.hypothesis(classifier, T)
            print "test:",len((H_test==Yt.ravel()).nonzero()[0])

    def train(self, shared=True):
        k = 10 
        kfolder = KFolder(self.D, k, normalize=True, shuffle=True)
        self.X_train, self.Y_train = [], []
        self.X_test, self.Y_test, self.P = [], [], []
        for i in range(1):
            # Get data and labels at fold k
            X,Y = kfolder.training(i)

            # Get the testing data
            Xi,Yi = kfolder.testing(i)

            # Solve for the vector of linear factors, W
            self.boost(X[:100], Y[:100], Xi, Yi) 

            # Store the results
            #self.X_train.append(X), self.Y_train.append(Y)
            #self.X_test.append(Xi), self.Y_test.append(Yi), self.P.append((u,cov,p))

if __name__=="__main__":
    # Get cmdline args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help='Shared or separate coveriance matrices.')
    args = parser.parse_args(sys.argv[1:])
    data_file = "../data/spambase/spambase.data"
    booster = Booster(data_file)
    booster.train()
