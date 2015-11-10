import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from parse import Parser
from kfolder import KFolder

class Booster:
    def __init__(self, config_file, data_file):
        vote_parser = Parser(config_file, data_file)
        vote_parser.parse_config()
        self.D = vote_parser.parse_data()

    def prediction(self, feature, thresh):
        h = np.ones_like(feature)
        idx = (feature > thresh).ravel()
        h[idx] = -1.0
        return h

    def err(self, x, Y, D, threshold):
        #print zip(x, Y.ravel())
        for t in threshold:
            idx1 = (x <= t).ravel()
            idx2 = (x > t).ravel()
            d1, d2 = D[idx1], D[idx2]
            y1, y2 = Y[idx1], Y[idx2]
            c1 = (y1 == 0.0).ravel().astype(float)
            c2 = (y2 == 1.0).ravel().astype(float)
            p = np.dot(c1, d1) + np.dot(c2, d2)# + 1.0/len(D)
            #print p
            if p == 0.0:
                return
            yield (np.abs(0.5-p), p, t)
        
    def split(self, X, Y, D, threshold):
        for i,f in enumerate(X.T):
            #print "f:",i
            best = max(self.err(f, Y, D, threshold[i]))
            yield (best[0], best[1], best[2], i)

    def hypothesis(self, classifier, X, thresh=0.0):
        H = np.zeros_like(X.T[0])
        for a,f,t in classifier:
            x = X.T[f]
            H += self.prediction(x, t)*a
        H[H>=thresh] = 1.0
        H[H<thresh] = 0.0
        return H

    def roc(self, X, Y, C, fast=False):
        H = np.zeros_like(X.T[0])
        for a,f,t in C:
            x = X.T[f]
            H += self.prediction(x, t)*a
        TPR, FPR = [], []
        if fast:
            idx = np.array(range(0,len(H), 5))
            sH = sorted(H[idx])
        else:
            sH = sorted(H)
        print len(sH)
        for thresh in sH:
            h = H.copy()
            h[h>=thresh] = 1.0
            h[h<thresh] = 0.0
            tp, tn, fp, fn = 0, 0, 0, 0
            for i,y in enumerate(h):
                if y==Y[i]==1.0:   tp += 1
                elif y==Y[i]==0.0: tn += 1
                elif y>Y[i]:       fp += 1
                else:              fn += 1
            tpr = float(tp) / (float(tp)+float(fn))
            fpr = float(fp) / (float(fp)+float(tn))
            TPR.append(tpr), FPR.append(fpr)
        return TPR, FPR

    def auc(self, X, Y, C):
        TPR, FPR = self.roc(X, Y, C)        
        #plt.plot(FPR, TPR)
        #plt.show()
        s = 0.0
        for k in range(2, len(FPR)):
            s += ((FPR[k]-FPR[k-1])*(TPR[k]+TPR[k-1]))
        s *= 0.5
        return abs(s)
        #print "AUC:",abs(s)


    def boost(self, X, Y, T, Yt, threshold):
        # Initialize result structures
        round_error = []
        train_error = []
        test_error = []
        # Initialize classifier
        classifier = []
        # Initialize weights
        m = len(X)
        D = np.ones(m)
        D /= float(m)
        for i in range(10):
            # Apply split with weight vector D
            split = self.split(X, Y, D, threshold)
            rank, err, thresh, feature = max(split)
            #print rank, err, thresh, feature
            round_error.append((i,err))
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
            classifier.append((a, feature, thresh))
            # Compute training error
            H_train = self.hypothesis(classifier, X)
            c_train = len((H_train!=Y.ravel()).nonzero()[0])
            e_train = float(c_train)/float(len(Y))
            #print "train:", c_train, '/', len(Y), ':', e_train
            train_error.append((i, e_train))
            # Compute testing error
            H_test = self.hypothesis(classifier, T)
            c_test = len((H_test!=Yt.ravel()).nonzero()[0])
            e_test = float(c_test)/float(len(Yt))
            #print "test:", c_test, '/', len(Yt), ':', e_test
            test_error.append((i, e_test))
        # Compute final test ROC
        #test_roc = self.roc(T, Yt, classifier, fast=False)
        #return round_error, train_error, test_error, test_roc
        return 1.0 - e_test

    def thresh(self, X):
        thresh = []
        for f in X.T:
            u = np.unique(f)
            thresh.append(u[:-1])
        return np.array(thresh)

    def train(self, shared=True):
        k = 10
        kfolder = KFolder(self.D, k, normalize=False, shuffle=False)
        self.X_train, self.Y_train = [], []
        self.X_test, self.Y_test, self.P = [], [], []
        test_acc = np.zeros(k)
        for i in range(k):
            # Get data and labels at fold k
            X,Y = kfolder.training(i)
            # Get the testing data
            Xi,Yi = kfolder.testing(i)
            # Solve for the vector of linear factors, W
            test_acc[i] = self.boost(X, Y, Xi, Yi, self.thresh(X)) 
        print "Average test_acc:",test_acc.mean()

    def plot(self, result):
        strat = 'Optimal '
        round_error, train_error, test_error, test_roc = result
        # Plot round_error
        x1,y1 = zip(*round_error)
        plt.plot(x1, y1, color='red')
        plt.title(strat+'Stump: Round Error')
        plt.xlabel('Iteration Step')
        plt.ylabel('Round Error')
        plt.show()
        # Plot train_error and test_error
        x2,y2 = zip(*train_error)
        x3,y3 = zip(*test_error)
        plt.plot(x2, y2, color='blue')
        plt.plot(x3, y3, color='red')
        plt.title(strat+'Stump: Train/Test Error')
        plt.xlabel('Iteration Step')
        plt.ylabel('Test/Train Error (Red/Blue Color)')
        plt.show()
        # Plot test_roc
        TPR,FPR = test_roc
        plt.plot(FPR, TPR, color='red')
        plt.title(strat+'Stump: ROC')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()
        

if __name__=="__main__":
    # Get cmdline args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Shared or separate coveriance matrices.')
    args = parser.parse_args(sys.argv[1:])
    if args.d == 'vote':
        config_file = '../data/vote/vote.config'
        data_file = '../data/vote/vote.data'
    elif args.d == 'crx':
        config_file = '../data/crx/crx.config'
        data_file = '../data/crx/crx.data'
    booster = Booster(config_file, data_file)
    booster.train()
    #booster.plot(result)
