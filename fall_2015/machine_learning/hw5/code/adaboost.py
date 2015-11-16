import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

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
        h = np.ones_like(feature)
        idx = (feature > thresh).ravel()
        h[idx] = -1.0
        return h

    def err(self, x, Y, D, threshold):
        for t in threshold:
            idx1 = (x <= t).ravel()
            idx2 = (x > t).ravel()
            d1, d2 = D[idx1], D[idx2]
            y1, y2 = Y[idx1], Y[idx2]
            c1 = (y1 == 0.0).ravel().astype(float)
            c2 = (y2 == 1.0).ravel().astype(float)
            p = np.dot(c1, d1) + np.dot(c2, d2)
            if p == 0.0:
                yield (1e-5, 0.5, t)
            else:
                yield (np.abs(0.5-p), p, t)
        
    def split_best(self, X, Y, D, threshold):
        for i,f in enumerate(X.T):
            best = max(self.err(f, Y, D, threshold[i]))
            yield (best[0], best[1], best[2], i)

    def split_random(self, X, Y, D, threshold):
        f = np.random.randint(len(X.T))
        x = X.T[f]
        i = np.random.randint(len(threshold[f]))
        t = threshold[f][i]
        idx1 = (x <= t).ravel()
        idx2 = (x > t).ravel()
        d1, d2 = D[idx1], D[idx2]
        y1, y2 = Y[idx1], Y[idx2]
        c1 = (y1 == 0.0).ravel().astype(float)
        c2 = (y2 == 1.0).ravel().astype(float)
        p = np.dot(c1, d1) + np.dot(c2, d2) 
        return (np.abs(0.5-p), p, t, f)

    def hypothesis(self, classifier, X, thresh=0.0):
        H = np.zeros_like(X.T[0])
        for a,f,t in classifier:
            x = X.T[f]
            H += self.prediction(x, t)*a
        H[H>=thresh] = 1.0
        H[H<thresh] = 0.0
        return H

    def roc(self, X, Y, C, fast=True):
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

    def margin(self, classifier, X, Y):
        # Convert labels to {-1,+1}
        y = np.ones_like(Y).ravel()
        idx = (y==False).ravel()
        y[idx] = -1.0
        # Prep
        num_features = len(X.T)
        print "num features:",num_features
        H = np.zeros_like(X.T)
        T = np.zeros(len(X))
        # Iterate through classifier to determine feature margin
        for a,f,t in classifier:
            x = X.T[f]
            h = self.prediction(x, t).ravel()
            H[f] += np.abs(h*a*y)
            T += np.abs(h*a*y)
        # Condense columns of H to elements of fmargin
        fmargin = H.sum(axis=1)/T.sum()
        print fmargin
        print np.argsort(-fmargin)[:15]
        print fmargin.sum()


    def boost(self, X, Y, T, Yt, threshold, random=False):
        # Initialize result structures
        round_error = []
        train_error = []
        test_error = []
        test_auc = []
        # Initialize classifier
        classifier = []
        # Initialize weights
        m = len(X)
        D = np.ones(m)
        D /= float(m)
        if random:
            rounds = 1000
        else:
            rounds = 300
        for i in range(rounds):
            print "Round:",i+1
            # Apply split with weight vector D
            if not random: 
                split = self.split_best(X, Y, D, threshold)
                rank, err, thresh, feature = max(split)
            else:
                rank, err, thresh, feature = self.split_random(X, Y, D, threshold)
            print rank, err, thresh, feature
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
            print "train:", c_train, '/', len(Y), ':', e_train
            train_error.append((i, e_train))
            # Compute testing error
            H_test = self.hypothesis(classifier, T)
            c_test = len((H_test!=Yt.ravel()).nonzero()[0])
            e_test = float(c_test)/float(len(Yt))
            print "test:", c_test, '/', len(Yt), ':', e_test
            test_error.append((i, e_test))
            # Compute test AUC
            #auc = self.auc(T, Yt, classifier)
            #test_auc.append((i, auc))
        self.margin(classifier, X, Y)
        # Compute final test ROC
        test_roc = self.roc(T, Yt, classifier, fast=False)
        return round_error, train_error, test_error, test_auc, test_roc

    def thresh(self, X):
        thresh = []
        for f in X.T:
            u = np.unique(f)
            thresh.append(u)
        return np.array(thresh)

    def train(self, random=False):
        k = 10 
        kfolder = KFolder(self.D, k, normalize=True, shuffle=False)
        self.X_train, self.Y_train = [], []
        self.X_test, self.Y_test, self.P = [], [], []
        for i in range(1):
            # Get data and labels at fold k
            X,Y = kfolder.training(i)
            # Get the testing data
            Xi,Yi = kfolder.testing(i)
            # Solve for the vector of linear factors, W
            return self.boost(X, Y, Xi, Yi, self.thresh(X), random=random) 

    def plot(self, result):
        if False:
            strat = 'Optimal '
        else:
            strat = 'Random '
        round_error, train_error, test_error, test_auc, test_roc = result
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
        # Plot test_auc
        x4,y4 = zip(*test_auc)
        plt.plot(x4, y4, color='red')
        plt.title(strat+'Stump: AUC')
        plt.xlabel('Iteration Step')
        plt.ylabel('AUC')
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
    parser.add_argument('-r', help='Random or not?')
    args = parser.parse_args(sys.argv[1:])
    data_file = "../data/spambase/spambase.data"
    booster = Booster(data_file)
    if args.r is not None:
        result = booster.train(random=bool(int(args.r)))
        #booster.plot(result)
