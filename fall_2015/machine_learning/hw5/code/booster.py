import numpy as np
from multiprocessing import Pool

class Booster:
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
            #if p == 0.0:
            #    yield (1e-5, 0.5, t)
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
        H = np.zeros(len(X.T[0]))
        for a,f,t in classifier:
            x = X.T[f]
            H += self.prediction(x, t)*a
        H[H>=thresh] = False
        H[H<thresh] = True
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
            PR.append(tpr), FPR.append(fpr)
        return TPR, FPR

    def auc(self, X, Y, C):
        TPR, FPR = self.roc(X, Y, C)        
        s = 0.0
        for k in range(2, len(FPR)):
            s += ((FPR[k]-FPR[k-1])*(TPR[k]+TPR[k-1]))
        s *= 0.5
        return abs(s)

    def boost(self, X, Y, T, threshold, rounds, random=False):
        # Initialize classifier
        classifier = []
        # Initialize weights
        m = len(X)
        D = np.ones(m)
        D /= float(m)
        for i in range(rounds):
            print "Round:",i
            # Apply split with weight vector D
            if random:
                rank, err, thresh, feature = self.split_random(X, Y, D, threshold)
            else:
                split = self.split(X, Y, D, threshold)
                rank, err, thresh, feature = max(split)
            #print rank, err, thresh, feature
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
        train_error = float(c_train)/float(len(Y))
        print "Train error:", train_error
        # Compute testing error
        H_test = self.hypothesis(classifier, T)
        #c_test = len((H_test!=Yt.ravel()).nonzero()[0])
        #test_error = float(c_test)/float(len(Yt))
        #print train_error,test_error
        # Compute final test AUC
        #test_auc = self.auc(T, Yt, classifier)
        return H_train, H_test

    def thresh(self, X):
        thresh = []
        for f in X.T:
            u = np.unique(f)
            thresh.append(u)#[:-1])
        return np.array(thresh)
