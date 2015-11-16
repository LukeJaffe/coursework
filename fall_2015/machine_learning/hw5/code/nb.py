import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from kfolder import KFolder
from evaluator import Evaluator

eps = 1e-5

class NB:
    def __init__(self):
        dataset = PollutedSpambase()
        train_data, train_labels = dataset.training()
        test_data, test_labels = dataset.testing()

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

    def train(self, method=None, bins=None):
        k = 10 
        kfolder = KFolder(self.D, k, normalize=True, shuffle=False)
        self.X_train, self.Y_train = [], []
        self.X_test, self.Y_test, self.P = [], [], []
        for i in range(k):
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

    def predict_histogram(self, Xi, Pi, j, k):
        histograms,p = Pi
        return histograms[k].prob(Xi[j][k])

    def predict(self, Xi, Yi, Pi, method, accuracy=False, error=False, roc=False, label='default'):
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
        # Draw ROC curve and calculated AUC
        if roc:
            thresh = []
            for i,l in enumerate(likelihood):
                thresh.append( np.log(l[1]/l[0]) )
            TPR, FPR = [], []
            for num,t in enumerate(sorted(thresh)):
                #print num,"/",len(thresh)
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
            plt.plot(FPR, TPR, label=label)
            plt.axis([0.0,1.0,0.0,1.0])
            s = 0.0 
            for k in range(2,m):
                s += ((FPR[k]-FPR[k-1])*(TPR[k]+TPR[k-1])) 
            s *= 0.5
            return abs(s)
        # Calculate error
        if error:
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
            return tpr,fpr,err
        # Calculate accuracy
        if accuracy:
            for i,l in enumerate(likelihood):
                y = 1 if l[1]>l[0] else 0
                c += 1 if y==Yi[i] else 0
            acc = float(c)/float(m)
            print c,"/",m,":",acc
            return acc

    def evaluate(self, X, Y, P, method, accuracy=False, error=False, roc=False, label='default'):
        k = len(X)
        if accuracy:
            tacc = 0.0
            for i in range(k):
                acc = self.predict(X[i], Y[i], P[i], method, accuracy=True)
                tacc += acc
            print "total:",tacc/float(len(X))
        elif error:
            ttpr, tfpr, terr = 0.0, 0.0, 0.0
            for i in range(k):
                tpr,fpr,err = self.predict(X[i], Y[i], P[i], method, error=True)
                print "Fold %d: TPR=%f, FPR=%f, Error=%f" % (i+1, tpr, fpr, err)
                ttpr += tpr
                tfpr += fpr
                terr += err
            print "Average accross k folds: TPR=%f, FPR=%f, Error=%f" % (ttpr/float(k), tfpr/float(k), terr/float(k))
        elif roc:
            auc = self.predict(X[0], Y[0], P[0], method, roc=True, label=label)
            print "AUC:", auc


    def test(self, method=None, accuracy=False, error=False, roc=False, label='default'):
        # Training error
        if not roc:
            print "Training error:"
            evaluator = self.evaluate(self.X_train, self.Y_train, self.P, method,
                                        accuracy=accuracy, error=error, roc=roc)
        print "Testing error:"
        evaluator = self.evaluate(self.X_test, self.Y_test, self.P, method,
                                    accuracy=accuracy, error=error, roc=roc, label=label)


if __name__=="__main__":
    # Get cmdline args
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', help='Select the number of bins for histogram method: {4, 9}')
    args = parser.parse_args(sys.argv[1:])
    # Do naive bayes estimation
    nb = NB(data_file)
    nb.train(method=args.m, bins=int(args.b))
    nb.test(method=args.m, accuracy=True, label=None)
