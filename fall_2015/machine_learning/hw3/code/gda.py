import sys
import argparse
import numpy as np

from kfolder import KFolder
from evaluator import Evaluator

class GDA:
    def __init__(self, data_file):
        dmat = []
        f = open(data_file, "r")
        for line in f:
            x = line.split(',')
            x = [float(e) for e in x]
            dmat.append(x)
        self.D = np.array(dmat)

    def estimate(self, X, Y, shared):
        m = len(Y)
        d = len(X.T)
        # Split spam and no spam
        nspam = np.array([x[(Y==0.0).T[0]] for x in X.T]).T
        spam = np.array([x[(Y==1.0).T[0]] for x in X.T]).T
        p = [float(len(nspam))/float(len(X)), float(len(spam))/float(len(X))]
        # Estimate u0, u1
        u0, u1 = nspam.mean(0), spam.mean(0)
        # Initialize covariance
        u = (u0, u1)
        if shared:
            cov = np.zeros((d,d))
        else:
            cov = [np.zeros((d,d)), np.zeros((d,d))]
        # Estimate covariance
        for i in range(m):
            d = (X[i] - u[int(Y[i])])
            if shared:
                cov += np.dot(d[:,None],d[None,:])
            else:
                cov[int(Y[i])] += np.dot(d[:,None],d[None,:])
        # Normalize covariance
        if shared:
            cov /= float(m)
            #print np.diagonal(cov)
            np.fill_diagonal(cov, np.diagonal(cov)+1e-5)
            #print np.diagonal(cov)
        else:
            cov = [c/float(m) for c in cov]
            [np.fill_diagonal(c, np.diagonal(c)+1e-5) for c in cov]
        # Print stats and return
        return u,cov,p
         

    def train(self, shared=True):
        k = 10 
        kfolder = KFolder(self.D, k, normalize=True, shuffle=True)
        self.X_train, self.Y_train = [], []
        self.X_test, self.Y_test, self.P = [], [], []
        for i in range(k):
            # Get data and labels at fold k
            X,Y = kfolder.training(i)

            # Solve for the vector of linear factors, W
            u,cov,p = self.estimate(X, Y, shared) 

            # Get the testing data
            Xi,Yi = kfolder.testing(i)

            # Store the results
            self.X_train.append(X), self.Y_train.append(Y)
            self.X_test.append(Xi), self.Y_test.append(Yi), self.P.append((u,cov,p))

    def mvg(self, x, u, cov):
        c = (2*np.pi)**(float(len(u))/2)     
        #print "c, det:", c, np.linalg.det(sig)
        det = np.linalg.det(cov)**(0.5)
        p1 = 1.0/(c*det)
        d = (x-u)[:,None]
        p3 = np.dot(d.T,np.linalg.pinv(cov))
        p4 = -0.5*np.dot(p3,d)
        return p1*np.exp(p4)

    def evaluate(self, X, Y, P, shared):
        tacc = 0.0
        for i in range(len(X)):
            c = 0
            u,cov,p = P[i]
            for j in range(len(X[i])):
                if shared:
                    l0 = self.mvg(X[i][j], u[0], cov)*p[0]
                    l1 = self.mvg(X[i][j], u[1], cov)*p[1]
                else:
                    l0 = self.mvg(X[i][j], u[0], cov[0])*p[0]
                    l1 = self.mvg(X[i][j], u[1], cov[1])*p[1]
                y = 1 if l1>l0 else 0
                c += 1 if y==Y[i][j] else 0
            acc = float(c)/float(len(X[0]))
            print c,"/",len(X[0]),":",acc
            tacc += acc
        print "total:",tacc/float(len(X))


    def test(self, shared=True):
        # Training error
        print "Training accuracy:"
        evaluator = self.evaluate(self.X_train, self.Y_train, self.P, shared)
        print "Testing accuracy:"
        evaluator = self.evaluate(self.X_test, self.Y_test, self.P, shared)

if __name__=="__main__":
    # Get cmdline args
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-d', help='Run regression on this dataset.')
    #parser.add_argument('-r', help='Lambda parameter for ridge regression.')
    #parser.add_argument('-m', help='Method of regression. {normal, descent}')
    #args = parser.parse_args(sys.argv[1:])
    data_file = "../data/spambase/spambase.data"
    gda = GDA(data_file)
    shared = False
    gda.train(shared=shared)
    gda.test(shared=shared)
