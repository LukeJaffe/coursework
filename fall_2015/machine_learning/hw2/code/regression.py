import sys
import argparse
import numpy as np
from math import exp

from kfolder import KFolder
from evaluator import Evaluator


class RegressionSolver:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y    

    def solve(self, l):
        p1 = np.dot(self.X.T,self.X)
        p2 = l*np.eye(len(self.X.T))
        i1 = np.linalg.pinv(p1+p2)
        p3 = np.dot(self.X.T, self.Y)
        return np.dot(i1, p3)


class SpamRegressor:
    def __init__(self, data_file):
        dmat = []
        f = open(data_file, "r")
        for line in f:
            x = line.split(',')
            x = [float(e) for e in x]
            dmat.append(x)
        self.D = np.array(dmat)

    def train(self):
        k = 10
        kfolder = KFolder(self.D, k, normalize=True)
        self.X_train, self.Y_train = [], []
        self.X_test, self.Y_test, self.W = [], [], []
        for i in range(k):
            # Get data and labels at fold k
            X,Y = kfolder.training(i)

            # Solve for the vector of linear factors, W
            rsolver = RegressionSolver(X, Y)
            Wi = rsolver.solve()

            # Get the testing data
            Xi,Yi = kfolder.testing(i)

            # Store the results
            self.X_train.append(X), self.Y_train.append(Y)
            self.X_test.append(Xi), self.Y_test.append(Yi), self.W.append(Wi)

    def test(self):
        # Training error
        print "Training error:"
        evaluator = Evaluator(self.X_train, self.Y_train, self.W)
        evaluator.MSE()
        evaluator.accuracy()
        # Testing error
        print "Testing error:"
        evaluator = Evaluator(self.X_test, self.Y_test, self.W)
        evaluator.MSE()
        evaluator.accuracy()


class HousingRegressor: 
    def __init__(self, train_file, test_file, normalize=True):
        self.X_train, self.Y_train = self.get_data(train_file)
        self.X_test, self.Y_test = self.get_data(test_file)
        if normalize:
            F_train, F_test = self.X_train.T, self.X_test.T
            for i in range(len(F_train)):
                # Normalize training data
                min_val = F_train[i].min()
                F_train[i] -= min_val
                max_val = F_train[i].max()
                F_train[i] /= max_val
                # Normalize testing data
                F_test[i] -= min_val
                F_test[i] /= max_val

        #print "Training:", self.X_train.shape, self.Y_train.shape
        #print "Testing:", self.X_test.shape, self.Y_test.shape
        
    def get_data(self, data_file):
        X, Y = [], []
        f = open(data_file, "r")
        for line in f:
            if line.strip():
                x = line.split()
                x = [float(e) for e in x]
                X.append( x[:-1] )
                Y.append( x[-1:] )
        return np.array(X), np.array(Y)

    def train(self, l):
        rsolver = RegressionSolver(self.X_train, self.Y_train)
        self.W = rsolver.solve(l)

    def test(self):
        # Measure training error
        print "Training error:"
        test_eval = Evaluator([self.X_train], [self.Y_train], [self.W])
        test_eval.MSE()
        # Measure testing error
        print "Testing error:"
        test_eval = Evaluator([self.X_test], [self.Y_test], [self.W])
        test_eval.MSE()


if __name__=="__main__":
    # Get cmdline args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Run regression on this dataset.')
    parser.add_argument('-r', help='Lambda parameter for ridge regression')
    args = parser.parse_args(sys.argv[1:])
    r = 0
    if args.r is not None:
        r = float(args.r)
    if args.d is not None:
        if args.d == 'spam':
            data_file = "../data/spambase/spambase.data"
            sr = SpamRegressor(data_file)
            sr.train()
            sr.test()
            
        elif args.d == 'housing':
            train_file = "../data/housing/housing_train.txt"
            test_file = "../data/housing/housing_test.txt"
            hr = HousingRegressor(train_file, test_file, normalize=False)
            hr.train(r)
            hr.test()
        else:
            print "Unknown dataset."
            sys.exit() 
    else:
        print "No dataset given."
