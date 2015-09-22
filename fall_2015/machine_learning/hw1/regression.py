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

    def solve(self):
        p1 = np.dot(self.X.T,self.X)
        i1 = np.linalg.pinv(p1)
        p2 = np.dot(self.X.T, self.Y)
        return np.dot(i1, p2)


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
        self.X, self.Y, self.W = [], [], []
        for i in range(k):
            # Get data and labels at fold k
            X,Y = kfolder.training(i)

            # Solve for the vector of linear factors, W
            rsolver = RegressionSolver(X, Y)
            Wi = rsolver.solve()

            # Get the testing data
            Xi,Yi = kfolder.testing(i)

            # Store the results
            self.X.append(Xi), self.Y.append(Yi), self.W.append(Wi)

    def test(self):
        evaluator = Evaluator(self.X, self.Y, self.W)
        evaluator.MSE()
        evaluator.accuracy()


class HousingRegressor: 
    def __init__(self, train_file, test_file):
        self.X_train, self.Y_train = self.get_data(train_file)
        self.X_test, self.Y_test = self.get_data(test_file)
        print "Training:", self.X_train.shape, self.Y_train.shape
        print "Testing:", self.X_test.shape, self.Y_test.shape
        
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

    def train(self):
        rsolver = RegressionSolver(self.X_train, self.Y_train)
        self.W = rsolver.solve()

    def test(self):
        evaluator = Evaluator([self.X_test], [self.Y_test], [self.W])
        evaluator.MSE()


if __name__=="__main__":
    # Get cmdline args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Run regression on this dataset.')
    args = parser.parse_args(sys.argv[1:])
    if args.d is not None:
        if args.d == 'spam':
            data_file = "data/spambase/spambase.data"
            sr = SpamRegressor(data_file)
            sr.train()
            sr.test()
            
        elif args.d == 'housing':
            train_file = "data/housing/housing_train.txt"
            test_file = "data/housing/housing_test.txt"
            hr = HousingRegressor(train_file, test_file)
            hr.train()
            hr.test()
        else:
            print "Unknown dataset."
            sys.exit() 
    else:
        print "No dataset given."
