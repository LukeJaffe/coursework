import sys
import argparse
import numpy as np
from math import exp
import matplotlib.pyplot as plt

from kfolder import KFolder
from evaluator import Evaluator
from descent import GradientDescent


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
        dmat = np.array(dmat)
        h,w = dmat.shape
        self.D = np.ones((h,w+1))
        self.D[:,1:] = dmat

    def train(self, l, method='normal'):
        k = 10
        kfolder = KFolder(self.D, k, normalize=True)
        self.X_train, self.Y_train = [], []
        self.X_test, self.Y_test, self.W = [], [], []
        for i in range(k):
            # Get data and labels at fold k
            X,Y = kfolder.training(i)

            # Solve for the vector of linear factors, W
            if method == 'normal':
                rsolver = RegressionSolver(X, Y)
                Wi = rsolver.solve(l)
            elif method == 'descent':
                gd = GradientDescent(X, Y)
                Wi = gd.linreg_stoch2()
            elif method == 'logistic':
                gd = GradientDescent(X, Y)
                Wi = gd.logreg_stoch()

            # Get the testing data
            Xi,Yi = kfolder.testing(i)

            # Store the results
            self.X_train.append(X), self.Y_train.append(Y)
            self.X_test.append(Xi), self.Y_test.append(Yi), self.W.append(Wi)

    def test(self, label=None):
        # Training error
        print "Training error:"
        evaluator = Evaluator(self.X_train, self.Y_train, self.W)
        #evaluator.MSE()
        evaluator.accuracy()
        #evaluator.confusion()
        # Testing error
        print "Testing error:"
        evaluator = Evaluator(self.X_test, self.Y_test, self.W)
        #evaluator.MSE()
        evaluator.accuracy()
        evaluator.confusion()
        FPR, TPR = evaluator.roc()
        plt.plot(FPR, TPR, label=label)
        plt.axis([0.0,0.5,0.5,1.0])


class HousingRegressor: 
    def __init__(self, train_file, test_file, normalize=True):
        self.X_train, self.Y_train = self.get_data(train_file)
        self.X_test, self.Y_test = self.get_data(test_file)
        if normalize:
            F_train, F_test = self.X_train.T, self.X_test.T
            for i in range(len(F_train)):
                # Normalize training data
                if F_train[i].min() == F_train[i].max():
                    min_val = 0.0
                    max_val = F_train[i].max()
                else:
                    min_val = F_train[i].min()
                    F_train[i] -= min_val
                    max_val = F_train[i].max()
                    F_train[i] /= max_val
                # Normalize testing data
                F_test[i] -= min_val
                F_test[i] /= max_val
            self.X_train = F_train.T
            self.X_test = F_test.T

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
        X, Y = np.array(X), np.array(Y)
        h,w = X.shape
        nX = np.ones((h,w+1))
        nX[:,1:] = X
        return nX, Y

    def train(self, l, method='normal'):
        if method == 'normal':
            rsolver = RegressionSolver(self.X_train, self.Y_train)
            self.W = rsolver.solve(l)
        elif method == 'descent':
            gd = GradientDescent(self.X_train, self.Y_train)
            self.W = gd.linreg_stoch1(r=1e-6)

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
    parser.add_argument('-r', help='Lambda parameter for ridge regression.')
    parser.add_argument('-m', help='Method of regression. {normal, descent}')
    args = parser.parse_args(sys.argv[1:])
    r = 0
    m = 'normal'
    if args.m is not None:
        m = args.m
    if args.r is not None:
        r = float(args.r)
    if args.d is not None:
        if args.d == 'spam':
            data_file = "../data/spambase/spambase.data"
            sr = SpamRegressor(data_file)
            if m == 'all':
                sr.train(r, 'normal')
                sr.test(label='Linear Regression (Normal Equations)')
                sr.train(r, 'descent')
                sr.test(label='Linear Regression (Gradient Descent)')
                sr.train(r, 'logistic')
                sr.test(label='Logistic Regression (Gradient Descent)')
                plt.title('Comparison of ROC for Regression Methods on Spam Data')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                legend = plt.legend(loc='lower right', shadow=True)
                plt.show()
            else:
                sr.train(r, m)
                sr.test()
            
        elif args.d == 'housing':
            train_file = "../data/housing/housing_train.txt"
            test_file = "../data/housing/housing_test.txt"
            hr = HousingRegressor(train_file, test_file, normalize=False)
            hr.train(r, m)
            hr.test()
        else:
            print "Unknown dataset."
            sys.exit() 
    else:
        print "No dataset given."
