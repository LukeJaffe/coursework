import sys
import argparse
import numpy as np
from math import exp


class KFolder:
    def __init__(self, X, k):
        self.k = k
        self.X = []
        np.random.shuffle(X)
        tot_len = len(X)
        fold_len = tot_len/k
        for i in range(0, tot_len, fold_len):
            self.X.append(X[i:i+fold_len])

        # Prepare the data dictionary
        training = {"data":[], "labels":[]}
        testing = {"data":[], "labels":[]}
        self.data = {"training":training,"testing":testing}

        # Prepare training and testing data
        for i in range(self.k):
            self.data["testing"]["data"].append(self.X[i].T[:-1])
            self.data["testing"]["labels"].append(self.X[i].T[-1:])
            #print self.data["testing"]["data"][i].shape,self.data["testing"]["labels"][i].shape
            tmp = self.X[:i]+self.X[i+1:]
            tmp = np.concatenate(tmp, axis=0)
            d = tmp.T[:-1]
            l = tmp.T[-1:]
            self.data["training"]["data"].append(d)
            self.data["training"]["labels"].append(l)
            #print self.data["training"]["data"][i].shape,self.data["training"]["labels"][i].shape
            #print


        # Normalize the data
        for i in range(self.k):
            d = self.data["testing"]["data"][i]
            t = self.data["training"]["data"][i]
            for j in range(len(d)):
                # Normalize the training data
                min_val = d[j].min()
                d[j] -= min_val
                max_val = d[j].max()
                d[j] /= max_val
                # Normalize the testing data with the same parameters
                t[j] -= min_val
                t[j] /= max_val
                # Clip testing data to [0.0, 1.0] 
                t[j] = np.clip(t[j], 0.0, 1.0)
            #print self.data["testing"]["data"][i].min(), self.data["testing"]["data"][i].max()
            #print self.data["training"]["data"][i].min(), self.data["training"]["data"][i].max()
            #print 

    def training(self, k):
        return self.data["training"]["data"][k].T, self.data["training"]["labels"][k].T

    def testing(self, k):
        return self.data["testing"]["data"][k].T, self.data["testing"]["labels"][k].T


class RegressionSolver:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y    

    def solve(self):
        p1 = np.dot(self.X.T,self.X)
        i1 = np.linalg.pinv(p1)
        p2 = np.dot(self.X.T, self.Y)
        return np.dot(i1, p2)


class Evaluator:
    def __init__(self, X, Y, W):
        self.X, self.Y, self.W = X, Y, W
            
    def MSE(self):
        mse = 0.0
        for i in range(len(self.X)):
            # Test solution against the testing data
            Xi, Yi, Wi = self.X[i], self.Y[i], self.W[i]
            ktotal = len(Yi)
            kmse = 0.0
            for j in range(len(Xi)):
                s = 0.0
                for e in range(len(Xi[j])):
                    # Basic predictor
                    s += Xi[j][e]*Wi[e]
                kmse += (Yi[j] - s)**2.0
            kmse /= float(ktotal)
            mse += kmse
        mse /= float(len(self.X))
        print "Average MSE:",mse

    def accuracy(self):
        acc = 0.0
        for i in range(len(self.X)):
            # Test solution against the testing data
            Xi, Yi, Wi = self.X[i], self.Y[i], self.W[i]
            ktotal = len(Yi)
            kcorrect = 0
            for j in range(len(Xi)):
                s = 0.0
                for e in range(len(Xi[j])):
                    # Basic predictor
                    s += Xi[j][e]*Wi[e]
                    # Logistic predictor
                    #s = 1.0 / (1.0 + exp(-s))
                if s >= 0.5:
                    s = 1.0
                else:
                    s = 0.0
                if s == Yi[j]:
                    kcorrect += 1
            kacc = float(kcorrect)/float(ktotal)
            acc += kacc
            #print "Correct:",str(kcorrect)+str('/')+str(ktotal)+' =',kacc
        acc /= float(len(self.X))
        print "Average accuracy:", acc


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
        kfolder = KFolder(self.D,k)
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
            data_file = "../data/spambase/spambase.data"
            sr = SpamRegressor(data_file)
            sr.train()
            sr.test()
            
        elif args.d == 'housing':
            train_file = "../data/housing/housing_train.txt"
            test_file = "../data/housing/housing_test.txt"
            hr = HousingRegressor(train_file, test_file)
            hr.train()
            hr.test()
        else:
            print "Unknown dataset."
            sys.exit() 
    else:
        print "No dataset given."
