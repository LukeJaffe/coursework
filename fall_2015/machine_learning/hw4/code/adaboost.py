import sys
import argparse
import numpy as np

class Stump:
    def __init__(self, f, t):
        self.f = f
        self.t = t

class Booster:
    def __init__(self, data_file):
        dmat = []
        f = open(data_file, "r")
        for line in f:
            x = line.split(',')
            x = [float(e) for e in x]
            dmat.append(x)
        self.D = np.array(dmat)
        print self.D.shape

    def boost(self, X, Y):
        for i,f in enumerate(X.T):
            idx = np.argsort(f)
            y = []
            for x in f[idx]:
                stump = Stump(i, x) 


    def train(self, shared=True):
        k = 10 
        kfolder = KFolder(self.D, k, normalize=True, shuffle=True)
        self.X_train, self.Y_train = [], []
        self.X_test, self.Y_test, self.P = [], [], []
        for i in range(1):
            # Get data and labels at fold k
            X,Y = kfolder.training(i)

            # Solve for the vector of linear factors, W
            self.boost(X, Y) 

            # Get the testing data
            Xi,Yi = kfolder.testing(i)

            # Store the results
            self.X_train.append(X), self.Y_train.append(Y)
            self.X_test.append(Xi), self.Y_test.append(Yi), self.P.append((u,cov,p))

if __name__=="__main__":
    # Get cmdline args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help='Shared or separate coveriance matrices.')
    args = parser.parse_args(sys.argv[1:])
    data_file = "../data/spambase/spambase.data"
    booster = Booster(data_file)
    booster.train()
