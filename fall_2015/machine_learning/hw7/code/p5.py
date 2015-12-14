import sys
import argparse
import numpy as np
from kfolder import KFolder
from svmutil import *

from nn import knn, relief


parser = argparse.ArgumentParser()
parser.add_argument('-k', help='Kernel')
args = parser.parse_args(sys.argv[1:])

data_file = "../data/spambase/spambase.data"
dmat = []
f = open(data_file, "r")
for line in f:
    x = line.split(',')
    x = [float(e) for e in x]
    dmat.append(x)
data = np.array(dmat)

# k-folds 
k = 10 
kfolder = KFolder(data, k, standard=True, shuffle=False)
for i in range(1):
    # Get data and labels at fold k
    X,Y = kfolder.training(i)
    
    # Get the testing data
    Xi,Yi = kfolder.testing(i)

    # Get top 5 features with relief
    F = relief(X, Y)
    print "Features selected:",F

    # Take features suggested by relief
    X, Xi = X.T[F].T, Xi.T[F].T

    # Run knn
    for j in [1]:
        H = knn(X, Xi, Y, k=j)
        c = np.sum(Yi.ravel()==H)
        print "k=%d:" % j, float(c)/float(len(Yi))
