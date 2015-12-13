import sys
import argparse
import numpy as np
from kfolder import KFolder
from svmutil import *

from knn import knn

parser = argparse.ArgumentParser()
parser.add_argument('-k', help='Kernel')
args = parser.parse_args(sys.argv[1:])

if args.k == "p":
        kernel = POLY
elif args.k == "l":
        kernel = LINEAR
elif args.k == "r":
        kernel = RBF

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
    print "Fold:", i+1
    # Get data and labels at fold k
    X,Y = kfolder.training(i)
    # Get the testing data
    Xi,Yi = kfolder.testing(i)

    # Run knn
    for j in [1,2,3]:
        H = knn2(X, Xi, Y, k=j)
        c = np.sum(Yi.ravel()==H)
        print "k=%d:" % j, float(c)/float(len(Yi))
