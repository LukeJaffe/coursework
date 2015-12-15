import sys
import argparse
import numpy as np
from kfolder import KFolder
from smo2 import gram, tgram, train, test

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

# k-folds xvalidation
k = 10 
kfolder = KFolder(data, k, standard=True, shuffle=True)
for i in range(k-1):
    print "Fold:", i+1
    
    # Get data and labels at fold k
    X,Y = kfolder.testing(i+1)
    
    # Get the testing data
    Xi,Yi = kfolder.testing(i)
    Yi[Yi==0] = -1.0
    
    # Train
    Y[Y==0] = -1.0
    G, Gi = gram(X), tgram(X, Xi)
    a, b = train(X, Y.ravel(), G, C=1e-4, tol=1e-4, eps=1e-3)

    # Test
    print "Training accuracy:", test(Y, Y, G, a, b)
    print "Testing accuracy:", test(Y, Yi, Gi, a, b)
