import sys
import argparse
import numpy as np
from kfolder import KFolder
from smo import train, test

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
kfolder = KFolder(data, k, standard=True, shuffle=False)
for i in range(1):
    print "Fold:", i+1
    
    # Get data and labels at fold k
    X,Y = kfolder.training(i)
    
    # Get the testing data
    Xi,Yi = kfolder.testing(i)
    Yi[Yi==0] = -1.0
    
    # Train
    Y[Y==0] = -1.0
    w, b = train(X, Y.ravel())

    # Test
    print "Training accuracy:", test(X, Y, w, b)
    print "Testing accuracy:", test(Xi, Yi, w, b)
