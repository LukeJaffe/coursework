import sys
import argparse
import numpy as np
from kfolder import KFolder
from svmutil import *


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

# k-folds shit
k = 10 
kfolder = KFolder(data, k, standard=True, shuffle=False)
for i in range(1):
    print "Fold:", i+1
    # Get data and labels at fold k
    X,Y = kfolder.training(i)
    # Get the testing data
    Xi,Yi = kfolder.testing(i)
    # Train
    px = svm_problem(Y.ravel().tolist(), X.tolist())
    pm = svm_parameter()
    pm.kernel_type = kernel
    #pm.C = 10
    m = svm_train(px, pm)
    # Test
    p_label, p_acc, p_val = svm_predict(Y.ravel().tolist(), X.tolist(), m) 
    h_label, h_acc, h_val = svm_predict(Yi.ravel().tolist(), Xi.tolist(), m) 
    # Eval
    ACC, MSE, SCC = evaluations(Y.ravel().tolist(), p_label)
    ACC, MSE, SCC = evaluations(Yi.ravel().tolist(), h_label)
