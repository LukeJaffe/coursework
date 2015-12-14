import sys
import numpy as np
from perceptron import preprocess, primal, dual
from nn import linear_distance, gaussian_distance

X, Y = [], []
fname = "../data/spiral/twoSpirals.txt"
f = open(fname, "r")
for line in f:
    sline = line.split()
    fline = [float(e) for e in sline]
    X.append(fline[:-1])
    Y.append(fline[-1:])
X, Y = np.array(X), np.array(Y)
print X.shape, Y.shape

if sys.argv[1] == 'p':
    primal(X, Y, r=0.001)
elif sys.argv[1] == 'l':
    dual(X, Y, K=linear_distance)
elif sys.argv[1] == 'g':
    dual(X, Y, K=gaussian_distance)
else:
    print "Invalid kernel."
