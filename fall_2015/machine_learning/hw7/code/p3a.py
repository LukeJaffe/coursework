import sys
import numpy as np

from data import Data
from perceptron import preprocess, primal, dual
from nn import linear_distance, gaussian_distance


if __name__=="__main__":
    data_file = "../data/perceptron/perceptronData.txt"
    data = Data(data_file)
    if sys.argv[1] == 'p':
        data = preprocess(data)
        X,Y = data.data(), data.labels()
        primal(X, Y, r=0.001)
    elif sys.argv[1] == 'd':
        X,Y = data.data(), data.labels()
        dual(X, Y, K=linear_distance)
    else:
        print "Invalid input argument."
