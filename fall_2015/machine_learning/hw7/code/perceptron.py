import numpy as np
from nn import linear_distance, gaussian_distance

def preprocess(data):
    for i,y in enumerate(data.labels()):
        if y < 0:
            data.data()[i] = -data.data()[i]
            data.labels()[i] = -data.labels()[i]
    return data

def primal(X, Y, r=0.001):
    W = np.zeros(len(X.T))+1e-5
    total_mistake = 1
    iterations = 0
    while total_mistake > 0:
        iterations += 1
        M = []
        for x in X:
            if np.dot(x,W) < 0:
                M.append(x)
        total_mistake = len(M)
        print "Iteration %d, total_mistake %d" % (iterations, total_mistake)
        s = 0.0
        for m in M:
            s += m
        W += r*s
    #print iterations 
    print "\nClassifier weights:",W
    print "\nNormalized with threshold:", -W/W[0]

def dual(X, Y, K=linear_distance):
    m = np.zeros(len(X))
    total_mistake = 1
    iterations = 0
    while total_mistake > 0:
        total_mistake = 0
        iterations += 1
        for j,xi in enumerate(X):
            d = 0.0
            for i,xj in enumerate(X):
                d += m[i]*K(xi,xj)
            if Y[j]*d <= 0:
                m[j] += Y[j]
                total_mistake += 1
        print "Iteration %d, total_mistake %d" % (iterations, total_mistake)
    #print iterations 
    w = np.zeros(len(X.T))
    for i,x in enumerate(X.T):
        w[i] = np.dot(m, x)
    print "\nClassifier weights:",w
    print "\nNormalized with threshold:", -w/w[0]
