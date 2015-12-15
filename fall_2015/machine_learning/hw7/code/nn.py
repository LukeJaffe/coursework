import numpy as np
from haar import timing
import scipy.spatial.distance as distance
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mode
from multiprocessing import Pool
import itertools

dist_func = None

def polyd2_distance(v1, v2):
    return -np.dot(v1,v2)**2

def gaussian_distance(v1, v2):
    sigma = 1.0
    return np.exp(-np.linalg.norm(v1-v2)**2 / (2 * (sigma ** 2)))

def linear_distance(v1,v2):
    return np.dot(v1,v2)

def cosine_distance(v1, v2):
    return distance.cosine(v1, v2)

def euclidean_distance(v1, v2):
    return np.linalg.norm(v2-v1)


def proc(food):
    Xi, Xj = food
    i,test = Xi
    j,train = Xj
    return i,j,dist_func(test, train)

def knn(train, test, Y, d=euclidean_distance, k=1):
    global dist_func
    dist_func = d
    H = np.zeros((len(test),len(train)))
    pool = Pool(processes=12)
    food = list(itertools.product(enumerate(test),enumerate(train)))
    result = pool.map(proc, food)
    for i,j,val in result:
        H[i][j] = val
    if d==euclidean_distance:
        idx = np.argsort(H, axis=1).T[:k]
    elif d==cosine_distance:
        idx = np.argsort(H, axis=1).T[:k]
    elif d==gaussian_distance:
        idx = np.argsort(H, axis=1).T[-k:]
    elif d==polyd2_distance:
        idx = np.argsort(H, axis=1).T[:k]
    topk = Y[idx]
    m = mode(topk)
    return m[0].ravel().astype(int)

def wnn(train, test, Y, d=euclidean_distance, r=2.5):
    global dist_func
    dist_func = d
    H = np.zeros((len(test),len(train)))
    pool = Pool(processes=12)
    food = list(itertools.product(enumerate(test),enumerate(train)))
    result = pool.map(proc, food)
    for i,j,val in result:
        H[i][j] = val

    # Count distances within radius
    R = np.zeros(len(test))
    empty = 0
    for i in range(len(test)):
        idx = np.where(H[i]<r)
        topr = Y[idx].ravel()
        if topr != []:
            m = mode(topr.ravel())
            R[i] = m[0][0]
        else:
            empty += 1
    print empty,"/",len(test)
    return R

def kde(train, test, Y, d=gaussian_distance):
    H = np.zeros(len(test))
    C = np.unique(Y)
    print C
    for i,x in enumerate(test):
        pA = 0.0
        pC = np.zeros(len(C))
        for c in C:
            idx = Y==c
            Z = train[idx]
            for z in Z:
                dist = d(x,z)
                pC[c] += dist
                pA += dist
        h = pC/pA
        H[i] = np.argmax(h)
    return H

def relief(X, Y, k=100, f=5, d=euclidean_distance):
    global dist_func

    # Build gram matrix for training set
    dist_func = d
    H = np.zeros((len(X),len(X)))
    pool = Pool(processes=12)
    food = list(itertools.product(enumerate(X),enumerate(X)))
    result = pool.map(proc, food)
    for i,j,val in result:
        H[i][j] = val
    idx = np.argsort(H, axis=1).T[:k]
    
    # Take closest data, labels from distance metric
    Z = X[idx]
    y = Y[idx]

    # Determine best features
    Q = np.zeros(len(X.T))
    for i,x in enumerate(X):
        for j,z in enumerate(Z):
            if Y[i] == y[j][i]:
                Q -= (x-z[i])**2
            else:
                Q += (x-z[i])**2

    # Take top f features
    F = np.argsort(Q)[-f:]
    return F
