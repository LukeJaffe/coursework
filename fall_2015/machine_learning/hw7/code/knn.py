import numpy as np
from haar import timing
import scipy.spatial.distance as distance
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mode
from multiprocessing import Pool
import itertools

dist_func = None

def polyd2_distance(v1, v2):
    return 1/np.dot(v1,v2)**2

def gaussian_distance(v1, v2):
    n = np.dot(v1,v2)
    s = 1.0
    d = np.exp(-n/s)
    return d

def linear_distance(v1,v2):
    return -np.dot(v1,v2)

def cosine_distance(v1, v2):
    return distance.cosine(v1, v2)

def euclidean_distance(v1, v2):
    return np.linalg.norm(v2-v1)


def knn(train, test, Y, d=euclidean_distance, k=1):
    H = np.zeros(len(test))
    for i in range(len(test)):
        dvec = np.zeros(len(train))
        for j in range(len(train)):
            dvec[j] = d(test[i], train[j])
        idx = np.argsort(dvec)[:k]
        topk = Y[idx]
        H[i] = mode(topk)[0][0]
    return H

def proc1(food):
    Xi, Xj = food
    i,test = Xi
    j,train = Xj
    return i,j,dist_func(test, train)

def knn2(train, test, Y, d=euclidean_distance, k=1):
    global dist_func
    dist_func = d
    H = np.zeros((len(test),len(train)))
    pool = Pool(processes=12)
    food = list(itertools.product(enumerate(test),enumerate(train)))
    result = pool.map(proc1, food)
    for i,j,val in result:
        H[i][j] = val
    idx1 = np.argmin(H, axis=1)
    idx2 = np.argsort(H, axis=1).T[:k]
    topk = Y[idx2]
    m = mode(topk)
    return m[0].ravel().astype(int)
    #return Y[idx1]

def proc2(food):
    h = []
    for i in range(len(food)):
        ((i,test),(j,train)) = food[i]
        h.append((i,j,polyd2_distance(test, train)))
    return h

def knn3(train, test, Y, d=euclidean_distance, k=1, num_proc=12):
    H = np.zeros((len(test),len(train)))
    pool = Pool(processes=num_proc)
    food1 = list(itertools.product(enumerate(test),enumerate(train)))
    food2 = [[] for i in range(num_proc)]
    for i in range(len(food1)):
        food2[i%12].append(food1[i])
    result = pool.map(proc2, food2)
    for t in range(num_proc):
        for i,j,val in result[t]:
            H[i][j] = val
    idx = np.argmin(H, axis=1)
    return Y[idx]
