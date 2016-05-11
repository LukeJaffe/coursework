import sys
import time
import math
import numpy as np
from multiprocessing import Pool

from load import load_mnist
from booster import Booster

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

def sample(data, n):
    return data[0][:n], data[1][:n]

def randint(b1, b2):
    if b1 == b2:
        return b1
    else:
        r =  np.random.randint(b1, b2)
        if r > 28:
            r = 28
        return r

def genbounds(W, H, n):
    bounds = []
    for i in range(n):
        w = randint(5,W+1)
        h = randint(math.ceil(130./w), 170/w+1)
        x1 = randint(0, W-w)
        x2 = x1+w
        y1 = randint(0, H-h)
        y2 = y1+h
        bounds.append((x1,x2,y1,y2,w,h))
    return bounds

@timing
def genrects(images, bounds):
    rects = []
    for i,im in enumerate(images):
        rects.append([]) 
        for b in bounds:
            x1, x2, y1, y2, w, h = b
            ax1, ax2, ay1, ay2 = x1, x1+w/2, y1, y2
            bx1, bx2, by1, by2 = x1+w/2+1, x2, y1, y2
            cx1, cx2, cy1, cy2 = x1, x2, y1, y1+h/2
            dx1, dx2, dy1, dy2 = x1, x2, y1+h/2+1, y2
            a = im[ay1:ay2, ax1:ax2]
            b = im[by1:by2, bx1:bx2]
            c = im[cy1:cy2, cx1:cx2]
            d = im[dy1:dy2, dx1:dx2]
            rects[i].append((a,b))
            rects[i].append((c,d))
    return rects


@timing
def genhaar(rects):
    features = []
    for i,im in enumerate(rects): 
        features.append([])
        for a,b in im:
            s = np.count_nonzero(a) - np.count_nonzero(b)
            features[i].append(s)
    return np.array(features)


def proc((c,code)):
    print c,code
    coded_labels = train_labels.copy()
    # Create the code-mapped label vector
    for i, label in enumerate(train_labels):
        coded_labels[i] = code[int(label)]
    # Boost for this code
    booster = Booster() 
    h_train, h_test = booster.boost(train_data, coded_labels, test_data, booster.thresh(train_data), 50) 
    return c, h_train, h_test


def evaluate(H, Y, codes):
    H = np.array(H).T
    G = np.zeros(len(H), dtype=int)
    for i,h in enumerate(H):
        dist = []
        for c, code in enumerate(zip(*codes)):
            dist.append( len((h.astype(bool)==np.array(code)).nonzero()[0]) )
        G[i] = dist.index(min(dist))
    total = len(H)
    correct = len((G==Y).nonzero()[0])
    return float(correct)/float(total)


def genfeatures(images, bounds):
    # Collect the haar rectangles
    rects = genrects(images, bounds) 
    # Generate the haar features with the rectangles
    data = genhaar(rects)
    return data


if __name__=="__main__":
    # Load data using specialized script
    train_dataset = load_mnist(path="../data/mnist/", dataset="training")
    test_dataset = load_mnist(path="../data/mnist/", dataset="testing")
    # Take a fraction of the data to speed computation
    train_images, train_labels = sample(train_dataset, 1000)
    #for i in range(10):
    #    print len(train_labels[train_labels==i])
    test_images, test_labels = test_dataset

    # Get the bounds of the haar rectangles
    bounds = genbounds(28, 28, 100)
    # Create data, using same rectangles for training and testing
    train_data = genfeatures(train_images, bounds)
    test_data = genfeatures(test_images, bounds)
    # Generate 50 random ECOC vectors
    codes = np.random.randint(0,2,size=(11,10)).astype(bool)

    # Iterate through each ECOC
    pool = Pool(processes=11) 
    result = pool.map(proc, enumerate(codes))
    ordered = zip(*sorted(result))
    H_train = ordered[1]
    H_test = ordered[2]
    #print H_train

    # Compare distance of code to codes and classify
    print "Training accuracy:", evaluate(H_train, train_labels, codes)
    print "Testing accuracy:", evaluate(H_test, test_labels, codes)
