import sys
import argparse

from multiprocessing import Pool
import numpy as np
from sklearn import preprocessing

from load import load_mnist
from haar import sample, genbounds, genfeatures, evaluate
from smo2 import gram, dual, train, hypothesis


def proc((c,code)):
    print c,code
    coded_labels = train_labels.copy()
    
    # Create the code-mapped label vector
    for i, label in enumerate(train_labels):
        coded_labels[i] = code[int(label)]
    
    # SVM for this code
    w, b = train(train_data, coded_labels, C=0.00001, tol=0.00001, eps=1e-2)

    # Hypothesis
    h_train = hypothesis(train_data, w, b)
    h_test = hypothesis(test_data, w, b)

    return c, h_train, h_test


class Model:
    def __init__(self, i, j, G):
        self.i = i
        self.j = j
        self.G = G

    def train(self, X, Y):
        # Get relevant data
        idx = Y==self.i
        jdx = Y==self.j
        ijdx = np.logical_or(idx, jdx)
        x = X[ijdx]
        y = Y[ijdx]
        # Project y onto {-1, +1}
        y[y==self.i] = -1.0
        y[y==self.j] = 1.0
        self.Y = y
        self.G = self.G[ijdx]
        G = self.G.T[ijdx].T
        # Train SMO
        self.a, self.b = train(x, y, G, C=1e-5, tol=1e-2, eps=1e-3)

    def test(self, i):
        #print self.G.shape, self.Y.shape
        h = dual(self.G, self.Y, self.a, self.b, i)
        if h < 0:
            return self.i
        else:
            return self.j
       
 
if __name__=="__main__":
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help='ECOC or VOTE')
    args = parser.parse_args(sys.argv[1:])

    # Load data using specialized script
    train_dataset = load_mnist(path="../data/mnist/", dataset="training")
    test_dataset = load_mnist(path="../data/mnist/", dataset="testing")
    
    # Take a fraction of the data to speed computation
    train_images, train_labels = sample(train_dataset, 5000)
    test_images, test_labels = sample(test_dataset, 100)

    # Get the bounds of the haar rectangles
    bounds = genbounds(28, 28, 100)
    
    # Create data, using same rectangles for training and testing
    train_data = genfeatures(train_images, bounds).astype(float)
    test_data = genfeatures(test_images, bounds).astype(float)

    # Normalize the data
    zmscaler = preprocessing.StandardScaler()
    train_data = zmscaler.fit_transform(train_data)
    test_data = zmscaler.transform(test_data)
    
    if args.m == 'ECOC':
        # Generate 50 random ECOC vectors
        codes = np.random.randint(0,2,size=(20,10))
        train_codes, test_codes = codes.copy(), codes.copy().astype(bool)
        train_codes[train_codes==0] = -1.0

        # Iterate through each ECOC
        pool = Pool(processes=11)
        result = pool.map(proc, enumerate(train_codes))
        ordered = zip(*sorted(result))
        H_train = ordered[1]
        H_test = ordered[2]
        #print H_train

        # Compare distance of code to codes and classify
        print "Training accuracy:", evaluate(H_train, train_labels, test_codes)
        print "Testing accuracy:", evaluate(H_test, test_labels, codes)
    elif args.m == 'VOTE':
        # Run SMO on each label pair {i=-1, j=+1}
        NUM_LABELS = 10
        m = len(train_data)
        models = {}

        G = gram(train_data)

        # Simple voting, no tie-resolution    
        votes_simple = np.zeros((m, NUM_LABELS))
        for i in range(NUM_LABELS):
            models[i] = {}
            for j in range(i+1, NUM_LABELS):
                print i,j
                model = Model(i, j, G)
                models[i][j] = model
                model.train(train_data, train_labels)
                for k in range(m):
                    v = model.test(k)
                    votes_simple[k][v] += 1
        ties = 0
        h_simple = np.zeros(m)
        for k in range(m):
            if len(np.argwhere(votes_simple[k] == np.amax(votes_simple[k]))) >= 2:
                ties += 1
            h_simple[k] = np.argsort(votes_simple[k])[-1:]    
        for i,thing in enumerate(votes_simple):
            print thing,train_labels[i], h_simple[i]==train_labels[i]
        print votes_simple.shape
        print "Ties:",ties,"/",m

        # Voting with tie-resolution
        #h_complex = np.zeros(m)
        #for k in range(m):
        #    vote = np.zeros(NUM_LABELS)
        #    for i in range(NUM_LABELS):
        #        for j in range(i+1, NUM_LABELS):
        #            model = models[i][j]
        #            v = model.test(train_data[k]) 
        #            vote[v] += 1
        #    # Tie-resolution
        #    max_votes = np.argwhere(vote == np.amax(vote)).ravel()
        #    if len(max_votes) == 2:
        #        i, j = max_votes[0], max_votes[1] 
        #        model = models[i][j]
        #        v = model.test(train_data[k])
        #        h_complex[k] = v 
        #    else:
        #        h_complex[k] = np.argsort(votes_simple[k])[-1:]    

        # Evaluation
        c_simple = np.sum(h_simple==train_labels)
        print "Simple  accuracy:",float(c_simple)/float(m)
        #c_complex = np.sum(h_complex==train_labels)
        #print "Complex accuracy:",float(c_complex)/float(m)
    else:
        print "Invalid method. Try {ECOC, VOTE}"
