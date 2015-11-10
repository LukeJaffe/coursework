import sys
import time
import argparse
import numpy as np
from math import exp, log
from operator import itemgetter
from collections import namedtuple
from bitstring import BitArray

from kfolder import KFolder


TestNode = namedtuple('Node', [ 'f',    # Feature
                                't',    # Threshold
                                'd',    # Decision
                                'lc',   # Left child
                                'rc'])  # Right child

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap


class DecisionTree:
    def __init__(self, X, Y, count_thresh=100):
        self.X = X
        self.Y = Y
        self.count_thresh = count_thresh
        self.root = BitArray(bin=len(self.X)*'1')

    def prob(self, A):
        if A.count(True) > 0.0:
            return float((A&self.Y).count(True))/float(A.count(True))
        else:
            return 0.0

    def entropy(self, p):
        if 0.0 < p < 1.0:
            return -(p*log(p, 2) + (1-p)*log((1-p), 2))
        else:
            return 0.0

    def info_gain(self, B, C, e_A, A_count):
        # Calculate probability for child nodes
        p_B = self.prob(B)
        p_C = self.prob(C)
        # Calculate entropy for child nodes
        e_B = self.entropy(p_B)
        e_C = self.entropy(p_C)
        # Calculate scaling factors for child nodes entropy
        k_B = float(B.count(True))/A_count
        k_C = float(C.count(True))/A_count
        # Calculate scaled entropy for child nodes
        s_B = k_B*e_B
        s_C = k_C*e_C
        # Calculate scaled entropy sum for child nodes
        e_S = s_B+s_C
        IG = e_A - e_S
        return IG,s_B,s_C

    def split(self, A):
        IG_list = []
        A_count = float(A.count(True))
        p_A = self.prob(A)
        e_A = self.entropy(p_A)
        # Get decision for node
        if p_A >= 0.5:
            d = 1
        else:
            d = 0
        # Check entropy of node
        if e_A < 0.3:
            print "Node entropy is low enough:", e_A
            return None,None,d,None,None
        # Check # elements
        if A_count < self.count_thresh:
            print "Node has few elements:", A_count
            return None,None,d,None,None

        F = self.X.T
        for f in range(len(F)):
            # Sort the relevant column and keep indices
            indices = F[f].argsort()
            pairs = zip(indices, F[f][indices])
            s = len(pairs)*'0'
            B = BitArray(bin=s)
            C = A.copy()
            i = 0
            IG_prev = 0.0
            # Threshold emulator loop
            while i < len(pairs)-1:
                if A[pairs[i][0]]:
                    B[pairs[i][0]] = 1
                    C[pairs[i][0]] = 0
                    if pairs[i][1] < pairs[i+1][1]:
                        t = pairs[i+1][1]
                        # Calculate information gain for the split
                        IG_curr,s_B,s_C = self.info_gain(B,C,e_A,A_count)
                        #print IG_curr
                        if IG_curr < IG_prev:
                            break
                        else:
                            IG_prev = IG_curr
                        # Check if entropy for any branches is below thresh
                        IG_list.append((IG_curr,f,t,d,B.copy(),C.copy(),s_B,s_C))
                i += 1

        if IG_list == []:
            #print "Boring decision..."
            return None,None,d,None,None
        else:
            max_val = max(IG[0] for IG in IG_list)
            IG,f,t,d,B,C,s_B,s_C = [IG for IG in IG_list if IG[0] == max_val][0]
            return f,t,d,B,C

    @timing
    def build(self, depth):
        btree = [self.root]
        etree = []
        for i in range(depth):
            index = 2**i
            print "Tree level:",i+1
            for j in range(index-1, 2*index-1):
                if btree[j] is None:
                    btree.append(None)
                    btree.append(None)
                    etree.append(None)
                else:
                    f,t,d,B,C = self.split(btree[j])
                    #print f,t,d
                    lc,rc = 2*j+1, 2*j+2
                    if B is None or (lc >= (2**depth)-1):
                        lc = None
                    if C is None or (rc >= (2**depth)-1):
                        rc = None
                    etree.append(TestNode(f=f, t=t, d=d, lc=lc, rc=rc)) 
                    btree.append(B)
                    btree.append(C)
        return etree


class SpamLearner:
    def __init__(self, data_file):
        dmat = []
        f = open(data_file, "r")
        for line in f:
            x = line.split(',')
            x = [float(e) for e in x]
            dmat.append(x)
        self.D = np.array(dmat)

    def eval_data(self, X, T):
        c = 0
        while True:
            q = T[c] 
            if q.f is None:
                return q.d
            if X[q.f] < q.t:
                c = q.lc
            else:
                c = q.rc
            if c is None:
                return q.d

    def eval_tree(self, X, T):
        total = len(X)
        h = np.zeros(total)
        for i in range(total):
            h[i] = self.eval_data(X[i], T)
        return h

    def train(self, depth, bags):
        k = 10
        kfolder = KFolder(self.D, k, normalize=False)
        self.T = []
        # Get data and labels at fold k
        self.X,self.Y = kfolder.training(0)
        C = np.append(self.X,self.Y,axis=1)
        n = len(C)
        for i in range(bags):
            print "Bag #", i+1
            # Sample a bag
            idx = np.random.randint(n, size=n)
            Xi, Yi = C.T[:-1].T[idx], C.T[-1:].T[idx].ravel()
            # Build the decision tree
            dt = DecisionTree(Xi, Yi)
            Ti = dt.build(depth)
            self.T.append(Ti)
        # Get the testing data
        self.Xt,self.Yt = kfolder.testing(0)

    def test(self, bags):
        # Training error
        H = np.zeros(len(self.Y))
        for i in range(bags):
            H += self.eval_tree(self.X, self.T[i])
        H /= float(bags)
        correct = sum((H.round()==self.Y.ravel()).astype(int))
        print "Training accuracy: %f" % (float(correct)/float(len(self.Y))) 
        # Testing error without bagging
        H = self.eval_tree(self.Xt, self.T[i])
        correct = sum((H.round()==self.Yt.ravel()).astype(int))
        print "Testing accuracy (no bagging): %f" % (float(correct)/float(len(self.Yt))) 
        # Testing error with bagging
        H = np.zeros(len(self.Yt))
        for i in range(bags):
            H += self.eval_tree(self.Xt, self.T[i])
        H /= float(bags)
        correct = sum((H.round()==self.Yt.ravel()).astype(int))
        print "Testing accuracy (bagging): %f" % (float(correct)/float(len(self.Yt))) 

if __name__=="__main__":
    # Get cmdline args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Depth of decision tree.')
    parser.add_argument('-b', help='Number of rounds of bagging.')
    args = parser.parse_args(sys.argv[1:])
    if args.d is not None:
        depth = int(args.d)
    else:
        depth = 5
    if args.b is not None:
        bags = int(args.b)
    else:
        bags = 50
    data_file = "../data/spambase/spambase.data"
    sl = SpamLearner(data_file)
    sl.train(depth, bags)
    sl.test(bags)
