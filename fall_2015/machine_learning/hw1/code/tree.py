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
                        IG_list.append((IG_curr,f,t,d,B,C,s_B,s_C))
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


class RegressionTree:
    def __init__(self, X, Y, count_thresh=0):
        self.X = X
        self.Y = Y
        self.count_thresh = count_thresh
        self.root = BitArray(bin=len(self.X)*'1')

    # Predicted value at node A
    def mean(self, A):
        t = float(A.count(True))
        s = sum(self.Y[np.array(A) == 1])
        if t == 0.0:
            return 0.0
        else:
            return s/t

    # Total square error (no normalization)
    def tse(self, A):
        m = self.mean(A)
        e = self.Y[np.array(A) == 1] 
        return sum((e-m)**2.0)
    
    # Mean square error (normalize by # elements at split node)
    def mse(self, N):
        s, t = 0.0, 0.0
        for n in N:
            t += float(n.count(True))
            s += self.tse(n)
        return s/t

    def split(self, A):
        gain_list = []
        A_count = float(A.count(True))
        d = self.mean(A)

        if A_count <= self.count_thresh:
            print "Node has few elements:", A_count
            return None,None,d,None,None
        mse_A = self.mse([A])
        #print "Curr mse:",mse_A

        F = self.X.T
        for f in range(len(F)):
            #print f
            # Sort the relevant column and keep indices
            indices = F[f].argsort()
            pairs = zip(indices, F[f][indices])
            s = len(pairs)*'0'
            B = BitArray(bin=s)
            C = A.copy()
            i = 0
            gain_prev = 0.0
            # Threshold emulator loop
            while i < len(pairs)-1:
                if A[pairs[i][0]]:
                    B[pairs[i][0]] = 1
                    C[pairs[i][0]] = 0
                    if pairs[i][1] < pairs[i+1][1]:
                        t = pairs[i+1][1]
                        # Calculate MSE for the split
                        mse_curr = self.mse([B,C])
                        gain_curr = mse_A - mse_curr
                        if gain_curr < 0:
                            print gain_curr
                        #print mse_curr
                        if gain_curr < gain_prev:
                            break
                        else:
                            gain_prev = gain_curr
                        # Check if entropy for any branches is below thresh
                        gain_list.append((gain_curr,f,t,d,B,C,mse_A,mse_curr))
                i += 1

        #if B.count(True) == 0 or C.count(True) == 0:
        #    print "Count of B or C = 0"
        #    return None,None,d,None,None
        #if self.mse([B]) > mse_A or self.mse([C]) > mse_A:
        #    print "MSE of split greater than parent"
        #    return None,None,d,None,None
        if gain_list == []:
            print "mse_list empty"
        #    return None,None,d,None,None
        else:
            max_val = max(mse[0] for mse in gain_list)
            gain,f,t,d,B,C,mse_A,mse_curr = [mse for mse in gain_list if mse[0] == max_val][0]
            if mse_curr <= 1.0:
                return None,None,d,None,None
            else:
                print "Gain:",mse_A,mse_curr
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
                #print "Less:",i, d[i], q.t, c
            else:
                c = q.rc
                #print "Greater:",i, d[i], q.t, c
            if c is None:
                return q.d
            #print c

    def eval_tree(self, X, Y, T):
        total = len(X)
        correct = 0
        for i in range(total):
            if self.eval_data(X[i], T) == Y[i]:
                correct += 1
        print "Correct:",str(correct)+str('/')+str(total)+' =',float(correct)/float(total)
        return float(correct)/float(total)

    def train(self):
        k = 10
        kfolder = KFolder(self.D, k, normalize=False)
        self.X_train, self.Y_train = [], []
        self.X_test, self.Y_test, self.T = [], [], []
        for i in range(1):#k
            # Get data and labels at fold k
            X,Y = kfolder.training(i)

            # Build the decision tree
            dt = DecisionTree(X, Y)
            Ti = dt.build(6)

            # Get the testing data
            Xi,Yi = kfolder.testing(i)

            # Store the results
            self.X_train.append(X), self.Y_train.append(Y)
            self.X_test.append(Xi), self.Y_test.append(Yi), self.T.append(Ti)

    def test(self):
        # Training error
        train_avg = 0.0
        for i in range(1):#range(len(self.T)):
            X, Y, T = self.X_train[i], self.Y_train[i], self.T[i]
            train_avg += self.eval_tree(X, Y, T)
        print "Training error: %f correct" % (train_avg/float(len(self.T))) 
        # Testing error
        test_avg = 0.0
        for i in range(1):#range(len(self.T)):
            X, Y, T = self.X_test[i], self.Y_test[i], self.T[i]
            test_avg += self.eval_tree(X, Y, T)
        print "Testing error: %f correct" % (test_avg/float(len(self.T)))


class HousingLearner:
    def __init__(self, train_file, test_file):
        self.X_train, self.Y_train = self.get_data(train_file)
        self.X_test, self.Y_test = self.get_data(test_file)
        print "Training:", self.X_train.shape, self.Y_train.shape
        print "Testing:", self.X_test.shape, self.Y_test.shape
        
    def get_data(self, data_file):
        X, Y = [], []
        f = open(data_file, "r")
        for line in f:
            if line.strip():
                x = line.split()
                x = [float(e) for e in x]
                X.append( x[:-1] )
                Y.append( x[-1:] )
        return np.array(X), np.array(Y)

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

    def train(self):
        rt = RegressionTree(self.X_train, self.Y_train)
        self.T = rt.build(5)

    def test(self):
        # Training error
        total = len(self.X_train)
        mse = 0.0
        for i in range(total):
            #print (self.eval_data(self.X_train[i], self.T) - self.Y_train[i])**2.0,
            mse += ((self.eval_data(self.X_train[i], self.T) - self.Y_train[i])**2.0) 
        #print "unnormalized:",mse
        mse /= float(total)
        print "Training MSE:",mse
        # Testing error
        total = len(self.X_test)
        mse = 0.0
        for i in range(total):
            mse += ((self.eval_data(self.X_test[i], self.T) - self.Y_test[i])**2.0)
        mse /= float(total)
        print "Testing MSE:",mse


if __name__=="__main__":
    # Get cmdline args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Run regression on this dataset.')
    args = parser.parse_args(sys.argv[1:])
    if args.d is not None:
        if args.d == 'spam':
            data_file = "../data/spambase/spambase.data"
            sl = SpamLearner(data_file)
            sl.train()
            sl.test()
        elif args.d == 'housing':
            train_file = "../data/housing/housing_train.txt"
            test_file = "../data/housing/housing_test.txt"
            hl = HousingLearner(train_file, test_file)
            hl.train()
            hl.test()
        else:
            print "Unknown dataset."
            sys.exit() 
    else:
        print "No dataset given."
