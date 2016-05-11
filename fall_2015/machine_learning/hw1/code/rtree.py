import sys
import time
import argparse
import numpy as np
from math import exp, log
from operator import itemgetter
from collections import namedtuple
from bitstring import BitArray

from kfolder import KFolder

glob = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])

TestNode = namedtuple('Node', [ 'f',    # Feature
                                't',    # Threshold
                                'd',    # Decision
                                'lc',   # Left child
                                'rc'])  # Right child


class RegressionTree:
    def __init__(self, X, Y, count_thresh=25):
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
            #print "Node has few elements:", A_count
            return None,None,d,None,None
        mse_A = self.mse([A])
        #if mse_A <= 15:
        #    print "Node has small MSE:", mse_A
        #    return None,None,d,None,None
        print "Elem, MSE:",A_count,mse_A

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
                            pass
                            #break
                        else:
                            gain_prev = gain_curr
                        # Check if entropy for any branches is below thresh
                        if B.count(True) == 0 or C.count(True) == 0:
                            pass
                        else:
                            gain_list.append((gain_curr,f,t,d,B.copy(),C.copy(),mse_A,mse_curr))
                i += 1

        if gain_list == []:
            print "mse_list empty"
            return None,None,d,None,None
        else:
            max_val = max(mse[0] for mse in gain_list)
            gain,f,t,d,B,C,mse_A,mse_curr = [mse for mse in gain_list if mse[0] == max_val][0]
            if B.count(True) == 0 or C.count(True) == 0:
                print "Count of B or C = 0:",B.count(True), C.count(True)
                for elem in gain_list:
                    print elem[4].count(True), elem[5].count(True)
                print
                return None,None,d,None,None
            else:
                return f,t,d,B,C

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
        d = 0
        global glob
        flist = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
        while True:
            q = T[c] 
            if q.f is None:
                d = q.d
                break
            flist[q.f] += 1
            if X[q.f] < q.t:
                c = q.lc
            else:
                c = q.rc
            if c is None:
                d = q.d
                break
        glob += flist
        return d

    def train(self, depth):
        rt = RegressionTree(self.X_train, self.Y_train)
        self.T = rt.build(depth)

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
        train_file = "../data/housing/housing_train.txt"
        test_file = "../data/housing/housing_test.txt"
        hl = HousingLearner(train_file, test_file)
        hl.train(int(args.d))
        hl.test()
        #for i in range(len(glob)):
        #    print i,":",glob[i]
    else:
        print "No dataset given."

