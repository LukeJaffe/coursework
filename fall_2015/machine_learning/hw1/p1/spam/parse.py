from bitstring import BitArray
from math import log
from operator import itemgetter
from collections import namedtuple
import time
import sys


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


def get_features(fname):
    feature_list = []
    f = open(fname, "r")
    for line in f:
        if line.find('|') == -1:
            if line.strip():
                line = line.split(':')
                feature = line[0]
                desc = line[1] 
                desc = desc.replace(" ","")
                desc = desc.split('.')[0]
                feature_list.append(feature)
    return feature_list


# Create dictionary with features as keys
def get_data(fname, feature_list):
    dmat = []
    fmat = {}
    for feature in feature_list:
        fmat[feature] = []

    Y = ""
    f = open(fname, "r")
    for line in f:
        x = line.split(',')
        Y += x[-1].strip()
        x = [float(e) for e in x]
        dmat.append(x[:-1])
        for i in range(len(x)-1):
            x[i] = float(x[i])
            fmat[feature_list[i]].append(x[i])
    Y = BitArray(bin=Y)
    return fmat,dmat,Y


class MLT:
    """ Machine Learning Tree class """

    def __init__(self, type, feature_list, fmat, dmat, Y, elem_thresh):
        if type == 'decision':
            self.type = 'decision'
        elif type == 'regression':
            self.type = 'regression'
        else:
            raise Exception("Cannot make an MLT of this type.")

        self.feature_list = feature_list
        self.fmat = fmat
        self.dmat = dmat
        self.Y = Y
        self.elem_thresh = elem_thresh
        self.root = BitArray(bin=len(self.dmat)*'1')

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

    def eval_node(self, A):
        IG_list = []
        A_count = float(A.count(True))
        p_A = self.prob(A)
        e_A = self.entropy(p_A)
        if p_A >= 0.5:
            d = 1
        else:
            d = 0

        print "#elem:",A_count
        if A_count < self.elem_thresh:
            print "Node has few elements."
            return None,None,d,None,None

        for f in self.feature_list:
            # Sort the relevant column and keep indices
            pairs = [sorted(enumerate(self.fmat[f]), key=itemgetter(1))][0]
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
            print "Boring decision..."
            return None,None,d,None,None
        else:
            max_val = max(IG[0] for IG in IG_list)
            IG,f,t,d,B,C,s_B,s_C = [IG for IG in IG_list if IG[0] == max_val][0]
            #if s_B < 0.1:
            #    B = None
            #if s_C < 0.1:
            #    C = None 
            #print s_B,s_C
            return f,t,d,B,C

    @timing
    def build_tree(self, depth):
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
                    f,t,d,B,C = self.eval_node(btree[j])
                    #print f,t,d
                    lc,rc = 2*j+1, 2*j+2
                    if B is None or (lc >= (2**depth)-1):
                        lc = None
                    if C is None or (rc >= (2**depth)-1):
                        rc = None
                    etree.append(TestNode(f=f, t=t, d=d, lc=lc, rc=rc)) 
                    btree.append(B)
                    btree.append(C)
        self.etree = etree

    def eval_data(self, d):
        c = 0
        while True:
            q = self.etree[c] 
            if q.f is None:
                return q.d
            i = self.feature_list.index(q.f)
            if d[i] < q.t:
                c = q.lc
                #print "Less:",i, d[i], q.t, c
            else:
                c = q.rc
                #print "Greater:",i, d[i], q.t, c
            if c is None:
                return q.d
            #print c

    def eval_tree(self):
        total = len(self.dmat)
        correct = 0
        for i in range(total):
            if self.eval_data(self.dmat[i]) == self.Y[i]:
                correct += 1
        print "Correct:",str(correct)+str('/')+str(total)+' =',float(correct)/float(total)



if __name__=="__main__":
    # Get feature list
    fname1 = ("/home/jaffe5/Documents/classes/fall_2015/"+
                "machine_learning/hw1/data/spambase/spambase.names")
    feature_list = get_features(fname1)

    # Get data
    fname2 = ("/home/jaffe5/Documents/classes/fall_2015/"
                +"machine_learning/hw1/data/spambase/spambase.data")
    fmat,dmat,Y = get_data(fname2, feature_list)

    depth = 3
    thresh = 100

    mlt = MLT('decision', feature_list, fmat, dmat, Y, thresh)
    mlt.build_tree(depth)
    mlt.eval_tree()
