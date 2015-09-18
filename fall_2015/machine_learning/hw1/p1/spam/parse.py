from bitstring import BitArray
from math import log
from operator import itemgetter
from collections import namedtuple
import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

feature_list = []

fname1 = "/home/jaffe5/Documents/classes/fall_2015/machine_learning/hw1/data/spambase/spambase.names"
f1 = open(fname1, "r")
for line in f1:
    if line.find('|') == -1:
        if line.strip():
            line = line.split(':')
            feature = line[0]
            desc = line[1] 
            desc = desc.replace(" ","")
            desc = desc.split('.')[0]
            feature_list.append(feature)

# Create dictionary with features as keys
fmat = {}
for feature in feature_list:
    fmat[feature] = []

Y = ""
num_data = 0
fname2 = "/home/jaffe5/Documents/classes/fall_2015/machine_learning/hw1/data/spambase/spambase.data"
f2 = open(fname2, "r")
for line in f2:
    num_data += 1
    x = line.split(',')
    for i in range(len(x)-1):
        x[i] = float(x[i])
        fmat[feature_list[i]].append(x[i])
    Y += x[-1].strip()
Y = BitArray(bin=Y)
#print Y
#print "# data elements:",num_data
data = BitArray(bin=num_data*'1')

thresh = {}
for key in fmat:
    # Form a set containing all unique data elements for each feature
    # This will be the set of thresholds
    thresh[key] = sorted(list(set(fmat[key])))
    thresh[key].append(thresh[key][-1]+1)

def prob(A, Y, spam=False):
    if A.count(True) > 0.0:
        return float((A&Y).count(spam))/float(A.count(True))
    else:
        return 0.0

def entropy(p):
    s = 0.0
    for p_qk in p:
        if p_qk > 0.0:
            s += p_qk*log(p_qk, 2)
    return -s

def info_gain():
    pass

@timing
def build_tree1(fmat, feature_list, thresh):
    B, C = '', ''
    e_list = []
    for f in [feature_list[0]]:
        print f
          
        for t in thresh[f]:
            #print t,
            s = ''
            for i in range(len(fmat[f])):
                if fmat[f][i] < t:
                    s += '1'
                else:
                    s += '0'
            B = BitArray(bin=s)
            #C = B.copy()
            #C.invert()

            p_true = prob(B, Y, spam=True)
            p_false = 1.0 - p_true
            e = entropy([p_true, p_false])
            if e > 0.0:
                e_list.append((f,t,e))

    min_val = min(e[2] for e in e_list)
    print "Min:",[e for e in e_list if e[2] == min_val]

@timing
def build_tree2(fmat, feature_list):
    B, C = '', ''
    e_list = []
    for f in [feature_list[0]]:
        print f

        # Sort the relevant column and keep indices
        pairs = [sorted(enumerate(fmat[f]), key=itemgetter(1))][0]

        s = len(pairs)*'0'
        B = BitArray(bin=s)
        i = 0
        # Threshold emulator loop
        while True:
            B[pairs[i][0]] = 1
            if pairs[i][1] < pairs[i+1][1]:
                t = pairs[i+1][1]
                # Evaluate entropy and yield result
                p_true = prob(B, Y, spam=True)
                p_false = 1.0 - p_true
                e = entropy([p_true, p_false])
                e_list.append((f,t,e))
            i += 1
            # Check end condition
            if i >= len(pairs)-1:
                break

    min_val = min(e[2] for e in e_list)
    print "Min:",[e for e in e_list if e[2] == min_val]

#@timing
def eval_node(A, fmat, feature_list):
    IG_list = []
    A_count = float(A.count(True))
    #print A_count
    p_A = prob(A, Y, spam=True)
    e_A = entropy([p_A, 1.0-p_A])
    for f in feature_list:
        # Sort the relevant column and keep indices
        pairs = [sorted(enumerate(fmat[f]), key=itemgetter(1))][0]

        s = len(pairs)*'0'
        B = BitArray(bin=s)
        C = A.copy()
        i = 0
        IG_prev = 0.0
        # Threshold emulator loop
        while True:
            if A[pairs[i][0]]:
                B[pairs[i][0]] = 1
                C[pairs[i][0]] = 0
                if pairs[i][1] < pairs[i+1][1]:
                    t = pairs[i+1][1]
                    # Calculate probability for child nodes
                    p_B = prob(B, Y, spam=True)
                    p_C = prob(C, Y, spam=True)
                    # Calculate entropy for child nodes
                    e_B = entropy([p_B, 1.0-p_B])
                    e_C = entropy([p_C, 1.0-p_C])
                    # Calculate scaling factors for child nodes entropy
                    k_B = float(B.count(True))/A_count
                    k_C = float(C.count(True))/A_count
                    # Calculate scaled entropy sum for child nodes
                    e_S = k_B*e_B + k_C*e_C
                    # Calculate information gain for the split
                    IG_curr = e_A - e_S
                    #print IG_curr
                    if IG_curr < IG_prev:
                        break
                    else:
                        IG_prev = IG_curr
                    IG_list.append((f,t,IG_curr,B,C))
            i += 1
            # Check end condition
            if i >= len(pairs)-1:
                break

    max_val = max(IG[2] for IG in IG_list)
    f,t,IG,B,C = [IG for IG in IG_list if IG[2] == max_val][0]
    print f,t,IG
    #print (B|C).count(True)
    return B,C

A = data

#build_tree1(fmat, feature_list, thresh)
#build_tree2(fmat, feature_list)

#B,C = eval_node(A, fmat, feature_list)
#eval_node(B, fmat, feature_list)
#eval_node(C, fmat, feature_list)

Node = namedtuple('Node', ['feature', 'thresh', 'decision', 'lchild', 'rchild'], verbose=True)

tree = [A]
d = 4 
for i in range(d):
    index = 2**i
    print "Tree level:",i+1
    for j in range(index, 2*index):
        B,C = eval_node(tree[j-1], fmat, feature_list)
        tree.append(B)
        tree.append(C)
