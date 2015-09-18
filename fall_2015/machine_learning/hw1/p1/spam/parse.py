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


#@timing
def eval_node(A, Y, fmat, feature_list):
    IG_list = []
    A_count = float(A.count(True))
    #A_count = float(A.len)
    p_A = prob(A, Y, spam=True)
    e_A = entropy([p_A, 1.0-p_A])
    #print "P(Q):",p_A
    if p_A >= 0.5:
        d = 1
    else:
        d = 0
    #print e_A*(A_count/4601.0)
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
                    # Calculate scaled entropy for child nodes
                    s_B = k_B*e_B
                    s_C = k_C*e_C
                    # Calculate scaled entropy sum for child nodes
                    e_S = s_B+s_C
                    # Calculate information gain for the split
                    IG_curr = e_A - e_S
                    #print IG_curr
                    if IG_curr < IG_prev:
                        break
                    else:
                        IG_prev = IG_curr
                    # Check if entropy for any branches is below thresh
                    IG_list.append((IG_curr,f,t,d,B,C,s_B,s_C))
            i += 1
            # Check end condition
            if i >= len(pairs)-1:
                break

    max_val = max(IG[0] for IG in IG_list)
    IG,f,t,d,B,C,s_B,s_C = [IG for IG in IG_list if IG[0] == max_val][0]
    #if s_B < 0.2:
    #    B = None
    #if s_C < 0.2:
    #    C = None 
    #print s_B,s_C
    return f,t,d,B,C


TestNode = namedtuple('Node', [ 'f',    # Feature
                                't',    # Threshold
                                'd',    # Decision
                                'lc',   # Left child
                                'rc'])  # Right child


def build_tree(root, Y, fmat, feature_list):
    btree = [root]
    etree = []
    depth = 4 
    for i in range(depth):
        index = 2**i
        print "Tree level:",i+1
        for j in range(index-1, 2*index-1):
            if btree[j] is None:
                btree.append(None)
                btree.append(None)
                etree.append(None)
            else:
                f,t,d,B,C = eval_node(btree[j], Y, fmat, feature_list)
                print f,t,d
                lc,rc = 2*j+1, 2*j+2
                if B is None or (lc >= (2**depth)-1):
                    lc = None
                if C is None or (rc >= (2**depth)-1):
                    rc = None
                etree.append(TestNode(f=f, t=t, d=d, lc=lc, rc=rc)) 
                btree.append(B)
                btree.append(C)
    return etree


def eval_data(t, d):
    c = 0
    while True:
        q = t[c] 
        i = feature_list.index(q.f)
        if d[i] < q.t:
            c = q.lc
            #print "Less:",i, d[i], q.t, c
        else:
            c = q.rc
            #print "Greater:",i, d[i], q.t, c
        if c is None:
            return q.d
        #print c


def eval_tree(dmat, etree, Y):
    total = len(dmat)
    correct = 0
    for i in range(len(dmat)):
        if eval_data(etree, dmat[i]) == Y[i]:
            correct += 1
    return correct,total


if __name__=="__main__":
    # Get feature list
    fname1 = ("/home/jaffe5/Documents/classes/fall_2015/"+
                "machine_learning/hw1/data/spambase/spambase.names")
    feature_list = get_features(fname1)

    # Get data
    fname2 = ("/home/jaffe5/Documents/classes/fall_2015/"
                +"machine_learning/hw1/data/spambase/spambase.data")
    fmat,dmat,Y = get_data(fname2, feature_list)

    # Build the decision tree
    root = BitArray(bin=len(dmat)*'1')
    etree = build_tree(root, Y, fmat, feature_list)

    print "Tree:"
    for i in range(len(etree)):
        if etree[i] is not None:
            print i, etree[i]

    # Evaluate the decision tree
    correct,total = eval_tree(dmat, etree, Y)
    print "Correct:",str(correct)+str('/')+str(total)+' =',float(correct)/float(total)
