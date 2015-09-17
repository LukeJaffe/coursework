from bitstring import BitArray
from math import log
from operator import itemgetter

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

A = data
p_true = prob(A, Y, spam=True)
p_false = 1.0 - p_true
print entropy([p_true, p_false])

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
