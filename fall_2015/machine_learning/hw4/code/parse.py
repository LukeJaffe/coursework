import numpy as np

missing = float('-inf')
cont = float('inf')

class Config:
    def __init__(self, n):
        self.n = n
        self.d = []
        self.c = []

    def label(self, y):
        return self.c.index(y)

    def data(self, f, x):
        if x == '?':
            return missing
        elif not self.d[f]:
            return cont
        else:
            return self.d[f].index(x)


class Parser:
    def __init__(self, config_file, data_file):
        self.config_file = config_file
        self.data_file = data_file

    def parse_config(self):
        fc = open(self.config_file, "r")
        header = fc.next()
        n, d, c = header.split()
        config = Config(n)
        for i in range(int(d)+int(c)):
            config.d.append([])
            feature = fc.next().split()
            if feature[0].startswith('-'):
                config.d[i] = False
            for j in range(1, len(feature)):
                config.d[i].append(feature[j])
        classes = fc.next().split()
        for i in range(1, len(classes)):
            config.c.append(classes[i])
        self.config = config

    def parse_data(self):
        fd = open(self.data_file, "r")
        X, Y = [], []
        for i,line in enumerate(fd):
            line = line.split()
            y = line[-1]
            Y.append(self.config.label(y))
            X.append([])
            for f,x in enumerate(line[:-1]):
                r = self.config.data(f,x)
                if r == cont:
                    X[i].append(float(x))
                else:
                    X[i].append(self.config.data(f, x)) 
        X = np.array(X)
        #print X.mean()
        for f in X.T:
            present = f[f>missing]
            f[f==missing] = np.median(present)
        #print X.mean()
        # Replace all missing values with feature median
        Y = np.array(Y, ndmin=2).T
        #print X.shape, Y.shape
        D = np.append(X, Y, axis=1)
        return D


if __name__=="__main__":
    # Test vote data
    p1 = Parser('../data/vote/vote.config', '../data/vote/vote.data')
    p1.parse_config()
    p1.parse_data()
    # Test crx data
    p2 = Parser('../data/crx/crx.config', '../data/crx/crx.data')
    p2.parse_config()
    p2.parse_data()
