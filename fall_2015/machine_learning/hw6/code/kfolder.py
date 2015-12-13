import numpy as np
from sklearn import preprocessing

class KFolder:
    def __init__(self, X, k, standard=False, minmax=False, shuffle=True):
        self.k = k
        self.X = []

        # Shuffle
        if shuffle:
            np.random.shuffle(X)
        
        # Normalize
        #if normalize:
        #    zmscaler = preprocessing.MinMaxScaler()
        #    X[:-1] = zmscaler.fit_transform(X[:-1])

        tot_len = len(X)
        fold_len = tot_len/k
        #for i in range(0, tot_len, fold_len):
        #    self.X.append(X[i:i+fold_len])
        for i in range(k):
            self.X.append([])
        for i in range(0, tot_len, k):
            for j in range(i,i+k):
                if j < tot_len:
                    self.X[j%k].append(X[j])
        for i in range(k):
            self.X[i] = np.array(self.X[i])
            #print self.X[i].T[-1]

        # Prepare the data dictionary
        training = {"data":[], "labels":[]}
        testing = {"data":[], "labels":[]}
        self.data = {"training":training,"testing":testing}

        # Prepare training and testing data
        for i in range(self.k):
            self.data["testing"]["data"].append(self.X[i].T[:-1])
            self.data["testing"]["labels"].append(self.X[i].T[-1:])
            #print self.data["testing"]["data"][i].shape,self.data["testing"]["labels"][i].shape
            tmp = self.X[:i]+self.X[i+1:]
            tmp = np.concatenate(tmp, axis=0)
            d = tmp.T[:-1]
            l = tmp.T[-1:]
            self.data["training"]["data"].append(d)
            self.data["training"]["labels"].append(l)
            #print self.data["training"]["data"][i].shape,self.data["training"]["labels"][i].shape
            #print

        if minmax:
            # Normalize the data
            for i in range(self.k):
                d = self.data["testing"]["data"][i]
                t = self.data["training"]["data"][i]
                for j in range(len(d)):
                    # Normalize the training data
                    if d[j].min() == d[j].max():
                        min_val = 0.0
                        max_val = d[j].max()
                    else:
                        min_val = d[j].min()
                        d[j] -= min_val
                        max_val = d[j].max()
                        d[j] /= max_val
                    # Normalize the testing data with the same parameters
                    t[j] -= min_val
                    t[j] /= max_val
                    # Clip testing data to [0.0, 1.0] 
                    t[j] = np.clip(t[j], 0.0, 1.0)

        if standard:
            # Normalize the data
            for i in range(self.k):
                d = self.data["testing"]["data"][i]
                t = self.data["training"]["data"][i]
                for j in range(len(d)):
                    # Get normalization params
                    mean = d[j].mean()
                    std = d[j].std()
                    # Normalize the training data
                    d[j] -= mean
                    d[j] /= std
                    # Normalize the testing data
                    t[j] -= mean
                    t[j] /= std
            

    def training(self, k):
        return self.data["training"]["data"][k].T, self.data["training"]["labels"][k].T

    def testing(self, k):
        return self.data["testing"]["data"][k].T, self.data["testing"]["labels"][k].T
