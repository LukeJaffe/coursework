import numpy as np

class KFolder:
    def __init__(self, X, k, normalize=True):
        self.k = k
        self.X = []
        np.random.shuffle(X)
        tot_len = len(X)
        fold_len = tot_len/k
        for i in range(0, tot_len, fold_len):
            self.X.append(X[i:i+fold_len])

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

        if normalize:
            # Normalize the data
            for i in range(self.k):
                d = self.data["testing"]["data"][i]
                t = self.data["training"]["data"][i]
                for j in range(len(d)):
                    # Normalize the training data
                    min_val = d[j].min()
                    d[j] -= min_val
                    max_val = d[j].max()
                    d[j] /= max_val
                    # Normalize the testing data with the same parameters
                    t[j] -= min_val
                    t[j] /= max_val
                    # Clip testing data to [0.0, 1.0] 
                    t[j] = np.clip(t[j], 0.0, 1.0)
                #print self.data["testing"]["data"][i].min(), self.data["testing"]["data"][i].max()
                #print self.data["training"]["data"][i].min(), self.data["training"]["data"][i].max()
                #print 

    def training(self, k):
        return self.data["training"]["data"][k].T, self.data["training"]["labels"][k].T

    def testing(self, k):
        return self.data["testing"]["data"][k].T, self.data["testing"]["labels"][k].T
