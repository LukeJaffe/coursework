import numpy as np

from parse import Parser

class UCIFolder:
    def __init__(self, X, k=20, normalize=True, shuffle=True):
        self.k = k
        self.X = []
        if shuffle:
            np.random.shuffle(X)
        tot_len = len(X)
        fold_len = tot_len/k
        for i in range(k):
            self.X.append([])
        for i in range(0, tot_len, k):
            for j in range(i,i+k):
                if j < tot_len:
                    self.X[j%k].append(X[j])
        for i in range(k):
            self.X[i] = np.array(self.X[i])

        # Prepare the data dictionary
        training = {"data":[], "labels":[]}
        testing = {"data":[], "labels":[]}
        self.data = {"training":training,"testing":testing}

        # Prepare training and testing data
        for i in range(self.k):
            self.data["testing"]["data"].append(self.X[i].T[:-1])
            self.data["testing"]["labels"].append(self.X[i].T[-1:])
            tmp = self.X[:i]+self.X[i+1:]
            tmp = np.concatenate(tmp, axis=0)
            d = tmp.T[:-1]
            l = tmp.T[-1:]
            self.data["training"]["data"].append(d)
            self.data["training"]["labels"].append(l)

        if normalize:
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

    def training(self, c):
        k = c/5
        data, labels = None, None
        for i in range(k):
            if data is None and labels is None:
                data = self.data["testing"]["data"][i].T
                labels = self.data["testing"]["labels"][i].T 
            else:
                data = np.append(data, self.data["testing"]["data"][i].T, axis=0)
                labels = np.append(labels, self.data["testing"]["labels"][i].T, axis=0)
        return data, labels

    def testing(self, c=20):
        k = c/5
        data, labels = None, None
        for i in range(self.k-1, self.k-k-1, -1):
            if data is None and labels is None:
                data = self.data["testing"]["data"][i].T
                labels = self.data["testing"]["labels"][i].T 
            else:
                data = np.append(data, self.data["testing"]["data"][i].T, axis=0)
                labels = np.append(labels, self.data["testing"]["labels"][i].T, axis=0)
        return data, labels


if __name__=="__main__":
    vote_parser = Parser('../data/vote/vote.config', '../data/vote/vote.data')
    vote_parser.parse_config()
    D = vote_parser.parse_data()
    print D.shape
    ucifolder = UCIFolder(D, normalize=False, shuffle=False)
    for c in [5,10,15,20,30,50,80]:
        train_data, train_labels = ucifolder.training(c)
        test_data, test_labels = ucifolder.testing()
        print train_data.shape, test_data.shape
