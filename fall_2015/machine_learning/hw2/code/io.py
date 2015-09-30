import numpy as np

class Data:
    def __init__(self, data_file):
        X, Y = [], []
        f = open(data_file, "r")
        for line in f:
            if line.strip():
                x = line.split()
                x = [float(e) for e in x]
                X.append( x[:-1] )
                Y.append( x[-1:] )
        self.X, self.Y = np.array(X), np.array(Y)

    def normalize(self):
        pass

    def data(self):
        return self.X

    def labels(self):
        return self.Y

if __name__=="__main__":
    data_file = "../data/perceptron/perceptronData.txt"
    data = Data(data_file)
    print data.data().shape, data.labels().shape
