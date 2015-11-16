import numpy as np

class PollutedSpambase:
    def __init__(self):
        train_data_file = "../data/spam_polluted/train_feature.txt"
        test_data_file = "../data/spam_polluted/test_feature.txt"
        train_label_file = "../data/spam_polluted/train_label.txt"
        test_label_file = "../data/spam_polluted/test_label.txt"

        self.train_data = self.load(train_data_file)
        self.test_data = self.load(test_data_file)
        self.train_labels = self.load(train_label_file)
        self.test_labels = self.load(test_label_file)

    def training(self):
        return self.train_data, self.train_labels

    def testing(self):
        return self.test_data, self.test_labels

    def load(self, fname):
        f = open(fname, "r")
        data = []
        for line in f:
            data.append(line.split())
        return np.array(data).astype(float)
