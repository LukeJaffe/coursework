from polluted import PollutedSpambase
from sklearn import linear_model
from sklearn import preprocessing
import numpy as np
import sys
import argparse

if __name__=="__main__":
    # Parse cmdline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help='Method of regularization. {lasso, ridge}')
    args = parser.parse_args(sys.argv[1:])

    # Get data
    dataset = PollutedSpambase()
    train_data, train_labels = dataset.training()
    test_data, test_labels = dataset.testing()

    zmscaler = preprocessing.StandardScaler()
    train_data = zmscaler.fit_transform(train_data)
    test_data = zmscaler.transform(test_data)

    # Ridge
    if args.m == 'ridge':
        clf = linear_model.Ridge(alpha=0.001, max_iter = 1000000)
    elif args.m == 'lasso':
        clf = linear_model.Lasso(alpha=0.001, max_iter=10000, selection='random')
    clf.fit(train_data, train_labels)
    W = clf.coef_.ravel()
    print len(W[W>0])

    print clf.predict(test_data).shape
    s = clf.predict(test_data)

    # Evaluate solution
    correct = 0
    total = len(test_labels)
    for j in range(len(test_data)):
        if s[j] >= 0.5:
            s[j] = 1.0
        else:
            s[j] = 0.0
        if s[j] == test_labels[j]:
            correct += 1
    acc = float(correct)/float(total)
    print acc
